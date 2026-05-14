from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict
import json
import math
import os
import random
import re

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from strategies.atr_regime_filter import ATRFilterParams, run_atr_regime_filter
from strategies.bollinger_strategy import BollingerParams
from strategies.donchian_strategy import DonchianParams, run_donchian_strategy
from strategies.ema_adx_strategy import EMAADXParams, run_ema_adx_strategy
from strategies.mean_reversion_strategy import MeanReversionParams, run_mean_reversion_strategy
from strategies.zscore_strategy import ZScoreParams, run_zscore_strategy
from trading.notifier import send_message
from trading.funding_rate import fetch_funding_rate, annualized_from_rate, FundingRateError
from strategies.btc_news_summary import get_gemini_news
from strategies.fear_gread import get_fgi


load_dotenv()


@dataclass
class GPTParams:
    provider: str = "openai"
    model: str = "gpt-5-mini"
    mode: str = "live"
    context_hours: int = 4
    weight_sma: float = 1.0
    weight_rsi: float = 1.0
    weight_macd: float = 0.35
    weight_bollinger: float = 0.9
    weight_ema_adx: float = 1.2
    weight_donchian: float = 1.1
    weight_mean_reversion: float = 1.1
    weight_zscore: float = 1.3
    vote_threshold: int = 2
    exit_vote_threshold: int = 1
    hour_momentum_threshold: float = 0.002
    show_prompt: bool = False
    min_confidence: int = 60
    gpt_authority: float = 0.30
    gpt_trigger_gap: float = 12.0
    gpt_trigger_score: float = 58.0
    atr_block_threshold: float = 0.85
    trend_regime_threshold: float = 0.55
    breakout_regime_threshold: float = 0.60
    range_regime_threshold: float = 0.60
    trend_following_boost: float = 1.25
    breakout_boost: float = 1.20
    mean_reversion_boost: float = 1.25
    off_regime_penalty: float = 0.72


def _load_best_params() -> dict:
    path = Path(__file__).resolve().parents[1] / "research" / "best_params.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    except Exception:
        return default


def _safe_last(series: pd.Series | None, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return default
    return _safe_float(series.iloc[-1], default)


def _tf_to_minutes(tf: str) -> int:
    tf = (tf or "15m").strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 1440
    return 15


def _parse_performance(raw: Any) -> float | None:
    if raw is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(raw))
    if not match:
        return None
    return float(match.group(0))


def _performance_multiplier(perf: float | None) -> float:
    if perf is None:
        return 0.90
    if perf >= 200:
        return 1.25
    if perf >= 75:
        return 1.10
    if perf >= 0:
        return 1.00
    if perf >= -20:
        return 0.80
    return 0.45


def _fit_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _build_backtest_label(best_map: dict, symbol: str, timeframe: str, strat: str) -> str:
    key = f"{symbol}_{timeframe}_{strat}"
    backtest_entry = best_map.get("__backtest", {}).get(key)
    if not isinstance(backtest_entry, dict):
        return "NA"
    performance = backtest_entry.get("performance", "NA")
    period = backtest_entry.get("period", "NA")
    return f"{performance} over {period}"


def _market_features(ohlcv: pd.DataFrame, timeframe: str, context_hours: int) -> dict[str, Any]:
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    tf_minutes = _tf_to_minutes(timeframe)
    bars = max(16, (context_hours * 60) // max(tf_minutes, 1))
    ctx = ohlcv.tail(bars).copy()

    atr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1).rolling(14).mean()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    returns = close.pct_change()

    def _ret(period_bars: int) -> float:
        if len(close) <= period_bars:
            return 0.0
        base = close.iloc[-period_bars - 1]
        if base == 0:
            return 0.0
        return (close.iloc[-1] / base) - 1

    volume_ma = volume.rolling(30).mean()
    volume_std = volume.rolling(30).std(ddof=0)
    volume_z = (volume - volume_ma) / volume_std.replace(0, pd.NA)

    lookback_high = ctx["high"].max()
    lookback_low = ctx["low"].min()
    channel_span = max(lookback_high - lookback_low, 1e-9)
    breakout_position = (close.iloc[-1] - lookback_low) / channel_span

    trend_strength = _clamp(
        abs(_safe_last(ema20 - ema50)) / max(close.iloc[-1], 1e-9) * 900,
        0,
        1,
    )
    atr_pct = _safe_last(atr / close.replace(0, pd.NA))
    atr_rank = _safe_float(
        (atr / close.replace(0, pd.NA)).rolling(96).apply(
            lambda values: pd.Series(values).rank(pct=True).iloc[-1],
            raw=False,
        ).iloc[-1],
        0.5,
    )
    range_bias = _clamp(1 - trend_strength + max(0.0, 0.55 - atr_rank), 0, 1)
    breakout_fit = _clamp((breakout_position * 0.6) + max(0.0, _safe_last(volume_z) / 3.0), 0, 1)

    return {
        "last_close": round(_safe_float(close.iloc[-1]), 2),
        "return_1h_pct": round(_ret(max(1, 60 // tf_minutes)) * 100, 2),
        "return_4h_pct": round(_ret(max(1, 240 // tf_minutes)) * 100, 2),
        "return_24h_pct": round(_ret(max(1, 1440 // tf_minutes)) * 100, 2),
        "ema20_distance_pct": round((_safe_last(close) / max(_safe_last(ema20, 1.0), 1e-9) - 1) * 100, 2),
        "ema50_distance_pct": round((_safe_last(close) / max(_safe_last(ema50, 1.0), 1e-9) - 1) * 100, 2),
        "ema200_distance_pct": round((_safe_last(close) / max(_safe_last(ema200, 1.0), 1e-9) - 1) * 100, 2),
        "atr_pct": round(atr_pct * 100, 3),
        "atr_rank": round(atr_rank, 3),
        "volume_zscore": round(_safe_last(volume_z), 2),
        "trend_strength": round(trend_strength, 3),
        "range_bias": round(range_bias, 3),
        "breakout_position": round(breakout_position, 3),
        "breakout_fit": round(breakout_fit, 3),
    }


def _strategy_summary(
    *,
    name: str,
    category: str,
    signal: str,
    confidence: float,
    reason: str,
    backtest_performance: str,
    regime_score: float,
    weight: float,
) -> dict[str, Any]:
    return {
        "signal": signal,
        "confidence": int(round(_clamp(confidence))),
        "reason": reason[:160],
        "backtest_performance": backtest_performance,
        "regime_fit": f"{_fit_label(regime_score)} ({regime_score:.2f})",
        "category": category,
        "weight": round(weight, 2),
        "regime_score": round(regime_score, 3),
        "name": name,
    }


def _regime_profile(market: dict[str, Any], params: GPTParams) -> dict[str, Any]:
    trend_strength = _safe_float(market.get("trend_strength"), 0.0)
    breakout_fit = _safe_float(market.get("breakout_fit"), 0.0)
    range_bias = _safe_float(market.get("range_bias"), 0.0)
    atr_rank = _safe_float(market.get("atr_rank"), 0.5)

    label = "mixed"
    if breakout_fit >= params.breakout_regime_threshold and trend_strength >= params.trend_regime_threshold * 0.9:
        label = "breakout"
    elif trend_strength >= params.trend_regime_threshold and atr_rank < params.atr_block_threshold:
        label = "trend"
    elif range_bias >= params.range_regime_threshold:
        label = "range"

    multipliers = {
        "sma": 1.0,
        "rsi": 1.0,
        "macd": 0.85,
        "bollinger": 1.0,
        "ema_adx": 1.0,
        "donchian": 1.0,
        "mean_reversion": 1.0,
        "zscore": 1.0,
    }

    if label == "trend":
        multipliers.update(
            {
                "sma": 1.10,
                "ema_adx": params.trend_following_boost,
                "donchian": 1.10,
                "rsi": params.off_regime_penalty,
                "bollinger": params.off_regime_penalty,
                "mean_reversion": params.off_regime_penalty,
                "zscore": params.off_regime_penalty,
            }
        )
    elif label == "breakout":
        multipliers.update(
            {
                "ema_adx": 1.10,
                "donchian": params.breakout_boost,
                "sma": 1.05,
                "rsi": params.off_regime_penalty,
                "bollinger": params.off_regime_penalty,
                "mean_reversion": params.off_regime_penalty,
                "zscore": params.off_regime_penalty,
            }
        )
    elif label == "range":
        multipliers.update(
            {
                "rsi": params.mean_reversion_boost,
                "mean_reversion": params.mean_reversion_boost,
                "bollinger": 1.10,
                "zscore": params.mean_reversion_boost,
                "sma": params.off_regime_penalty,
                "ema_adx": params.off_regime_penalty,
                "donchian": params.off_regime_penalty,
            }
        )

    return {
        "label": label,
        "multipliers": multipliers,
        "reason": (
            f"trend={trend_strength:.2f}, breakout={breakout_fit:.2f}, "
            f"range={range_bias:.2f}, atr_rank={atr_rank:.2f}"
        ),
    }


def _summaries_from_outputs(
    symbol: str,
    timeframe: str,
    ohlcv: pd.DataFrame,
    other_results: Dict[str, Dict[str, pd.Series]],
    params: GPTParams,
    best_map: dict,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    close = ohlcv["close"]
    market = _market_features(ohlcv, timeframe, params.context_hours)
    regime = _regime_profile(market, params)
    market["regime_label"] = regime["label"]
    market["regime_reason"] = regime["reason"]

    best_map_local = _load_best_params()

    def _best_p(strat: str, defaults: dict) -> dict:
        key = f"{symbol}_{timeframe}_{strat}"
        p = best_map_local.get(key)
        return p if isinstance(p, dict) else defaults

    ema_adx_p = _best_p("ema_adx", {"fast": 21, "slow": 55, "adx_window": 14, "adx_threshold": 18.0})
    mr_p = _best_p("mean_reversion", {"rsi_window": 14, "rsi_lower": 32.0, "rsi_upper": 68.0, "bb_window": 20, "bb_std": 2.2})
    zs_p = _best_p("zscore", {"window": 100, "entry_z": 1.5, "exit_z": 0.5})

    ema_adx_out = run_ema_adx_strategy(ohlcv, EMAADXParams(**ema_adx_p))
    donchian_out = run_donchian_strategy(ohlcv, DonchianParams())
    mean_rev_out = run_mean_reversion_strategy(close, MeanReversionParams(**mr_p))
    zscore_out = run_zscore_strategy(close, ZScoreParams(**zs_p))
    atr_filter_out = run_atr_regime_filter(ohlcv, ATRFilterParams(max_percentile=params.atr_block_threshold))

    all_outputs: dict[str, Dict[str, Any]] = dict(other_results)
    all_outputs["ema_adx"] = ema_adx_out
    all_outputs["donchian"] = donchian_out
    all_outputs["mean_reversion"] = mean_rev_out
    all_outputs["zscore"] = zscore_out
    all_outputs["atr_filter"] = atr_filter_out

    weights = {
        "sma": params.weight_sma,
        "rsi": params.weight_rsi,
        "macd": params.weight_macd,
        "bollinger": params.weight_bollinger,
        "ema_adx": params.weight_ema_adx,
        "donchian": params.weight_donchian,
        "mean_reversion": params.weight_mean_reversion,
        "zscore": params.weight_zscore,
    }

    summaries: dict[str, dict[str, Any]] = {}

    fast_ma = all_outputs["sma"]["indicators"]["fast_ma"]
    slow_ma = all_outputs["sma"]["indicators"]["slow_ma"]
    sma_spread_pct = abs(_safe_last(fast_ma) - _safe_last(slow_ma)) / max(_safe_last(close, 1.0), 1e-9) * 100
    sma_signal = "BUY" if _safe_last(fast_ma) > _safe_last(slow_ma) else "SELL"
    sma_conf = 48 + sma_spread_pct * 150
    if sma_spread_pct < 0.15:
        sma_signal = "HOLD"
        sma_conf = 44
    summaries["sma"] = _strategy_summary(
        name="SMA",
        category="trend",
        signal=sma_signal,
        confidence=sma_conf,
        reason=f"Fast/slow MA spread {sma_spread_pct:.2f}% with trend state {sma_signal}.",
        backtest_performance=_build_backtest_label(best_map, symbol, timeframe, "sma"),
        regime_score=market["trend_strength"],
        weight=weights["sma"],
    )

    rsi_series = all_outputs["rsi"]["indicators"]["rsi"]
    rsi_value = _safe_last(rsi_series, 50)
    rsi_params = all_outputs["rsi"].get("params", {})
    rsi_lower = _safe_float(rsi_params.get("lower"), 30)
    rsi_upper = _safe_float(rsi_params.get("upper"), 70)
    if rsi_value < rsi_lower:
        rsi_signal = "BUY"
        rsi_conf = 55 + (rsi_lower - rsi_value) * 1.8
    elif rsi_value > rsi_upper:
        rsi_signal = "SELL"
        rsi_conf = 55 + (rsi_value - rsi_upper) * 1.8
    else:
        rsi_signal = "HOLD"
        rsi_conf = 42 + abs(rsi_value - 50) * 0.2
    summaries["rsi"] = _strategy_summary(
        name="RSI",
        category="mean_reversion",
        signal=rsi_signal,
        confidence=rsi_conf,
        reason=f"RSI={rsi_value:.1f} vs thresholds {rsi_lower:.0f}/{rsi_upper:.0f}.",
        backtest_performance=_build_backtest_label(best_map, symbol, timeframe, "rsi"),
        regime_score=market["range_bias"],
        weight=weights["rsi"],
    )

    macd_line = all_outputs["macd"]["indicators"]["macd"]
    signal_line = all_outputs["macd"]["indicators"]["signal"]
    macd_diff = _safe_last(macd_line) - _safe_last(signal_line)
    macd_signal = "BUY" if macd_diff > 0 else "SELL"
    macd_conf = 38 + min(abs(macd_diff) / max(_safe_last(close, 1.0), 1e-9) * 60000, 28)
    if abs(macd_diff) / max(_safe_last(close, 1.0), 1e-9) < 0.0003:
        macd_signal = "HOLD"
        macd_conf = 35
    summaries["macd"] = _strategy_summary(
        name="MACD",
        category="trend",
        signal=macd_signal,
        confidence=macd_conf,
        reason=f"MACD spread={macd_diff:.2f}; kept low weight due weak historical reliability.",
        backtest_performance=_build_backtest_label(best_map, symbol, timeframe, "macd"),
        regime_score=max(market["trend_strength"], 0.35),
        weight=weights["macd"],
    )

    boll_lower = all_outputs["bollinger"]["indicators"]["lower"]
    boll_upper = all_outputs["bollinger"]["indicators"]["upper"]
    close_last = _safe_last(close)
    lower_last = _safe_last(boll_lower, close_last)
    upper_last = _safe_last(boll_upper, close_last)
    if close_last < lower_last:
        boll_signal = "BUY"
        boll_conf = 54 + (lower_last - close_last) / max(close_last, 1e-9) * 1000
    elif close_last > upper_last:
        boll_signal = "SELL"
        boll_conf = 54 + (close_last - upper_last) / max(close_last, 1e-9) * 1000
    else:
        boll_signal = "HOLD"
        boll_conf = 40
    summaries["bollinger"] = _strategy_summary(
        name="Bollinger",
        category="mean_reversion",
        signal=boll_signal,
        confidence=boll_conf,
        reason=f"Price vs bands: close={close_last:.2f}, lower={lower_last:.2f}, upper={upper_last:.2f}.",
        backtest_performance=_build_backtest_label(best_map, symbol, timeframe, "bollinger"),
        regime_score=market["range_bias"],
        weight=weights["bollinger"],
    )

    ema_fast = ema_adx_out["indicators"]["fast_ema"]
    ema_slow = ema_adx_out["indicators"]["slow_ema"]
    adx_series = ema_adx_out["indicators"]["adx"]
    adx_value = _safe_last(adx_series)
    ema_adx_signal = "HOLD"
    ema_adx_conf = 45
    if adx_value >= 18:
        if _safe_last(ema_fast) > _safe_last(ema_slow):
            ema_adx_signal = "BUY"
            ema_adx_conf = 54 + (adx_value - 18) * 1.4
        elif _safe_last(ema_fast) < _safe_last(ema_slow):
            ema_adx_signal = "SELL"
            ema_adx_conf = 54 + (adx_value - 18) * 1.4
    summaries["ema_adx"] = _strategy_summary(
        name="EMA+ADX",
        category="trend",
        signal=ema_adx_signal,
        confidence=ema_adx_conf,
        reason=f"EMA state with ADX={adx_value:.1f}; trend filter requires ADX>=18.",
        backtest_performance=_build_backtest_label(best_map, symbol, timeframe, "ema_adx"),
        regime_score=max(market["trend_strength"], market["breakout_fit"] * 0.8),
        weight=weights["ema_adx"],
    )

    donchian_upper = donchian_out["indicators"]["upper"]
    donchian_lower = donchian_out["indicators"]["lower"]
    upper_last = _safe_last(donchian_upper, close_last)
    lower_last = _safe_last(donchian_lower, close_last)
    if close_last > upper_last:
        donchian_signal = "BUY"
        donchian_conf = 58 + (close_last - upper_last) / max(close_last, 1e-9) * 1200
    elif close_last < lower_last:
        donchian_signal = "SELL"
        donchian_conf = 58 + (lower_last - close_last) / max(close_last, 1e-9) * 1200
    else:
        donchian_signal = "HOLD"
        donchian_conf = 42
    summaries["donchian"] = _strategy_summary(
        name="Donchian",
        category="breakout",
        signal=donchian_signal,
        confidence=donchian_conf,
        reason=f"Close within breakout channel [{lower_last:.2f}, {upper_last:.2f}].",
        backtest_performance=_build_backtest_label(best_map, symbol, timeframe, "donchian"),
        regime_score=max(market["breakout_fit"], market["trend_strength"] * 0.8),
        weight=weights["donchian"],
    )

    mean_rev_rsi = mean_rev_out["indicators"]["rsi"]
    mean_rev_z = mean_rev_out["indicators"]["zscore"]
    mr_rsi = _safe_last(mean_rev_rsi, 50)
    mr_z = _safe_last(mean_rev_z, 0)
    mr_lower = _safe_last(mean_rev_out["indicators"]["lower"], close_last)
    mr_upper = _safe_last(mean_rev_out["indicators"]["upper"], close_last)
    if close_last < mr_lower and mr_rsi < 35:
        mr_signal = "BUY"
        mr_conf = 58 + abs(mr_z) * 8 + max(0.0, 35 - mr_rsi)
    elif close_last > mr_upper and mr_rsi > 65:
        mr_signal = "SELL"
        mr_conf = 58 + abs(mr_z) * 8 + max(0.0, mr_rsi - 65)
    else:
        mr_signal = "HOLD"
        mr_conf = 43
    summaries["mean_reversion"] = _strategy_summary(
        name="RSI+Bollinger",
        category="mean_reversion",
        signal=mr_signal,
        confidence=mr_conf,
        reason=f"Enhanced MR uses RSI={mr_rsi:.1f} and z-score={mr_z:.2f}.",
        backtest_performance=_build_backtest_label(best_map, symbol, timeframe, "mean_reversion"),
        regime_score=market["range_bias"],
        weight=weights["mean_reversion"],
    )

    zs_series = zscore_out["indicators"]["zscore"]
    zs_value = _safe_last(zs_series, 0.0)
    zs_entry = _safe_float(zs_p.get("entry_z"), 1.5)
    zs_exit = _safe_float(zs_p.get("exit_z"), 0.5)
    if zs_value < -zs_entry:
        zs_signal = "BUY"
        zs_conf = 55 + abs(zs_value + zs_entry) * 12
    elif zs_value > zs_entry:
        zs_signal = "SELL"
        zs_conf = 55 + abs(zs_value - zs_entry) * 12
    elif abs(zs_value) < zs_exit:
        zs_signal = "HOLD"
        zs_conf = 50
    else:
        zs_signal = "HOLD"
        zs_conf = 42
    summaries["zscore"] = _strategy_summary(
        name="Z-Score",
        category="mean_reversion",
        signal=zs_signal,
        confidence=zs_conf,
        reason=f"Z-score={zs_value:.2f} vs thresholds ±{zs_entry}; suited for ratio/range markets.",
        backtest_performance=_build_backtest_label(best_map, symbol, timeframe, "zscore"),
        regime_score=market["range_bias"],
        weight=weights["zscore"],
    )

    allow_trade = bool(atr_filter_out["indicators"]["allow_trade"].iloc[-1])
    atr_pct = _safe_last(atr_filter_out["indicators"]["atr_pct"]) * 100
    atr_rank = _safe_last(atr_filter_out["indicators"]["atr_rank"], 1.0)
    summaries["atr_filter"] = {
        "signal": "ALLOW" if allow_trade else "HOLD",
        "confidence": 95 if not allow_trade else 70,
        "reason": (
            f"ATR noise filter allows trading; ATR={atr_pct:.3f}% rank={atr_rank:.2f}."
            if allow_trade
            else f"ATR noise too high; force HOLD with ATR={atr_pct:.3f}% rank={atr_rank:.2f}."
        ),
        "backtest_performance": "NA",
        "regime_fit": "risk_gate",
        "category": "risk",
        "weight": 0.0,
        "regime_score": 1.0 if allow_trade else 0.0,
        "name": "ATR filter",
    }

    for name, summary in summaries.items():
        if name == "atr_filter":
            continue
        base_weight = _safe_float(summary.get("weight"), 1.0)
        regime_multiplier = _safe_float(regime["multipliers"].get(name), 1.0)
        summary["base_weight"] = round(base_weight, 2)
        summary["regime_multiplier"] = round(regime_multiplier, 2)
        summary["weight"] = round(base_weight * regime_multiplier, 2)
        summary["regime_fit"] = f"{summary['regime_fit']} | regime={regime['label']}"

    return summaries, market


def _position_state(symbol: str) -> dict[str, Any]:
    try:
        from trading.order_executor import get_balances, assets_from_symbol

        assets = assets_from_symbol(symbol)
        balances = get_balances(assets=assets)
        base_asset = assets[0]
        return {
            "balances": balances,
            "in_position": _safe_float(balances.get(base_asset, 0.0)) > 0,
        }
    except Exception:
        return {"balances": {}, "in_position": False}


def _local_decision(
    summaries: dict[str, dict[str, Any]],
    market: dict[str, Any],
    params: GPTParams,
    in_position: bool,
) -> dict[str, Any]:
    buy_raw = 0.0
    sell_raw = 0.0
    max_possible = 0.0
    buy_count = 0
    sell_count = 0

    for name, summary in summaries.items():
        if name == "atr_filter":
            continue

        weight = _safe_float(summary.get("weight"), 0.0)
        perf = _parse_performance(summary.get("backtest_performance"))
        perf_mult = _performance_multiplier(perf)
        regime_score = _safe_float(summary.get("regime_score"), 0.5)
        confidence = _safe_float(summary.get("confidence"), 0.0)
        contribution = weight * perf_mult * regime_score * (confidence / 100.0)
        max_possible += weight * max(perf_mult, 0.6)

        if summary.get("signal") == "BUY":
            buy_raw += contribution
            buy_count += 1
        elif summary.get("signal") == "SELL":
            sell_raw += contribution
            sell_count += 1

    divisor = max(max_possible, 1e-9)
    buy_score = _clamp(100 * buy_raw / divisor)
    sell_score = _clamp(100 * sell_raw / divisor)

    risk_blocked = summaries["atr_filter"]["signal"] != "ALLOW"
    allowed_actions = ["BUY", "HOLD", "SELL"]

    if not in_position:
        sell_score = 0.0
        allowed_actions = ["BUY", "HOLD"]

    gap = abs(buy_score - sell_score)
    mixed_signals = buy_count > 0 and sell_count > 0

    if risk_blocked:
        hold_score = 96.0
        local_action = "HOLD"
        local_conf = hold_score
        rationale = summaries["atr_filter"]["reason"]
    else:
        hold_score = _clamp(50 + (params.min_confidence - max(buy_score, sell_score)) + (10 if gap < params.gpt_trigger_gap else 0))
        top_action = "BUY" if buy_score >= sell_score else "SELL"
        top_score = max(buy_score, sell_score)

        if top_score < params.min_confidence or gap < 6:
            local_action = "HOLD"
            local_conf = max(hold_score, params.min_confidence - gap)
            rationale = f"Local ensemble is inconclusive: buy={buy_score:.1f}, sell={sell_score:.1f}."
        else:
            local_action = top_action
            local_conf = top_score
            rationale = f"Local ensemble favors {top_action}: buy={buy_score:.1f}, sell={sell_score:.1f}."

    should_call_gpt = (
        not risk_blocked
        and (
            mixed_signals
            or gap <= params.gpt_trigger_gap
            or max(buy_score, sell_score) < params.gpt_trigger_score
            or local_action == "HOLD"
        )
    )

    return {
        "action": local_action,
        "confidence": int(round(_clamp(local_conf))),
        "reason": rationale,
        "scores": {
            "BUY": round(buy_score, 2),
            "SELL": round(sell_score, 2),
            "HOLD": round(hold_score, 2),
        },
        "allowed_actions": allowed_actions,
        "mixed_signals": mixed_signals,
        "should_call_gpt": should_call_gpt,
        "risk_blocked": risk_blocked,
        "market_snapshot": market,
        "regime_label": market.get("regime_label", "mixed"),
    }


def _funding_summary(symbol: str) -> str:
    try:
        funding_data = fetch_funding_rate(symbol)
        period_pct = funding_data["rate"] * 100
        annual_pct = annualized_from_rate(funding_data["rate"]) * 100
        next_time = funding_data["next_funding_time"].isoformat()
        return f"Funding 8h={period_pct:.4f}% annualized={annual_pct:.2f}% next={next_time}"
    except FundingRateError as exc:
        return f"Funding unavailable: {exc}"
    except Exception as exc:
        return f"Funding fetch error: {exc}"


def _gpt_prompt(
    symbol: str,
    timeframe: str,
    summaries: dict[str, dict[str, Any]],
    market: dict[str, Any],
    local_view: dict[str, Any],
    position_state: dict[str, Any],
) -> str:
    news = get_gemini_news()
    fgi = get_fgi()

    structured_cards = {
        name: {
            "signal": summary["signal"],
            "confidence": summary["confidence"],
            "reason": summary["reason"],
            "backtest_performance": summary["backtest_performance"],
            "regime_fit": summary["regime_fit"],
            "category": summary["category"],
            "weight": summary["weight"],
            "base_weight": summary.get("base_weight", summary["weight"]),
            "regime_multiplier": summary.get("regime_multiplier", 1.0),
        }
        for name, summary in summaries.items()
    }

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "allowed_actions": local_view["allowed_actions"],
        "gpt_authority": "Approximately 30% of final decision weight. Override only when local evidence is mixed or clearly misspecified.",
        "local_engine": {
            "action": local_view["action"],
            "confidence": local_view["confidence"],
            "reason": local_view["reason"],
            "scores": local_view["scores"],
            "regime": local_view.get("regime_label", "mixed"),
        },
        "position": position_state,
        "market_features": market,
        "sentiment": {
            "fear_greed": fgi,
            "funding": _funding_summary(symbol),
            "news_summary": news,
        },
        "strategies": structured_cards,
        "instructions": [
            "You are the arbiter, not the sole trader.",
            "Respect the allowed_actions list strictly.",
            "If ATR risk gate blocks trading, answer HOLD.",
            "Only overturn the local engine when the structured evidence materially supports it.",
            "Do not invent raw price action beyond the supplied summaries.",
            "Return exactly one JSON object with keys action, confidence, reason.",
        ],
        "output_schema": {
            "action": "BUY|SELL|HOLD",
            "confidence": "0-100",
            "reason": "<=160 chars, concrete, references structured evidence",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _query_openai(prompt: str, model: str, show: bool) -> dict[str, Any]:
    if show:
        print("\n=== GPT Arbiter Prompt ===")
        print(prompt)
        print("==========================\n")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [
        {
            "role": "system",
            "content": (
                "You arbitrate between structured trading signals. "
                "Be conservative, obey allowed_actions, and answer JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )

    full_response = (resp.choices[0].message.content or "").strip()
    parsed = _extract_json(full_response) or {}
    action = str(parsed.get("action", "HOLD")).upper()
    confidence = int(round(_clamp(_safe_float(parsed.get("confidence"), 50))))
    reason = str(parsed.get("reason", "No reason provided."))[:160]

    print(f"\n=== GPT Arbiter Response ===\n{full_response}\n============================\n")
    send_message(f"\n=== GPT Arbiter Response ===\n{full_response}\n============================\n")

    return {
        "action": action if action in {"BUY", "SELL", "HOLD"} else "HOLD",
        "confidence": confidence,
        "reason": reason,
        "raw_response": full_response,
    }


def _blend_with_gpt(local_view: dict[str, Any], gpt_view: dict[str, Any] | None, params: GPTParams) -> dict[str, Any]:
    local_scores = dict(local_view["scores"])
    allowed = set(local_view["allowed_actions"])

    if not gpt_view:
        final_action = local_view["action"]
        return {
            "action": final_action,
            "confidence": int(round(_clamp(local_scores.get(final_action, local_view["confidence"])))),
            "reason": local_view["reason"],
            "blended_scores": local_scores,
            "gpt_used": False,
        }

    gpt_action = gpt_view["action"] if gpt_view["action"] in allowed else "HOLD"
    gpt_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
    gpt_scores[gpt_action] = gpt_view["confidence"]

    authority = max(0.0, min(1.0, params.gpt_authority))
    blended_scores = {
        side: round((1 - authority) * _safe_float(local_scores.get(side, 0.0)) + authority * gpt_scores.get(side, 0.0), 2)
        for side in ("BUY", "SELL", "HOLD")
    }

    if "SELL" not in allowed:
        blended_scores["SELL"] = 0.0

    ranked = sorted(blended_scores.items(), key=lambda item: item[1], reverse=True)
    final_action, final_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if final_action not in allowed:
        final_action = "HOLD"
        final_score = blended_scores["HOLD"]

    if final_score < params.min_confidence or (final_score - runner_up_score) < 4:
        final_action = "HOLD"
        final_score = max(blended_scores["HOLD"], final_score)

    reason = local_view["reason"]
    if final_action == local_view["action"]:
        reason = f"{local_view['reason']} GPT review did not materially overturn the local edge."
    else:
        reason = (
            f"Local engine={local_view['action']} ({local_view['confidence']}) vs GPT={gpt_action} "
            f"({gpt_view['confidence']}); blended decision moved to {final_action}."
        )

    return {
        "action": final_action,
        "confidence": int(round(_clamp(final_score))),
        "reason": reason[:160],
        "blended_scores": blended_scores,
        "gpt_used": True,
    }


def run_gpt_strategy(
    symbol: str,
    ohlcv: pd.DataFrame,
    other_results: Dict[str, Dict[str, pd.Series]],
    timeframe: str,
    params: GPTParams,
    initial_run: bool = False,
) -> Dict[str, Any]:
    del initial_run

    best_map = _load_best_params()
    position_state = _position_state(symbol)
    summaries, market = _summaries_from_outputs(symbol, timeframe, ohlcv, other_results, params, best_map)
    local_view = _local_decision(summaries, market, params, position_state["in_position"])

    gpt_view = None
    provider = (params.provider or "openai").lower()
    if local_view["should_call_gpt"] and provider == "openai" and os.getenv("OPENAI_API_KEY"):
        prompt = _gpt_prompt(symbol, timeframe, summaries, market, local_view, position_state)
        gpt_view = _query_openai(prompt, params.model, params.show_prompt)
    elif local_view["should_call_gpt"] and provider != "openai":
        gpt_action = random.choice(local_view["allowed_actions"])
        gpt_view = {
            "action": gpt_action,
            "confidence": random.randint(45, 75),
            "reason": "Mock arbiter output.",
            "raw_response": "mock",
        }

    blended = _blend_with_gpt(local_view, gpt_view, params)

    idx = ohlcv.index if isinstance(ohlcv.index, pd.Index) else pd.RangeIndex(len(ohlcv))
    entries = pd.Series(False, index=idx, name="entries")
    exits = pd.Series(False, index=idx, name="exits")

    return {
        "entries": entries,
        "exits": exits,
        "params": asdict(params),
        "stats": {
            "decisions_count": 1,
            "local_scores": local_view["scores"],
            "blended_scores": blended["blended_scores"],
            "gpt_used": blended["gpt_used"],
        },
        "last_signal": blended["action"],
        "confidence": blended["confidence"],
        "reason": blended["reason"],
        "local_view": local_view,
        "gpt_view": gpt_view,
        "strategy_summaries": summaries,
        "market_features": market,
        "regime_label": market.get("regime_label", "mixed"),
        "position_state": position_state,
    }
