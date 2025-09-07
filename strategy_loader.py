# strategy_loader.py

from __future__ import annotations

from typing import Callable, Dict, Any
from pathlib import Path
import json
import yaml
import pandas as pd

# base strategies
from strategies.sma_strategy import SMAParams, run_sma_strategy
from strategies.rsi_strategy import RSIParams, run_rsi_strategy
from strategies.macd_strategy import MACDParams, run_macd_strategy
from strategies.bollinger_strategy import BollingerParams, run_bollinger_strategy

# GPT meta
from strategies.gpt_strategy import GPTParams as GPTCfg, run_gpt_strategy

ROOT_DIR = Path(__file__).resolve().parent
BEST_PATH = ROOT_DIR / "research" / "best_params.json"


def _load_yaml(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _load_best_params() -> dict:
    if BEST_PATH.exists():
        try:
            with open(BEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _best_or_default(best_map: dict, symbol: str, timeframe: str, strat: str, defaults: dict) -> dict:
    key = f"{symbol}_{timeframe}_{strat}"
    params = best_map.get(key)
    if isinstance(params, dict):
        return params
    return defaults


def _signals_from_last_bar(out: Dict[str, pd.Series]) -> str:
    """Convert a strategy's entries/exits on the last closed bar to BUY/SELL/HOLD."""
    try:
        if bool(out["entries"].iloc[-1]):
            return "BUY"
        if bool(out["exits"].iloc[-1]):
            return "SELL"
    except Exception:
        pass
    return "HOLD"


def _build_basic_runner(strat_name: str, symbol: str, timeframe: str, best_map: dict) -> Callable:
    """
    Return a callable(df: DataFrame) -> Dict containing entries/exits/params/last_signal
    for non-GPT strategies.
    """
    strat_name = strat_name.lower()

    # defaults for each strategy
    if strat_name == "sma":
        defaults = {"fast": 10, "slow": 50}
        def _runner(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
            close = df["close"]
            p = _best_or_default(best_map, symbol, timeframe, "sma", defaults)
            out = run_sma_strategy(close, SMAParams(**p))
            return {"entries": out["entries"], "exits": out["exits"], "params": p, "last_signal": _signals_from_last_bar(out)}
        return _runner

    if strat_name == "rsi":
        defaults = {"window": 14, "lower": 30, "upper": 70}
        def _runner(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
            close = df["close"]
            p = _best_or_default(best_map, symbol, timeframe, "rsi", defaults)
            out = run_rsi_strategy(close, RSIParams(**p))
            return {"entries": out["entries"], "exits": out["exits"], "params": p, "last_signal": _signals_from_last_bar(out)}
        return _runner

    if strat_name == "macd":
        defaults = {"fast": 12, "slow": 26, "signal": 9}
        def _runner(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
            close = df["close"]
            p = _best_or_default(best_map, symbol, timeframe, "macd", defaults)
            out = run_macd_strategy(close, MACDParams(**p))
            return {"entries": out["entries"], "exits": out["exits"], "params": p, "last_signal": _signals_from_last_bar(out)}
        return _runner

    if strat_name == "bollinger":
        defaults = {"window": 20, "std": 2}
        def _runner(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
            close = df["close"]
            p = _best_or_default(best_map, symbol, timeframe, "bollinger", defaults)
            out = run_bollinger_strategy(close, BollingerParams(**p))
            return {"entries": out["entries"], "exits": out["exits"], "params": p, "last_signal": _signals_from_last_bar(out)}
        return _runner

    raise ValueError(f"Unknown strategy: {strat_name}")


def _build_gpt_runner(symbol: str, timeframe: str, best_map: dict, gpt_cfg: dict) -> Callable:
    """
    Return a callable(df, initial_run=False) that:
      - computes base strategies on the same last closed bar
      - calls run_gpt_strategy (live-only)
    """
    # load base params (best-or-default)
    sma_defaults = {"fast": 10, "slow": 50}
    rsi_defaults = {"window": 14, "lower": 30, "upper": 70}
    macd_defaults = {"fast": 12, "slow": 26, "signal": 9}
    boll_defaults = {"window": 20, "std": 2}

    def _runner(df: pd.DataFrame, initial_run: bool = False, **kwargs) -> Dict[str, Any]:
        close = df["close"]

        sma_p = _best_or_default(best_map, symbol, timeframe, "sma", sma_defaults)
        rsi_p = _best_or_default(best_map, symbol, timeframe, "rsi", rsi_defaults)
        macd_p = _best_or_default(best_map, symbol, timeframe, "macd", macd_defaults)
        boll_p = _best_or_default(best_map, symbol, timeframe, "bollinger", boll_defaults)

        sma_out = run_sma_strategy(close, SMAParams(**sma_p))
        rsi_out = run_rsi_strategy(close, RSIParams(**rsi_p))
        macd_out = run_macd_strategy(close, MACDParams(**macd_p))
        boll_out = run_bollinger_strategy(close, BollingerParams(**boll_p))

        other_results = {
            "sma": sma_out,
            "rsi": rsi_out,
            "macd": macd_out,
            "bollinger": boll_out,
        }

        # Build GPT params from config
        params = GPTCfg(
            provider=str(gpt_cfg.get("provider", "openai")),
            mode="live",
            context_hours=int(gpt_cfg.get("context_hours", 4)),
            weight_sma=float(gpt_cfg.get("weight_sma", 1.0)),
            weight_rsi=float(gpt_cfg.get("weight_rsi", 1.0)),
            weight_macd=float(gpt_cfg.get("weight_macd", 1.0)),
            weight_bollinger=float(gpt_cfg.get("weight_bollinger", 1.0)),
            show_prompt=bool(gpt_cfg.get("show_prompt", False)),
        )

        return run_gpt_strategy(
            symbol=symbol,
            ohlcv=df,
            other_results=other_results,
            timeframe=timeframe,
            params=params,
            initial_run=initial_run,
        )

    return _runner


def load_strategy(config_path: str = "config.yaml") -> Callable:
    """
    Read config.yaml and return a callable: (df, **kwargs) -> Dict[str, Any]
    Supported names: gpt | sma | rsi | macd | bollinger
    """
    cfg = _load_yaml(Path(config_path))
    best_map = _load_best_params()

    # Resolve symbol / timeframe (be flexible with legacy structures)
    trading = cfg.get("trading", {}) or {}
    data = cfg.get("data", {}) or {}

    symbol = trading.get("symbol") or data.get("symbol") or "BTCUSDT"
    timeframe = trading.get("timeframe") or data.get("timeframe") or "15m"

    # Resolve strategy name
    strat_section = cfg.get("strategy")
    if isinstance(strat_section, dict):
        strat_name = strat_section.get("name", "gpt")
    elif isinstance(strat_section, str):
        strat_name = strat_section
    else:
        strat_name = trading.get("strategy", "gpt")

    strat_name = (strat_name or "gpt").lower()

    # GPT-specific params (optional)
    params_root = cfg.get("params", {}) or {}
    gpt_cfg = params_root.get("gpt", {}) or {}
    # also allow putting gpt params directly under strategy section
    if isinstance(strat_section, dict):
        gpt_cfg = {**gpt_cfg, **(strat_section.get("params", {}) or {})}

    if strat_name == "gpt":
        return _build_gpt_runner(symbol, timeframe, best_map, gpt_cfg)
    else:
        return _build_basic_runner(strat_name, symbol, timeframe, best_map)
