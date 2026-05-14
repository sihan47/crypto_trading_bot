# research/parameter_tuner.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Any

import pandas as pd

# Fallback-safe tqdm import
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    def _tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable

from data_manager.data_manager import get_ohlcv
from backtesting.backtester import Backtester

from strategies.sma_strategy import SMAParams, run_sma_strategy
from strategies.rsi_strategy import RSIParams, run_rsi_strategy
from strategies.macd_strategy import MACDParams, run_macd_strategy
from strategies.bollinger_strategy import BollingerParams, run_bollinger_strategy
from strategies.ema_adx_strategy import EMAADXParams, run_ema_adx_strategy
from strategies.mean_reversion_strategy import MeanReversionParams, run_mean_reversion_strategy
from strategies.zscore_strategy import ZScoreParams, run_zscore_strategy


# ---------- Paths ----------
THIS_DIR = Path(__file__).resolve().parent
BEST_PATH = THIS_DIR / "best_params.json"


# ---------- Best file IO ----------
def _load_best() -> dict:
    if BEST_PATH.exists():
        try:
            with open(BEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_best(d: dict) -> None:
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def record_best(symbol: str, timeframe: str, strat_name: str, params: dict,
                perf_pct: float, start: str, end: str) -> None:
    """
    Persist best params and backtest metadata for GPT prompt decoration.
    """
    key = f"{symbol}_{timeframe}_{strat_name}"
    data = _load_best()
    data[key] = params
    bt_meta = data.get("__backtest", {})
    bt_meta[key] = {
        "period": f"{start}→{end}",
        "performance": f"{perf_pct:.2f}%"
    }
    data["__backtest"] = bt_meta
    _save_best(data)


# ---------- Utilities ----------
def _max_window_in_params(strat: str, params: dict) -> int:
    if strat == "sma":
        return max(params["fast"], params["slow"])
    if strat == "rsi":
        return params["window"]
    if strat == "macd":
        return params["slow"] + params["signal"]
    if strat == "bollinger":
        return params["window"]
    if strat == "ema_adx":
        return max(params.get("slow", 55), params.get("adx_window", 14))
    if strat == "mean_reversion":
        return max(params.get("rsi_window", 14), params.get("bb_window", 20))
    if strat == "zscore":
        return params.get("window", 100)
    return 0


def _evaluate(close: pd.Series, entries: pd.Series, exits: pd.Series,
              fee: float = 0.001, init_cash: float = 10_000.0) -> float:
    """
    Run full backtest to compute performance % robustly.
    """
    bt = Backtester(close, entries, exits, fee=fee, init_cash=init_cash)
    bt.run()
    eq = bt.equity
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return 0.0
    return (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0


def _param_grid_list(grid_dict: Dict[str, Iterable]) -> List[Dict[str, Any]]:
    keys = list(grid_dict.keys())
    values = [list(grid_dict[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ---------- Strategy tuners (with tqdm) ----------
def tune_sma(close: pd.Series) -> Tuple[Dict[str, Any], float]:
    grid = {
        "fast": [5, 10, 20, 25],
        "slow": [50, 80, 100, 200],
    }
    combos = _param_grid_list(grid)
    best_params, best_perf = None, -1e9
    for p in _tqdm(combos, desc="Tuning SMA", total=len(combos)):
        if p["fast"] >= p["slow"]:
            continue
        if len(close) < _max_window_in_params("sma", p):
            continue
        out = run_sma_strategy(close, SMAParams(**p))
        perf = _evaluate(close, out["entries"], out["exits"])
        if perf > best_perf:
            best_perf, best_params = perf, p
    return best_params or {"fast": 10, "slow": 50}, float(best_perf)


def tune_rsi(close: pd.Series) -> Tuple[Dict[str, Any], float]:
    grid = {
        "window": [14, 21, 26],
        "lower": [20, 30],
        "upper": [70, 80],
    }
    combos = _param_grid_list(grid)
    best_params, best_perf = None, -1e9
    for p in _tqdm(combos, desc="Tuning RSI", total=len(combos)):
        if p["lower"] >= p["upper"]:
            continue
        if len(close) < _max_window_in_params("rsi", p):
            continue
        out = run_rsi_strategy(close, RSIParams(**p))
        perf = _evaluate(close, out["entries"], out["exits"])
        if perf > best_perf:
            best_perf, best_params = perf, p
    return best_params or {"window": 14, "lower": 30, "upper": 70}, float(best_perf)


def tune_macd(close: pd.Series) -> Tuple[Dict[str, Any], float]:
    grid = {
        "fast": [9, 12],
        "slow": [26, 39],
        "signal": [5, 9],
    }
    combos = _param_grid_list(grid)
    best_params, best_perf = None, -1e9
    for p in _tqdm(combos, desc="Tuning MACD", total=len(combos)):
        if p["fast"] >= p["slow"]:
            continue
        if len(close) < _max_window_in_params("macd", p):
            continue
        out = run_macd_strategy(close, MACDParams(**p))
        perf = _evaluate(close, out["entries"], out["exits"])
        if perf > best_perf:
            best_perf, best_params = perf, p
    return best_params or {"fast": 12, "slow": 26, "signal": 9}, float(best_perf)


def tune_bollinger(close: pd.Series) -> Tuple[Dict[str, Any], float]:
    grid = {
        "window": [20, 29],
        "std": [2, 3],
    }
    combos = _param_grid_list(grid)
    best_params, best_perf = None, -1e9
    for p in _tqdm(combos, desc="Tuning BOLLINGER", total=len(combos)):
        if len(close) < _max_window_in_params("bollinger", p):
            continue
        out = run_bollinger_strategy(close, BollingerParams(**p))
        perf = _evaluate(close, out["entries"], out["exits"])
        if perf > best_perf:
            best_perf, best_params = perf, p
    return best_params or {"window": 20, "std": 2}, float(best_perf)


def tune_ema_adx(ohlcv: pd.DataFrame) -> Tuple[Dict[str, Any], float]:
    grid = {
        "fast": [13, 21],
        "slow": [34, 55],
        "adx_window": [14],
        "adx_threshold": [15.0, 20.0, 25.0],
    }
    combos = _param_grid_list(grid)
    best_params, best_perf = None, -1e9
    close = ohlcv["close"]
    for p in _tqdm(combos, desc="Tuning EMA_ADX", total=len(combos)):
        if p["fast"] >= p["slow"]:
            continue
        if len(close) < _max_window_in_params("ema_adx", p):
            continue
        out = run_ema_adx_strategy(ohlcv, EMAADXParams(**p))
        perf = _evaluate(close, out["entries"], out["exits"])
        if perf > best_perf:
            best_perf, best_params = perf, p
    return (
        best_params or {"fast": 21, "slow": 55, "adx_window": 14, "adx_threshold": 18.0},
        float(best_perf),
    )


def tune_mean_reversion(close: pd.Series) -> Tuple[Dict[str, Any], float]:
    grid = {
        "rsi_window": [14, 21],
        "rsi_lower": [28.0, 32.0],
        "rsi_upper": [68.0, 72.0],
        "bb_window": [20, 30],
        "bb_std": [2.0, 2.5],
    }
    combos = _param_grid_list(grid)
    best_params, best_perf = None, -1e9
    for p in _tqdm(combos, desc="Tuning MEAN_REVERSION", total=len(combos)):
        if p["rsi_lower"] >= p["rsi_upper"]:
            continue
        if len(close) < _max_window_in_params("mean_reversion", p):
            continue
        out = run_mean_reversion_strategy(close, MeanReversionParams(**p))
        perf = _evaluate(close, out["entries"], out["exits"])
        if perf > best_perf:
            best_perf, best_params = perf, p
    return (
        best_params or {"rsi_window": 14, "rsi_lower": 32.0, "rsi_upper": 68.0, "bb_window": 20, "bb_std": 2.2},
        float(best_perf),
    )


def tune_zscore(close: pd.Series) -> Tuple[Dict[str, Any], float]:
    """Z-score mean reversion — suited for ratio pairs like BNBBTC."""
    grid = {
        "window": [50, 100, 150, 200],
        "entry_z": [1.0, 1.5, 2.0],
        "exit_z": [0.0, 0.5, 1.0],
    }
    combos = _param_grid_list(grid)
    best_params, best_perf = None, -1e9
    for p in _tqdm(combos, desc="Tuning ZSCORE", total=len(combos)):
        if p["exit_z"] >= p["entry_z"]:
            continue
        if len(close) < _max_window_in_params("zscore", p):
            continue
        out = run_zscore_strategy(close, ZScoreParams(**p))
        perf = _evaluate(close, out["entries"], out["exits"])
        if perf > best_perf:
            best_perf, best_params = perf, p
    return best_params or {"window": 100, "entry_z": 1.5, "exit_z": 0.5}, float(best_perf)


# ---------- Orchestrator (with tqdm over timeframes) ----------
def run_tuning(
    symbol: str = "BTCUSDT",
    timeframes: Tuple[str, ...] = ("15m",),
    start: str = "2022-09-01",
    end: str = "2026-05-01",
    fee: float = 0.0004,
    init_cash: float = 10_000.0,
    strategies: List[str] = ("sma", "rsi", "macd", "bollinger", "ema_adx", "mean_reversion", "zscore"),
) -> None:
    """
    Run parameter tuning per strategy and timeframe, persist best params and backtest metadata.
    """
    for timeframe in _tqdm(timeframes, desc=f"{symbol} timeframes", total=len(timeframes)):
        ohlcv = get_ohlcv(symbol, timeframe=timeframe, start=start, end=end)
        if not isinstance(ohlcv, pd.DataFrame) or "close" not in ohlcv.columns:
            raise RuntimeError("get_ohlcv did not return a DataFrame with 'close' column.")
        close = ohlcv["close"].dropna()
        if len(close) < 100:
            raise RuntimeError(f"Not enough data for tuning: len(close)={len(close)}")

        if "sma" in strategies:
            sma_params, sma_perf = tune_sma(close)
            record_best(symbol, timeframe, "sma", sma_params, sma_perf, start, end)

        if "rsi" in strategies:
            rsi_params, rsi_perf = tune_rsi(close)
            record_best(symbol, timeframe, "rsi", rsi_params, rsi_perf, start, end)

        if "macd" in strategies:
            macd_params, macd_perf = tune_macd(close)
            record_best(symbol, timeframe, "macd", macd_params, macd_perf, start, end)

        if "bollinger" in strategies:
            boll_params, boll_perf = tune_bollinger(close)
            record_best(symbol, timeframe, "bollinger", boll_params, boll_perf, start, end)

        if "ema_adx" in strategies:
            ema_adx_params, ema_adx_perf = tune_ema_adx(ohlcv)
            record_best(symbol, timeframe, "ema_adx", ema_adx_params, ema_adx_perf, start, end)

        if "mean_reversion" in strategies:
            mr_params, mr_perf = tune_mean_reversion(close)
            record_best(symbol, timeframe, "mean_reversion", mr_params, mr_perf, start, end)

        if "zscore" in strategies:
            zs_params, zs_perf = tune_zscore(close)
            record_best(symbol, timeframe, "zscore", zs_params, zs_perf, start, end)

        # No GPT tuning; GPT is a meta strategy.


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune strategy parameters for a symbol")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--start", type=str, default="2022-09-01")
    parser.add_argument("--end", type=str, default="2025-09-01")
    parser.add_argument(
        "--strategies", type=str, default="sma,rsi,macd,bollinger",
        help="Comma-separated list of strategies to tune"
    )
    args = parser.parse_args()

    run_tuning(
        symbol=args.symbol,
        timeframes=(args.timeframe,),
        start=args.start,
        end=args.end,
        strategies=args.strategies.split(","),
    )
