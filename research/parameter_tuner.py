# research/parameter_tuner.py

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


# ---------- Orchestrator (with tqdm over timeframes) ----------
def run_tuning(
    symbol: str = "BTCUSDT",
    timeframes: Tuple[str, ...] = ("15m",),
    start: str = "2022-09-01",
    end: str = "2025-09-01",
    fee: float = 0.0004,
    init_cash: float = 10_000.0,
    strategies: List[str] = ("sma", "rsi", "macd", "bollinger"),
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

        # No GPT tuning; GPT is a meta strategy.


if __name__ == "__main__":
    run_tuning(
        symbol="BTCUSDT",
        timeframes=("15m",),
        start="2022-09-01",
        end="2025-09-01",
    )
