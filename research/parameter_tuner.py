import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from data_manager.data_manager import get_ohlcv
from backtesting.quick_backtester import quick_backtest

from strategies.sma_strategy import SMAParams, run_sma_strategy
from strategies.rsi_strategy import RSIParams, run_rsi_strategy
from strategies.macd_strategy import MACDParams, run_macd_strategy
from strategies.bollinger_strategy import BollingerParams, run_bollinger_strategy

PARAMS_FILE = Path(__file__).resolve().parent / "best_params.json"


def tune_sma(close, fast_range=(5, 30, 5), slow_range=(20, 100, 10)):
    best_ret = -np.inf
    best_params = None

    fast_values = list(range(*fast_range))
    slow_values = list(range(*slow_range))
    total = len(fast_values) * len(slow_values)

    with tqdm(total=total, desc="Tuning SMA") as pbar:
        for fast in fast_values:
            for slow in slow_values:
                if fast >= slow:
                    pbar.update(1)
                    continue
                sma_out = run_sma_strategy(close, SMAParams(fast=fast, slow=slow))
                perf = quick_backtest(close, sma_out["entries"], sma_out["exits"])
                if perf["total_return"] > best_ret:
                    best_ret = perf["total_return"]
                    best_params = {"fast": fast, "slow": slow}
                pbar.update(1)
    return best_params


def tune_rsi(close, window_range=(10, 30, 2), lower=30, upper=70):
    best_ret = -np.inf
    best_params = None

    win_values = list(range(*window_range))

    with tqdm(total=len(win_values), desc="Tuning RSI") as pbar:
        for w in win_values:
            rsi_out = run_rsi_strategy(close, RSIParams(window=w, lower=lower, upper=upper))
            perf = quick_backtest(close, rsi_out["entries"], rsi_out["exits"])
            if perf["total_return"] > best_ret:
                best_ret = perf["total_return"]
                best_params = {"window": w, "lower": lower, "upper": upper}
            pbar.update(1)
    return best_params


def tune_macd(close, fast_range=(8, 16, 2), slow_range=(20, 30, 2), signal_range=(5, 15, 2)):
    best_ret = -np.inf
    best_params = None

    fast_values = list(range(*fast_range))
    slow_values = list(range(*slow_range))
    signal_values = list(range(*signal_range))
    total = len(fast_values) * len(slow_values) * len(signal_values)

    with tqdm(total=total, desc="Tuning MACD") as pbar:
        for fast in fast_values:
            for slow in slow_values:
                if fast >= slow:
                    pbar.update(len(signal_values))
                    continue
                for sig in signal_values:
                    macd_out = run_macd_strategy(close, MACDParams(fast=fast, slow=slow, signal=sig))
                    perf = quick_backtest(close, macd_out["entries"], macd_out["exits"])
                    if perf["total_return"] > best_ret:
                        best_ret = perf["total_return"]
                        best_params = {"fast": fast, "slow": slow, "signal": sig}
                    pbar.update(1)
    return best_params


def tune_bollinger(close, window_range=(10, 30, 2), std_range=(1, 4, 1)):
    best_ret = -np.inf
    best_params = None

    win_values = list(range(*window_range))
    std_values = list(range(*std_range))
    total = len(win_values) * len(std_values)

    with tqdm(total=total, desc="Tuning Bollinger") as pbar:
        for w in win_values:
            for s in std_values:
                boll_out = run_bollinger_strategy(close, BollingerParams(window=w, std=s))
                perf = quick_backtest(close, boll_out["entries"], boll_out["exits"])
                if perf["total_return"] > best_ret:
                    best_ret = perf["total_return"]
                    best_params = {"window": w, "std": s}
                pbar.update(1)
    return best_params


def save_params(strategy: str, params: dict):
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE, "r") as f:
            all_params = json.load(f)
    else:
        all_params = {}

    all_params[strategy] = params

    with open(PARAMS_FILE, "w") as f:
        json.dump(all_params, f, indent=2)

    print(f"✅ Best params for {strategy} saved: {params}")


def tune_all(symbol: str, timeframe: str, start: str, end: str):
    print(f"🚀 Starting parameter tuning for {symbol} {timeframe} ({start} → {end})")

    ohlcv = get_ohlcv(symbol, start=start, end=end, timeframe=timeframe)
    close = ohlcv["close"]

    # SMA
    best_sma = tune_sma(close)
    save_params("sma", best_sma)

    # RSI
    best_rsi = tune_rsi(close)
    save_params("rsi", best_rsi)

    # MACD
    best_macd = tune_macd(close)
    save_params("macd", best_macd)

    # Bollinger
    best_boll = tune_bollinger(close)
    save_params("bollinger", best_boll)

    print("🎯 Parameter tuning completed!")


if __name__ == "__main__":
    # Default example run
    tune_all("BTCUSDT", "5m", start="2024-01-01", end="2024-06-01")
