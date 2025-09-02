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


def save_params(params_dict):
    # always overwrite with fresh dict (avoid old keys lingering)
    with open(PARAMS_FILE, "w") as f:
        json.dump(params_dict, f, indent=2)


def tune_sma(close, fast_range=(5, 30, 5), slow_range=(20, 100, 20)):
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


def tune_macd(close, fast_range=(8, 10), slow_range=(20, 40), signal_range=(5, 6)):
    best_ret = -np.inf
    best_params = None

    with tqdm(total=(fast_range[1]-fast_range[0])*(slow_range[1]-slow_range[0])*(signal_range[1]-signal_range[0]), desc="Tuning MACD") as pbar:
        for fast in range(*fast_range):
            for slow in range(*slow_range):
                for sig in range(*signal_range):
                    if fast >= slow:
                        pbar.update(1)
                        continue
                    macd_out = run_macd_strategy(close, MACDParams(fast=fast, slow=slow, signal=sig))
                    perf = quick_backtest(close, macd_out["entries"], macd_out["exits"])
                    if perf["total_return"] > best_ret:
                        best_ret = perf["total_return"]
                        best_params = {"fast": fast, "slow": slow, "signal": sig}
                    pbar.update(1)
    return best_params


def tune_bollinger(close, window_range=(10, 30), std_range=(1, 3)):
    best_ret = -np.inf
    best_params = None

    with tqdm(total=(window_range[1]-window_range[0])*(std_range[1]-std_range[0]), desc="Tuning Bollinger") as pbar:
        for w in range(*window_range):
            for s in range(*std_range):
                boll_out = run_bollinger_strategy(close, BollingerParams(window=w, std=s))
                perf = quick_backtest(close, boll_out["entries"], boll_out["exits"])
                if perf["total_return"] > best_ret:
                    best_ret = perf["total_return"]
                    best_params = {"window": w, "std": s}
                pbar.update(1)
    return best_params


def run_tuning(symbol="BTCUSDT", timeframes=["1m", "5m", "15m"], start=None, end=None):
    bests = {}

    for timeframe in timeframes:
        print(f"\n⏳ Tuning {symbol} {timeframe}...")
        ohlcv = get_ohlcv(symbol, start=start, end=end, timeframe=timeframe)
        close = ohlcv["close"]

        # SMA
        sma_params = tune_sma(close)
        bests[f"{symbol}_{timeframe}_sma"] = sma_params

        # RSI
        rsi_params = tune_rsi(close)
        bests[f"{symbol}_{timeframe}_rsi"] = rsi_params

        # MACD
        macd_params = tune_macd(close)
        bests[f"{symbol}_{timeframe}_macd"] = macd_params

        # Bollinger
        boll_params = tune_bollinger(close)
        bests[f"{symbol}_{timeframe}_bollinger"] = boll_params

        save_params(bests)
        print(f"✅ Best params for {symbol} {timeframe} saved to {PARAMS_FILE}")
        for k, v in bests.items():
            if k.startswith(f"{symbol}_{timeframe}"):
                print(k, v)


if __name__ == "__main__":
    run_tuning(symbol="BTCUSDT", timeframes=["15m", "30m"], start="2022-09-01", end="2025-09-01")
