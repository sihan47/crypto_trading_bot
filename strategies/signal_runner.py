import json
from pathlib import Path
import pandas as pd

from data_manager.data_manager import get_ohlcv
from backtesting.quick_backtester import quick_backtest

# strategies
from strategies.sma_strategy import SMAParams, run_sma_strategy
from strategies.rsi_strategy import RSIParams, run_rsi_strategy

REPORT_DIR = Path(__file__).resolve().parent / "signals"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_strategies(symbol: str, timeframe: str, start: str = None, end: str = None):
    # 1) 讀取資料
    ohlcv = get_ohlcv(symbol, start=start, end=end, timeframe=timeframe)
    close = ohlcv["close"]

    results = {}

    # 2) 跑 SMA 策略
    sma_params = SMAParams(fast=10, slow=50)
    sma_out = run_sma_strategy(close, sma_params)
    sma_perf = quick_backtest(close, sma_out["entries"], sma_out["exits"])
    results["sma"] = {**sma_out, "stats": sma_perf}

    # 3) 跑 RSI 策略
    rsi_params = RSIParams(window=14, lower=30, upper=70)
    rsi_out = run_rsi_strategy(close, rsi_params)
    rsi_perf = quick_backtest(close, rsi_out["entries"], rsi_out["exits"])
    results["rsi"] = {**rsi_out, "stats": rsi_perf}

    # 4) 存 JSON
    out_path = REPORT_DIR / f"{symbol}_{timeframe}_signals.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✅ Signals saved: {out_path}")
    return results


if __name__ == "__main__":
    run_strategies("BTCUSDT", "5m", start="2024-01-01", end="2024-06-01")
