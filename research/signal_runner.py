# research/signal_runner.py

import json
from pathlib import Path
import matplotlib.pyplot as plt

from data_manager.data_manager import get_ohlcv
from backtesting.quick_backtester import quick_backtest
from backtesting.backtester import Backtester
from backtesting.plotter import plot_equity_and_drawdown
import yaml

# strategies
from strategies.sma_strategy import SMAParams, run_sma_strategy
from strategies.rsi_strategy import RSIParams, run_rsi_strategy
from strategies.macd_strategy import MACDParams, run_macd_strategy
from strategies.bollinger_strategy import BollingerParams, run_bollinger_strategy

REPORT_DIR = Path(__file__).resolve().parent / "signals"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_params.json"


def load_best_params():
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH, "r" , encoding="utf-8") as f:
            return json.load(f)
    return {}


best_params = load_best_params()


def get_params(strategy_name, symbol, timeframe, default_params):
    key = f"{symbol}_{timeframe}_{strategy_name}"
    if key in best_params:
        print(f"⚡ Using best params for {key}: {best_params[key]}")
        return best_params[key]
    return default_params


def run_strategies(symbol: str, timeframe: str, start: str = None, end: str = None, mode: str = "quick"):
    # Load OHLCV data
    ohlcv = get_ohlcv(symbol, start=start, end=end, timeframe=timeframe)
    close = ohlcv["close"]

    results = {}
    equity_curves = {}  # collect for combined plot (full mode)

    # Helper to choose quick vs full backtest
    def run_backtest(entries, exits, strat_name: str):
        if mode == "quick":
            return quick_backtest(close, entries, exits)
        elif mode == "full":
            bt = Backtester(close, entries, exits, fee=0.001, init_cash=10000)
            bt.run()
            stats = bt.stats()

            # save individual plot
            plots_dir = REPORT_DIR / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            out_path = plots_dir / f"{symbol}_{timeframe}_{strat_name}_equity.png"
            plot_equity_and_drawdown(bt.equity, out_path, title=f"{symbol} {timeframe} {strat_name.upper()}")

            # save equity curve for combined plot
            equity_curves[strat_name] = bt.equity

            return stats
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # === SMA ===
    sma_defaults = {"fast": 10, "slow": 50}
    sma_params = get_params("sma", symbol, timeframe, sma_defaults)
    sma_out = run_sma_strategy(close, SMAParams(**sma_params))
    sma_perf = run_backtest(sma_out["entries"], sma_out["exits"], "sma")
    results["sma"] = {**sma_out, "stats": sma_perf}

    # === RSI ===
    rsi_defaults = {"window": 14, "lower": 30, "upper": 70}
    rsi_params = get_params("rsi", symbol, timeframe, rsi_defaults)
    rsi_out = run_rsi_strategy(close, RSIParams(**rsi_params))
    rsi_perf = run_backtest(rsi_out["entries"], rsi_out["exits"], "rsi")
    results["rsi"] = {**rsi_out, "stats": rsi_perf}

    # === MACD ===
    macd_defaults = {"fast": 12, "slow": 26, "signal": 9}
    macd_params = get_params("macd", symbol, timeframe, macd_defaults)
    macd_out = run_macd_strategy(close, MACDParams(**macd_params))
    macd_perf = run_backtest(macd_out["entries"], macd_out["exits"], "macd")
    results["macd"] = {**macd_out, "stats": macd_perf}

    # === Bollinger Bands ===
    boll_defaults = {"window": 20, "std": 2}
    boll_params = get_params("bollinger", symbol, timeframe, boll_defaults)
    boll_out = run_bollinger_strategy(close, BollingerParams(**boll_params))
    boll_perf = run_backtest(boll_out["entries"], boll_out["exits"], "bollinger")
    results["bollinger"] = {**boll_out, "stats": boll_perf}

    # === GPT Meta-Strategy (consumes others' outputs + last N hours K-line) ===

    # === Combined equity plot ===
    if mode == "full" and equity_curves:
        plots_dir = REPORT_DIR / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        combined_path = plots_dir / f"{symbol}_{timeframe}_combined_equity.png"

        plt.figure(figsize=(12, 6))
        for strat_name, eq in equity_curves.items():
            plt.plot(eq.index, eq.values, label=strat_name.upper())
        plt.title(f"Combined Equity Curves ({symbol} {timeframe})")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(combined_path)
        plt.close()
        print(f"📊 Combined equity plot saved: {combined_path}")

    # Save JSON
    out_path = REPORT_DIR / f"{symbol}_{timeframe}_{mode}_signals.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✅ Signals saved: {out_path}")
    return results


def run_multi_timeframes(symbol="BTCUSDT", timeframes=["1m", "5m", "15m"], start=None, end=None, mode="full"):
    for tf in timeframes:
        print(f"\n🚀 Running strategies for {symbol} {tf}...")
        run_strategies(symbol, tf, start=start, end=end, mode=mode)


def run_from_config(config_file: str = "config.yaml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    timeframe = data_cfg.get("timeframe", "15m")
    start = data_cfg.get("start", None)
    end = data_cfg.get("end", None)
    # delegate to existing runner (no accidental call to run_tuning)
    return run_multi_timeframes(symbol=symbol, timeframes=[timeframe], start=start, end=end, mode="full")


if __name__ == "__main__":
    try:
        run_from_config()
    except Exception:
        run_multi_timeframes(symbol="BTCUSDT", timeframes=["15m"], start="2022-08-03", end="2025-09-01", mode="full")
