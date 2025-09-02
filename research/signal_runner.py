import json
from pathlib import Path
import matplotlib.pyplot as plt

from data_manager.data_manager import get_ohlcv
from backtesting.quick_backtester import quick_backtest
from backtesting.backtester import Backtester
from backtesting.plotter import plot_equity_and_drawdown

# strategies
from strategies.sma_strategy import SMAParams, run_sma_strategy
from strategies.rsi_strategy import RSIParams, run_rsi_strategy
from strategies.macd_strategy import MACDParams, run_macd_strategy
from strategies.bollinger_strategy import BollingerParams, run_bollinger_strategy

REPORT_DIR = Path(__file__).resolve().parent / "signals"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_strategies(symbol: str, timeframe: str, start: str = None, end: str = None, mode: str = "quick"):
    # Load OHLCV data
    ohlcv = get_ohlcv(symbol, start=start, end=end, timeframe=timeframe)
    close = ohlcv["close"]

    results = {}
    equity_curves = {}  # collect for combined plot

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
    sma_out = run_sma_strategy(close, SMAParams(fast=10, slow=50))
    sma_perf = run_backtest(sma_out["entries"], sma_out["exits"], "sma")
    results["sma"] = {**sma_out, "stats": sma_perf}

    # === RSI ===
    rsi_out = run_rsi_strategy(close, RSIParams(window=14, lower=30, upper=70))
    rsi_perf = run_backtest(rsi_out["entries"], rsi_out["exits"], "rsi")
    results["rsi"] = {**rsi_out, "stats": rsi_perf}

    # === MACD ===
    macd_out = run_macd_strategy(close, MACDParams(fast=12, slow=26, signal=9))
    macd_perf = run_backtest(macd_out["entries"], macd_out["exits"], "macd")
    results["macd"] = {**macd_out, "stats": macd_perf}

    # === Bollinger Bands ===
    boll_out = run_bollinger_strategy(close, BollingerParams(window=20, std=2))
    boll_perf = run_backtest(boll_out["entries"], boll_out["exits"], "bollinger")
    results["bollinger"] = {**boll_out, "stats": boll_perf}

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


if __name__ == "__main__":
    run_strategies("BTCUSDT", "1m", start="2022-09-01", end="2025-09-01", mode="full")
