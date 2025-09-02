import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt

from data_manager.data_manager import get_ohlcv
from backtesting.utils import _to_pandas_freq

# strategies
from strategies.sma_strategy import SMAParams, run_sma_strategy
from strategies.rsi_strategy import RSIParams, run_rsi_strategy

REPORT_DIR = Path(__file__).resolve().parent / "vbt_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def parse_tuple3(text: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Range must be 'start,stop,step'.")
    return parts[0], parts[1], parts[2]


def run_backtest(
    symbol: str,
    timeframe: str,
    strategy: str = "sma",   # "sma", "rsi", or "multi"
    start: str = None,
    end: str = None,
    fee_pct: float = 0.001,
    slippage: float = 0.0,
    init_cash: float = 10_000.0,
    grid_fast: Tuple[int, int, int] = (5, 101, 5),
    grid_slow: Tuple[int, int, int] = (10, 201, 10),
    rsi_window: Tuple[int, int, int] = (14, 31, 2),
    rsi_lower: int = 30,
    rsi_upper: int = 70,
):
    # 1) Load OHLCV
    ohlcv = get_ohlcv(symbol, start=start, end=end, timeframe=timeframe)
    close = ohlcv["close"].copy()
    freq = _to_pandas_freq(timeframe)

    reports = {}
    signals = {}
    portfolios = {}

    # ============================
    # === SMA Strategy ===
    # ============================
    if strategy in ("sma", "multi"):
        fast_range = list(range(*grid_fast))
        slow_range = list(range(*grid_slow))

        fast_ma = vbt.MA.run(close, window=fast_range).ma
        slow_ma = vbt.MA.run(close, window=slow_range).ma

        entries = (fast_ma.values[:, :, None] > slow_ma.values[:, None, :])
        exits = (fast_ma.values[:, :, None] < slow_ma.values[:, None, :])

        cols = pd.MultiIndex.from_product([fast_range, slow_range], names=["fast", "slow"])
        entries_df = pd.DataFrame(entries.reshape(len(close), -1), index=close.index, columns=cols)
        exits_df = pd.DataFrame(exits.reshape(len(close), -1), index=close.index, columns=cols)

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries_df,
            exits=exits_df,
            fees=fee_pct,
            slippage=slippage,
            init_cash=init_cash,
            size=np.inf,
            freq=freq,
        )

        total_returns = pf.total_return()
        best_loc = total_returns.idxmax()
        best_fast, best_slow = best_loc
        best_pf = pf[(best_fast, best_slow)]

        best_params = {"fast": int(best_fast), "slow": int(best_slow)}

        stats = {
            "Total Return": float(best_pf.total_return()),
            "CAGR": float(best_pf.annualized_return()),
            "Sharpe": float(best_pf.sharpe_ratio()),
            "Max Drawdown": float(best_pf.max_drawdown()),
            "Win Rate": float(best_pf.trades.win_rate()),
            "Profit Factor": float(best_pf.trades.profit_factor()),
            "Avg Trade Return": float(best_pf.trades.returns.mean()),
            "Total Trades": int(best_pf.trades.count()),
        }

        reports["sma"] = {"best_params": best_params, "stats": stats}
        signals["sma"] = {
            "entries": entries_df[best_loc].astype(int).tolist(),
            "exits": exits_df[best_loc].astype(int).tolist(),
        }
        portfolios["sma"] = best_pf

        # Save equity curve
        eq_png = REPORT_DIR / f"equity_{symbol}_{timeframe}_sma.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        best_pf.value().plot(ax=ax, title=f"Equity Curve ({symbol} {timeframe}, SMA)")
        ax.set_ylabel("Portfolio Value")
        fig.tight_layout()
        fig.savefig(eq_png)
        plt.close(fig)

    # ============================
    # === RSI Strategy ===
    # ============================
    if strategy in ("rsi", "multi"):
        window_range = list(range(*rsi_window))

        entries = {}
        exits = {}
        for w in window_range:
            out = run_rsi_strategy(close, RSIParams(window=w, lower=rsi_lower, upper=rsi_upper))
            entries[w] = out["signal_long"]
            exits[w] = out["signal_exit"]

        entries_df = pd.DataFrame(entries)
        exits_df = pd.DataFrame(exits)

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries_df,
            exits=exits_df,
            fees=fee_pct,
            slippage=slippage,
            init_cash=init_cash,
            size=np.inf,
            freq=freq,
        )

        total_returns = pf.total_return()
        best_w = total_returns.idxmax()
        best_pf = pf[best_w]

        best_params = {"window": int(best_w), "lower": rsi_lower, "upper": rsi_upper}

        stats = {
            "Total Return": float(best_pf.total_return()),
            "CAGR": float(best_pf.annualized_return()),
            "Sharpe": float(best_pf.sharpe_ratio()),
            "Max Drawdown": float(best_pf.max_drawdown()),
            "Win Rate": float(best_pf.trades.win_rate()),
            "Profit Factor": float(best_pf.trades.profit_factor()),
            "Avg Trade Return": float(best_pf.trades.returns.mean()),
            "Total Trades": int(best_pf.trades.count()),
        }

        reports["rsi"] = {"best_params": best_params, "stats": stats}
        signals["rsi"] = {
            "entries": entries_df[best_w].astype(int).tolist(),
            "exits": exits_df[best_w].astype(int).tolist(),
        }
        portfolios["rsi"] = best_pf

        # Save equity curve
        eq_png = REPORT_DIR / f"equity_{symbol}_{timeframe}_rsi.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        best_pf.value().plot(ax=ax, title=f"Equity Curve ({symbol} {timeframe}, RSI)")
        ax.set_ylabel("Portfolio Value")
        fig.tight_layout()
        fig.savefig(eq_png)
        plt.close(fig)

    # ============================
    # === Combine Portfolios ===
    # ============================
    if strategy == "multi" and len(portfolios) > 1:
        cash_per_strategy = init_cash / len(portfolios)
        combined_pf = vbt.Portfolio.combine(list(portfolios.values()), cash_shares=cash_per_strategy)

        stats = {
            "Total Return": float(combined_pf.total_return()),
            "CAGR": float(combined_pf.annualized_return()),
            "Sharpe": float(combined_pf.sharpe_ratio()),
            "Max Drawdown": float(combined_pf.max_drawdown()),
            "Win Rate": float(combined_pf.trades.win_rate()),
            "Profit Factor": float(combined_pf.trades.profit_factor()),
            "Avg Trade Return": float(combined_pf.trades.returns.mean()),
            "Total Trades": int(combined_pf.trades.count()),
        }

        reports["multi"] = {"best_params": "均分資金", "stats": stats}

        eq_png = REPORT_DIR / f"equity_{symbol}_{timeframe}_multi.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        combined_pf.value().plot(ax=ax, title=f"Equity Curve ({symbol} {timeframe}, Multi)")
        ax.set_ylabel("Portfolio Value")
        fig.tight_layout()
        fig.savefig(eq_png)
        plt.close(fig)

    # ============================
    # === Save Combined Report ===
    # ============================
    report = {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_mode": strategy,
        "start": str(close.index.min()),
        "end": str(close.index.max()),
        "reports": reports,
        "signals": signals,
        "fee_pct": fee_pct,
        "slippage": slippage,
        "init_cash": init_cash,
    }

    out_json = REPORT_DIR / f"{symbol}_{timeframe}_{strategy}_grid.json"
    out_json.write_text(json.dumps(report, indent=2))

    print("=== Backtest Summary ===")
    for strat, res in reports.items():
        print(f"[{strat.upper()}] Params: {res['best_params']} | "
              f"Return: {res['stats']['Total Return']:.2%}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--strategy", default="sma", choices=["sma", "rsi", "multi"])
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--fast", type=str, default="5,101,5")
    p.add_argument("--slow", type=str, default="10,201,10")
    p.add_argument("--rsi_window", type=str, default="14,31,2")
    p.add_argument("--rsi_lower", type=int, default=30)
    p.add_argument("--rsi_upper", type=int, default=70)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_backtest(
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy=args.strategy,
        start=args.start,
        end=args.end,
        fee_pct=args.fee,
        slippage=args.slippage,
        init_cash=args.cash,
        grid_fast=parse_tuple3(args.fast),
        grid_slow=parse_tuple3(args.slow),
        rsi_window=parse_tuple3(args.rsi_window),
        rsi_lower=args.rsi_lower,
        rsi_upper=args.rsi_upper,
    )
