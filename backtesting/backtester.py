import pandas as pd
from backtesting.metrics import calculate_metrics
from backtesting.plotter import plot_backtest
from data_manager.data_manager import load_data

# strategies
from strategies.sma_strategy import SMAStrategy
from strategies.gpt_strategy import GPTStrategy


def run_backtest(symbol, strategy_name, start_date=None, end_date=None, verbose=False, plot=False):
    # 1. Load historical data
    df = load_data(symbol, "1m")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # 2. Trim by date range
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    if verbose:
        print(f"Backtest range: {df.index.min()} → {df.index.max()} (total {len(df)} rows)")

    # 3. Select strategy
    if strategy_name == "sma":
        strategy = SMAStrategy(short_window=10, long_window=50)
    elif strategy_name == "gpt":
        strategy = GPTStrategy(lookback=20, model="gpt-4o-mini")
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # 4. Generate signals
    df["signal"] = strategy.generate_signals(df)

    # 5. Position and returns
    df["position"] = df["signal"].shift().fillna(0)
    df["returns"] = df["close"].pct_change().fillna(0)
    df["strategy_returns"] = (df["position"] * df["returns"]).fillna(0)
    df["equity"] = (1 + df["strategy_returns"]).cumprod() * 10000  # initial capital 10000



    # 6. Calculate metrics
    stats = calculate_metrics(df)

    # 7. Plot result
    plot_backtest(df, symbol, save=True, show=plot)

    return stats
