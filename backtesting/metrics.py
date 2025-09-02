import numpy as np
import pandas as pd

def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate backtest performance metrics from DataFrame."""

    start_equity = df["equity"].iloc[0]
    end_equity = df["equity"].iloc[-1]

    # Total Return %
    if pd.isna(start_equity) or start_equity == 0:
        total_return = 0
    else:
        total_return = (end_equity / start_equity - 1) * 100

    # Strategy returns
    returns = df["strategy_returns"].dropna()
    mean_return = returns.mean()
    std_return = returns.std()

    # Annualization factor (for 1m data: 365*24*60 minutes per year)
    periods_per_year = 365 * 24 * 60
    annualized_return = (1 + mean_return) ** periods_per_year - 1 if mean_return != 0 else 0

    # Sharpe Ratio
    sharpe = (mean_return / std_return * np.sqrt(periods_per_year)) if std_return > 0 else 0

    # Max Drawdown %
    running_max = df["equity"].cummax()
    drawdown = (df["equity"] - running_max) / running_max
    max_dd = drawdown.min() * 100

    # Win Rate %
    trades = returns[returns != 0]
    win_trades = (trades > 0).sum()
    total_trades = len(trades)
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        "Final Balance": round(float(end_equity), 2),
        "Total Return %": round(float(total_return), 2),
        "Annualized Return %": round(float(annualized_return * 100), 2),
        "Sharpe Ratio": round(float(sharpe), 2),
        "Max Drawdown %": round(float(max_dd), 2),
        "Win Rate %": round(float(win_rate), 2),
        "Total Trades": int(total_trades),
    }
