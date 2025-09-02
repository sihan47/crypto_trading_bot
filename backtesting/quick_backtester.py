import numpy as np
import pandas as pd


def quick_backtest(close: pd.Series, entries: pd.Series, exits: pd.Series, fee: float = 0.001):
    """
    Simplified backtest: assume full allocation, only entry/exit signals.
    Calculates basic performance metrics.
    """
    # Daily returns
    ret = close.pct_change().fillna(0)

    # Position: cumulative entries minus exits
    pos = entries.cumsum() - exits.cumsum()

    # Strategy returns
    strat_ret = pos.shift(1).fillna(0) * ret - fee * (entries.astype(int) + exits.astype(int))

    # Equity curve
    equity = (1 + strat_ret).cumprod()

    total_return = equity.iloc[-1] - 1
    sharpe = strat_ret.mean() / (strat_ret.std() + 1e-9) * np.sqrt(252 * 24 * 12)  # minute freq
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "equity_curve": equity.tolist(),
    }
