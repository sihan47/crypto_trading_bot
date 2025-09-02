import numpy as np
import pandas as pd


def quick_backtest(close: pd.Series, entries: pd.Series, exits: pd.Series, fee: float = 0.001):
    """
    Simplified backtest with fee: full allocation, entry/exit signals.
    Assumes all-in, all-out trades with given entry/exit signals.
    """
    cash = 1.0
    position = 0
    entry_price = None
    equity_curve = []

    for i in range(len(close)):
        price = close.iloc[i]

        # Exit
        if position > 0 and exits.iloc[i]:
            pnl = (price - entry_price) / entry_price
            pnl_after_fee = (1 + pnl) * (1 - fee) - 1
            cash *= (1 + pnl_after_fee)
            position = 0
            entry_price = None

        # Entry
        elif position == 0 and entries.iloc[i]:
            position = 1
            entry_price = price
            cash *= (1 - fee)  # apply entry fee

        equity_curve.append(cash if position == 0 else cash * (price / entry_price))

    equity = pd.Series(equity_curve, index=close.index, name="equity")

    total_return = equity.iloc[-1] - 1
    returns = equity.pct_change().fillna(0)
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 24 * 12)
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "equity_curve": equity.tolist(),
    }
