import pandas as pd
import numpy as np


class Backtester:
    def __init__(self, close: pd.Series, entries: pd.Series, exits: pd.Series,
                 fee: float = 0.001, slippage: float = 0.0, init_cash: float = 10000):
        self.close = close
        self.entries = entries.fillna(False)
        self.exits = exits.fillna(False)
        self.fee = fee
        self.slippage = slippage
        self.init_cash = init_cash

        self.equity = None
        self.trades = []

    def run(self):
        cash = self.init_cash
        position = 0
        entry_price = None
        entry_date = None
        equity_curve = []

        for i in range(len(self.close)):
            price = self.close.iloc[i]
            date = self.close.index[i]

            # Exit
            if position > 0 and self.exits.iloc[i]:
                pnl = (price - entry_price) / entry_price
                fee_cost = self.fee + self.slippage
                pnl_after_fee = (1 + pnl) * (1 - fee_cost) - 1
                cash *= (1 + pnl_after_fee)

                self.trades.append({
                    "entry_date": str(entry_date),
                    "exit_date": str(date),
                    "entry_price": float(entry_price),
                    "exit_price": float(price),
                    "return": float(pnl_after_fee)
                })

                position = 0
                entry_price = None
                entry_date = None

            # Entry
            elif position == 0 and self.entries.iloc[i]:
                position = 1
                entry_price = price
                entry_date = date

            equity_curve.append(cash if position == 0 else cash * (price / entry_price))

        self.equity = pd.Series(equity_curve, index=self.close.index, name="equity")
        return self

    def stats(self):
        if self.equity is None:
            raise ValueError("Run the backtest first with .run()")

        total_return = self.equity.iloc[-1] / self.init_cash - 1
        returns = self.equity.pct_change().fillna(0)
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 24 * 12)
        max_dd = (self.equity / self.equity.cummax() - 1).min()
        win_rate = np.mean([1 if t["return"] > 0 else 0 for t in self.trades]) if self.trades else 0
        profit_factor = (
            sum([t["return"] for t in self.trades if t["return"] > 0]) /
            abs(sum([t["return"] for t in self.trades if t["return"] < 0]) or 1)
        ) if self.trades else 0

        return {
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "max_dd": float(max_dd),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "trades": len(self.trades),
            "trade_log": self.trades  # ✅ now includes detailed trades
        }
