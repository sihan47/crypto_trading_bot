"""Long-only, next-open research simulator with costs on both legs."""
import numpy as np
import pandas as pd


def simulate(df, entries, exits, fee=0.001, slippage=0.0005, initial_cash=10000.0):
    if len(df) < 2 or not entries.index.equals(df.index) or not exits.index.equals(df.index):
        raise ValueError('Aligned prices and signals with at least two bars required')
    if fee < 0 or slippage < 0 or fee >= 1 or slippage >= 1:
        raise ValueError('Invalid costs')
    opens, closes = df['open'].to_numpy(), df['close'].to_numpy()
    if not np.isfinite(opens).all() or not np.isfinite(closes).all() or min(opens.min(), closes.min()) <= 0:
        raise ValueError('Invalid prices')
    buys, sells = entries.fillna(False).to_numpy(), exits.fillna(False).to_numpy()
    cash, qty, entry_value = initial_cash, 0.0, 0.0
    equity, trades, fees = [], [], 0.0
    held, orders = 0, 0
    step = df.index.to_series().diff().dropna().mode().iloc[0]
    for i in range(len(df)):
        # A bar's close cannot be used to execute at that same close/open.
        if i and df.index[i] - df.index[i-1] == step:
            if qty > 0 and sells[i-1]:
                proceeds = qty * opens[i] * (1 - slippage)
                fees += proceeds * fee
                cash = proceeds * (1 - fee)
                trades.append(cash / entry_value - 1)
                qty = 0.0
                orders += 1
            elif qty == 0 and buys[i-1]:
                entry_value = cash
                gross = cash / (1 + fee)
                fees += gross * fee
                qty = gross / (opens[i] * (1 + slippage))
                cash = 0.0
                orders += 1
        held += qty > 0
        equity.append(cash + qty * closes[i])
    # Mark open positions at liquidation value for consistent final comparisons.
    if qty > 0:
        proceeds = qty * closes[-1] * (1 - slippage)
        fees += proceeds * fee
        equity[-1] = proceeds * (1 - fee)
        trades.append(equity[-1] / entry_value - 1)
        orders += 1
    curve = pd.Series(equity, index=df.index)
    daily = curve.resample('1D').last().pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(365)) if len(daily) > 1 and daily.std() > 0 else 0.0
    drawdown = curve / curve.cummax().clip(lower=initial_cash) - 1
    return {
        'return_pct': float((curve.iloc[-1] / initial_cash - 1) * 100),
        'max_drawdown_pct': float(drawdown.min() * 100),
        'sharpe_daily': sharpe, 'round_trips': len(trades), 'orders': orders,
        'win_rate_pct': float(np.mean(np.array(trades) > 0) * 100) if trades else 0.0,
        'exposure_pct': float(held / len(df) * 100), 'fees': float(fees),
        'entry_signal_bars': int(buys.sum()), 'exit_signal_bars': int(sells.sum()),
    }
