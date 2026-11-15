"""Research-only spot signals. No account access or order submission.

Day-based lookbacks remain comparable across execution timeframes. The band
thresholds are research choices, not a replication of a published optimum.
"""
import numpy as np
import pandas as pd


RULES = {
    'sma20_band': ('sma', 20, 0, .01),
    'sma50_band': ('sma', 50, 0, .01),
    'ema8_32_band': ('ema', 8, 32, .01),
    'momentum28': ('momentum', 28, 0, .01),
    'momentum84': ('momentum', 84, 0, .01),
    'channel20_10': ('channel', 20, 10, 0),
    'channel55_20': ('channel', 55, 20, 0),
    'macd_standard': ('macd', 12, 26, 0),
    'macd_trend_filter': ('macd_filter', 12, 26, 0),
}


def candidate_signals(df, bars_per_day):
    c = df['close']
    output = {}
    for name,(kind,a,b,band) in RULES.items():
        if kind == 'sma':
            avg = c.rolling(a*bars_per_day).mean()
            buy, sell = c > avg*(1+band), c < avg*(1-band)
        elif kind == 'ema':
            fast = c.ewm(span=a*bars_per_day,adjust=False,min_periods=a*bars_per_day).mean()
            slow = c.ewm(span=b*bars_per_day,adjust=False,min_periods=b*bars_per_day).mean()
            buy, sell = fast > slow*(1+band), fast < slow*(1-band)
        elif kind == 'momentum':
            change = c/c.shift(a*bars_per_day)-1
            buy, sell = change > band, change < -band
        elif kind == 'channel':
            upper = df['high'].rolling(a*bars_per_day).max().shift(1)
            lower = df['low'].rolling(b*bars_per_day).min().shift(1)
            buy, sell = c > upper, c < lower
        else:
            # Conventional bullish crossover; distinct from repo's inverse MACD.
            fast = c.ewm(span=a,adjust=False,min_periods=a).mean()
            slow = c.ewm(span=b,adjust=False,min_periods=b).mean()
            line = fast-slow
            signal = line.ewm(span=9,adjust=False,min_periods=9).mean()
            buy = (line > signal) & (line.shift() <= signal.shift())
            sell = (line < signal) & (line.shift() >= signal.shift())
            if kind == 'macd_filter':
                trend = c.rolling(20*bars_per_day).mean()
                buy &= c > trend
                sell |= c < trend
        output[name] = (buy.fillna(False),sell.fillna(False))
    return output
