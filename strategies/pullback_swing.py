"""Research/runtime shared long-only RSI2 trend pullback, market-entry adaptation."""
import numpy as np
import pandas as pd


def pullback_features(df,bars_per_day=6,trend_days=100,entry_rsi=15,exit_rsi=30):
    if bars_per_day not in (1,6):
        raise ValueError('Only daily or 4h pullback candles supported')
    if not isinstance(df.index,pd.DatetimeIndex) or df.index.tz is None or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('Ordered timezone-aware candles required')
    if not np.isfinite(df.close).all() or (df.close<=0).any():
        raise ValueError('Invalid close')
    hours=24//bars_per_day
    if any(t.hour%hours or t.minute or t.second for t in df.index.tz_convert('UTC')):
        raise ValueError('Candle timestamps do not match timeframe')
    c=df.close
    if not 2<=trend_days<=200 or not 0<entry_rsi<exit_rsi<100:
        raise ValueError('Invalid pullback thresholds')
    trend=c.rolling(trend_days*bars_per_day).mean()
    delta=c.diff()
    gain=delta.clip(lower=0).ewm(alpha=.5,adjust=False,min_periods=2).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=.5,adjust=False,min_periods=2).mean()
    rsi=(100*gain/(gain+loss).replace(0,np.nan)).fillna(50.)
    ready=trend.notna()
    entry=ready&(c>trend)&(rsi<entry_rsi)&(rsi.shift()>=entry_rsi)
    exit_=ready&((rsi>exit_rsi)|(c<trend))
    return pd.DataFrame({'close':c,'trend_sma':trend,'rsi2':rsi,'previous_rsi2':rsi.shift(),'ready':ready,
                         'entries':entry,'exits':exit_},index=df.index)
