"""50-day SMA buffer, shared by runtime and research. Closed 4h bars only."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BandParams:
    days: int = 50
    band: float = .01

    def __post_init__(self):
        if isinstance(self.days,bool) or not isinstance(self.days,int) or not 2<=self.days<=150:
            raise ValueError('days must be an integer in [2,150]')
        if not np.isfinite(self.band) or not 0<self.band<.2:
            raise ValueError('Invalid band')


def band_features(df, params=BandParams()):
    if not isinstance(df.index,pd.DatetimeIndex) or df.index.tz is None or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('Ordered timezone-aware bars required')
    if not np.isfinite(df.close).all() or (df.close<=0).any():
        raise ValueError('Invalid close')
    if any(t.hour%4 or t.minute or t.second for t in df.index.tz_convert('UTC')):
        raise ValueError('4h UTC opening timestamps required')
    window=params.days*6
    sma=df.close.rolling(window).mean()
    ready=sma.notna()
    return pd.DataFrame({'close':df.close,'sma':sma,'upper':sma*(1+params.band),'lower':sma*(1-params.band),
                         'ready':ready,'entries':ready&(df.close>sma*(1+params.band)),
                         'exits':ready&(df.close<sma*(1-params.band))},index=df.index)


def band_decision(row,in_position):
    action='SELL' if in_position and row['exits'] else 'BUY' if not in_position and row['entries'] else 'HOLD'
    score=80. if action!='HOLD' else 0.
    return {'action':action,'confidence':int(score),'reason':f"Closed-bar technical signal: {action}",
            'scores':{'BUY':score if action=='BUY' else 0.,'SELL':score if action=='SELL' else 0.,'HOLD':100-score},
            'allowed_actions':['HOLD','SELL'] if in_position else ['HOLD','BUY'] if row['entries'] else ['HOLD'],
            'should_call_gpt':action!='HOLD','entry_threshold':60.,'exit_threshold':55.,
            'coverage':1.,'regime_label':'above_band' if row['entries'] else 'below_band' if row['exits'] else 'buffer'}
