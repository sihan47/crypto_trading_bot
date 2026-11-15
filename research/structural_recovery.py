"""Iteration B: sell-pressure shock/recovery proxy, NOT observed liquidations."""
from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from research.structural_btc import Parameters,features,run_research


@dataclass(frozen=True)
class RecoveryParameters:
    day: int=24
    week: int=168
    shock: float=2.
    participation: float=2.
    recovery: float=.5
    trail: float=2.
    max_hours: int=24
    risk: float=.005
    liquidity_cap: float=.001


def recovery_features(raw,p=RecoveryParameters()):
    shared=Parameters(day=p.day,week=p.week,shock=p.shock,participation=p.participation,
                      trail=p.trail,max_hours=p.max_hours,risk=p.risk,liquidity_cap=p.liquidity_cap)
    f=features(raw,shared).drop(columns='compression')
    g=raw.resample('1h')
    h=g.agg({'high':'max','low':'min','close':'last'})
    h.loc[g.close.count().ne(4),:]=np.nan
    h.index=f.index
    f['recovery']=(h.close-h.low)/(h.high-h.low).replace(0,np.nan)
    f['initial_stop']=h.low
    f['signal']=(f.shock<=-p.shock)&(f.participation>=p.participation)&(f.recovery>=p.recovery)
    f['signal'] &= h.close.notna().rolling(p.week+1).sum().eq(p.week+1)
    return f


if __name__=='__main__':
    run_research('recovery',recovery_features,RecoveryParameters())
