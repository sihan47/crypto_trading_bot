"""Iteration E: price impulse + OI contraction + local taker buy dominance."""
from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from research.structural_btc import Parameters,features,run_research,simulate,metrics
from research.structural_deleveraging import read_panel
ROOT=Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CoverParameters:
    day: int=24
    week: int=168
    price_shock: float=2.
    oi_shock: float=1.
    taker_ratio: float=2.
    trail: float=2.
    max_hours: int=24
    risk: float=.005
    liquidity_cap: float=.001


def covering_signals(panel,p=CoverParameters()):
    if ((panel.metric_available_at>panel.index)&panel.metric_available_at.notna()).any():
        raise ValueError('Future metric availability')
    oi=np.log(panel.sum_open_interest.where(panel.sum_open_interest>0)).diff()
    oi=oi.where(panel.create_time.diff().eq(pd.Timedelta(hours=1)))
    oi_z=oi/oi.rolling(p.week).std().shift(1).replace(0,np.nan)
    ret=np.log(panel.spot_close.where(panel.spot_tradeable.eq(True))).diff()
    price_z=ret/ret.rolling(p.week).std().shift(1).replace(0,np.nan)
    event_end=(panel.create_time+pd.Timedelta(minutes=5)).dt.ceil('1h')
    f=pd.DataFrame(index=panel.index)
    f['oi_z']=oi_z
    f['price_z']=price_z.reindex(pd.DatetimeIndex(event_end)).to_numpy()
    f['taker_ratio']=panel.sum_taker_long_short_vol_ratio.where(np.isfinite(panel.sum_taker_long_short_vol_ratio))
    f['initial_stop']=panel.spot_low.where(panel.spot_tradeable.eq(True)).reindex(pd.DatetimeIndex(event_end)).to_numpy()
    f['signal']=(f.oi_z<=-p.oi_shock)&(f.price_z>=p.price_shock)&(f.taker_ratio>=p.taker_ratio)
    f['signal'] &= event_end.le(panel.index)&panel.spot_tradeable.eq(True)&f.initial_stop.notna()
    return f


def covering_features(raw,p=CoverParameters(),panel=None,extra_delay_hours=0):
    panel=read_panel() if panel is None else panel
    signals=covering_signals(panel,p)
    signals.index+=pd.Timedelta(hours=extra_delay_hours)-pd.Timedelta(minutes=15)
    common=Parameters(day=p.day,week=p.week,trail=p.trail,max_hours=p.max_hours,risk=p.risk,liquidity_cap=p.liquidity_cap)
    f=features(raw,common)
    evidence=signals.reindex(f.index)
    for col in ['oi_z','price_z','taker_ratio','initial_stop']: f[col]=evidence[col]
    f['signal']=evidence.signal.eq(True)&f.sigma_day.notna()&f.shock.notna()
    return f


def main():
    panel=read_panel();p=CoverParameters()
    report=run_research('short_covering',lambda raw,params:covering_features(raw,params,panel),p)
    raw=pd.read_csv(ROOT/'data/research_15m/BTCUSDT.csv',index_col=0,parse_dates=True)
    raw.index=pd.to_datetime(raw.index,utc=True);raw=raw.loc[raw.volume>0]
    hold=raw.loc[raw.index>='2025-01-01']
    report['extra_data_delays']={}
    for hours in [1,2]:
        f=covering_features(raw,p,panel,extra_delay_hours=hours)
        eq,tr,cost=simulate(hold,f,p)
        report['extra_data_delays'][str(hours)]=metrics(eq,tr,cost)
    report['tests']['extra_data_delays_positive']=all(x['return_pct']>0 for x in report['extra_data_delays'].values())
    report['passed']=all(report['tests'].values())
    (ROOT/'research/structural_short_covering_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('Delays:',json.dumps(report['extra_data_delays'],indent=2),flush=True)


if __name__=='__main__': main()
