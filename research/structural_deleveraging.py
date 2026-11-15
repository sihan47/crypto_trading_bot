"""Iteration D: OI quantity + price shock + pre-event account crowding."""
from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from research.structural_btc import Parameters,features,run_research,simulate,metrics
ROOT=Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DeleveragingParameters:
    day: int=24
    week: int=168
    shock: float=2.
    crowding_tail: float=.25
    trail: float=2.
    max_hours: int=24
    risk: float=.005
    liquidity_cap: float=.001


def read_panel():
    panel=pd.read_csv(ROOT/'data/research_structural/btc_hourly_research.csv',index_col=0)
    panel.index=pd.to_datetime(panel.index,utc=True)
    for col in ['create_time','metric_available_at']:
        panel[col]=pd.to_datetime(panel[col],utc=True,errors='coerce')
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing: raise ValueError('Invalid panel timestamps')
    return panel


def signals_from_panel(panel,p):
    if ((panel.metric_available_at>panel.index)&panel.metric_available_at.notna()).any():
        raise ValueError('Future metric availability')
    oi=np.log(panel.sum_open_interest.where(panel.sum_open_interest>0)).diff()
    contiguous=panel.create_time.diff().eq(pd.Timedelta(hours=1))
    oi=oi.where(contiguous)
    oi_sigma=oi.rolling(p.week).std().shift(1).replace(0,np.nan)
    price_ret=np.log(panel.spot_close.where(panel.spot_tradeable.eq(True))).diff()
    price_sigma=price_ret.rolling(p.week).std().shift(1).replace(0,np.nan)
    event_end=(panel.create_time+pd.Timedelta(minutes=5)).dt.ceil('1h')
    # Reindex to EVENT time. NaT/missing periods remain NaN, never backfilled.
    event_price_z=(price_ret/price_sigma).reindex(pd.DatetimeIndex(event_end)).to_numpy()
    event_low=panel.spot_low.where(panel.spot_tradeable.eq(True)).reindex(pd.DatetimeIndex(event_end)).to_numpy()
    crowd=panel.count_long_short_ratio.shift(1)
    cutoff=panel.count_long_short_ratio.rolling(p.week).quantile(1-p.crowding_tail).shift(1)
    f=pd.DataFrame(index=panel.index)
    f['oi_z']=oi/oi_sigma;f['price_z']=event_price_z
    f['pre_event_crowding']=crowd;f['crowding_cutoff']=cutoff
    f['initial_stop']=event_low
    f['signal']=(f.oi_z<=-p.shock)&(f.price_z<=-p.shock)&(crowd>cutoff)
    f['signal'] &= event_end.le(panel.index)&panel.spot_tradeable.eq(True)&f.initial_stop.notna()
    return f


def deleveraging_features(raw,p=DeleveragingParameters(),panel=None,extra_delay_hours=0):
    panel=read_panel() if panel is None else panel
    signals=signals_from_panel(panel,p)
    signals.index+=pd.Timedelta(hours=extra_delay_hours)-pd.Timedelta(minutes=15)
    common=Parameters(day=p.day,week=p.week,trail=p.trail,max_hours=p.max_hours,risk=p.risk,liquidity_cap=p.liquidity_cap)
    baseline=features(raw,common)
    # Risk/diagnostic context is known at execution-decision time. Entry
    # evidence (including event low) is delayed as one inseparable record.
    f=baseline.copy()
    evidence=signals.reindex(f.index)
    for col in ['oi_z','price_z','pre_event_crowding','crowding_cutoff','initial_stop']:
        f[col]=evidence[col]
    f['signal']=evidence.signal.eq(True)&f.sigma_day.notna()&f.shock.notna()
    return f


def main():
    panel=read_panel();p=DeleveragingParameters()
    report=run_research('deleveraging',lambda raw,params:deleveraging_features(raw,params,panel),p)
    raw=pd.read_csv(ROOT/'data/research_15m/BTCUSDT.csv',index_col=0,parse_dates=True)
    raw.index=pd.to_datetime(raw.index,utc=True);raw=raw.loc[raw.volume>0]
    hold=raw.loc[raw.index>='2025-01-01']
    report['extra_data_delays']={}
    for hours in [1,2]:
        f=deleveraging_features(raw,p,panel,extra_delay_hours=hours)
        eq,tr,cost=simulate(hold,f,p)
        report['extra_data_delays'][str(hours)]=metrics(eq,tr,cost)
    report['tests']['extra_data_delays_positive']=all(x['return_pct']>0 for x in report['extra_data_delays'].values())
    report['passed']=all(report['tests'].values())
    (ROOT/'research/structural_deleveraging_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('Delay tests:',json.dumps(report['extra_data_delays'],indent=2),flush=True)


if __name__=='__main__': main()
