"""Rare impulse EDA on the discovery partition only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from scipy.stats import skew
from research.mine_anomalies import engineer,outcomes,contrast,bh
ROOT=Path(__file__).resolve().parents[1]


def discovery():
    meta=json.loads((ROOT/'research/anomaly_findings.json').read_text())
    d=pd.read_csv(ROOT/'data/research_structural/btc_orthogonal_1h.csv',index_col=0,nrows=meta['discovery_rows'])
    d.index=pd.to_datetime(d.index,utc=True)
    if d.index.max()>=pd.Timestamp(meta['reserved_start']):raise ValueError('Reserved boundary breached')
    return d


def impulses(d,scale=1.):
    tr=pd.concat([d.futures_high-d.futures_low,(d.futures_high-d.futures_close.shift()).abs(),
                  (d.futures_low-d.futures_close.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(24).mean().shift(1)
    end=(pd.to_datetime(d.create_time,utc=True)+pd.Timedelta(minutes=5)).dt.ceil('h')
    result={}
    for h in [1,3]:
        oi=d.sum_open_interest/d.sum_open_interest.shift(h)-1
        contiguous=(pd.to_datetime(d.create_time,utc=True).diff(h)==pd.Timedelta(hours=h))
        move=(d.futures_close-d.futures_open.shift(h-1))/atr
        flow=(2*d.futures_taker_buy_base_asset_volume.rolling(h).sum()/d.futures_volume.rolling(h).sum()-1)
        cutoff=flow.rolling(720).quantile(.01).shift(1)
        event=pd.DataFrame({'move':move,'flow':flow,'cutoff':cutoff,
                            'low':d.spot_low.rolling(h).min()},index=d.index).reindex(pd.DatetimeIndex(end))
        event.index=d.index
        common=contiguous&end.le(d.index)&pd.to_datetime(d.metric_available_at,utc=True).le(d.index)
        result[f'cascade_{h}h']=(common&(oi<-.03*scale)&(event.move< -2.5*scale)&(event.flow<event.cutoff)).fillna(False)
        extreme=d.funding_rate>d.funding_rate.rolling(720).quantile(.99).shift(1)
        result[f'exhaustion_{h}h']=(common&(oi>.03*scale)&(event.move>2.5*scale)&extreme&d.funding_rate.gt(0)).fillna(False)
        result[f'low_{h}h']=event.low
    return pd.DataFrame(result,index=d.index)


def main():
    d=discovery();events=impulses(d);_,regime,ret=engineer(d);rows=[]
    for name in [x for x in events if not x.startswith('low')]:
        e=events[name];frequency=float(e.mean())
        for h in [4,12]:
            y=outcomes(ret,h)['return']
            for state in ['all','bull','bear']:
                valid=y.notna() if state=='all' else y.notna()&regime.eq(state)
                values=y[e&valid];times=values.index;spaced=[]
                for stamp in times:
                    if not spaced or stamp-spaced[-1]>=pd.Timedelta(hours=h):spaced.append(stamp)
                test=contrast(y,e,valid)
                rows.append({'event':name,'frequency':frequency,'rare_under_2pct':frequency<.02,'horizon':h,'regime':state,
                    'n':len(values),'nonoverlap_n':len(spaced),'mean':float(values.mean()) if len(values) else None,
                    'median':float(values.median()) if len(values) else None,'skew':float(skew(values)) if len(values)>2 else None,
                    't_hac':test['t_hac'] if test else None,'p':test['p'] if test else None,
                    'inference':'HAC contrast' if test else 'Insufficient for pre-existing 100-event inference floor; descriptive only'})
    tested=[x for x in rows if x['p'] is not None]
    if tested:
        q=np.minimum(bh([x['p'] for x in tested])*sum(1/np.arange(1,len(tested)+1)),1)
        for x,v in zip(tested,q):x['q_by']=float(v)
    report={'discovery_end':str(d.index.max()),'spec':'1h/3h price move +/-2.5 prior 24h ATR; OI +/-3%; cascade flow below past 30d 1% quantile; exhaustion positive funding above past 30d 99% quantile. Metrics aligned to event hour with conservative availability. No threshold search.','results':rows}
    (ROOT/'research/impulse_findings.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
