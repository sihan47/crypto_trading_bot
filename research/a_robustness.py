"""Predeclared A diagnostics, discovery data only; no production imports."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import io
import json
import numpy as np
import pandas as pd
from research.impulse_scan import discovery
from research.archetype_tournament import Spec,walk,signals
ROOT=Path(__file__).resolve().parents[1]


def a_features(d,normalization=720,macro=2160,risk_window=24,data_delay=0,signal_delay=0,ablation=None):
    def z(x):
        past=x.rolling(normalization).agg(['mean','std']).shift(1)
        return (x-past['mean'])/past['std'].replace(0,np.nan)
    funding=d.funding_rate.shift(data_delay)
    oi=np.log(d.sum_open_interest.where(d.sum_open_interest.gt(0)).shift(data_delay))
    divergence=np.log(d.sum_toptrader_long_short_ratio.where(d.sum_toptrader_long_short_ratio>0).shift(data_delay))-np.log(d.count_long_short_ratio.where(d.count_long_short_ratio>0).shift(data_delay))
    price=d.futures_close.where(d.futures_volume.gt(0))
    bull=price.gt(price.shift(macro))&price.shift(macro).notna()
    low_funding=z(funding).lt(0);positive_divergence=z(divergence).gt(0)
    event=bull&low_funding&positive_divergence
    if ablation=='bull_only':event=bull
    elif ablation=='funding_only':event=bull&low_funding
    elif ablation=='divergence_only':event=bull&positive_divergence
    # Identical valid-sample mask for ablations; missing data is not an advantage.
    ready=z(funding).notna()&z(divergence).notna()&price.shift(macro).notna()&oi.notna()&oi.rolling(normalization).mean().shift(1).notna()
    variance=(.5*np.log(d.spot_high/d.spot_low)**2-(2*np.log(2)-1)*np.log(d.spot_close/d.spot_open)**2).clip(lower=0)
    sigma=np.sqrt(variance.rolling(risk_window).mean()*risk_window)
    event=(event&ready&sigma.notna()).shift(signal_delay,fill_value=False)
    vol=np.sqrt(np.log(price).diff().pow(2).rolling(168).mean())
    frame=pd.DataFrame({'signal':event&sigma.notna(),'sigma':sigma,
        'stop':d.spot_close*np.exp(-2*sigma),'limit':np.nan,'target':np.nan,
        'regime':pd.Series(np.where(bull,'bull','bear'),index=d.index).where(price.shift(macro).notna()),
        'vol_regime':pd.Series(np.where(vol>vol.rolling(720).median().shift(1),'high','low'),index=d.index).where(vol.notna())},index=d.index)
    frame.index-=pd.Timedelta(minutes=15)
    return frame


def read_prices(d):
    lines=[]
    with (ROOT/'data/research_15m/BTCUSDT.csv').open() as source:
        lines.append(next(source))
        for line in source:
            stamp=pd.Timestamp(line.split(',',1)[0])
            if stamp+pd.Timedelta(minutes=15)>d.index.max():break
            if stamp>=d.index.min()-pd.Timedelta(hours=1):lines.append(line)
    raw=pd.read_csv(io.StringIO(''.join(lines)),index_col=0)
    raw.index=pd.to_datetime(raw.index,utc=True)
    return raw.loc[raw.volume>0]


def bootstrap(eq,seed=20261115,reps=2000):
    daily=eq.resample('D').last().ffill().pct_change();daily.iloc[0]=eq.resample('D').last().iloc[0]/10000-1
    x=daily.to_numpy();rng=np.random.default_rng(seed);n=len(x);out=[]
    for _ in range(reps):
        starts=rng.integers(0,n,size=int(np.ceil(n/7)))
        y=x[((starts[:,None]+np.arange(7))%n).ravel()[:n]]
        out.append([np.prod(1+y)-1,y.mean()/y.std(ddof=1)*np.sqrt(365) if y.std()>0 else 0])
    a=np.asarray(out)
    return {'method':'Circular 7-day moving-block bootstrap of daily returns, 2000 draws; descriptive, no selection-bias correction',
            'return_95pct':(np.quantile(a[:,0],[.025,.975])*100).tolist(),
            'sharpe_95pct':np.quantile(a[:,1],[.025,.975]).tolist(),
            'positive_return_fraction':float((a[:,0]>0).mean())}


def main():
    d=discovery();raw=read_prices(d);base=a_features(d)
    original=signals(d)['A']
    # Algebraic reduction must reproduce the established A before new tests.
    pd.testing.assert_series_equal(base.signal,original.signal)
    np.testing.assert_allclose(base.stop,original.stop,equal_nan=True)
    cases={'base':{}}
    for key,value in [('normalization',720),('macro',2160),('risk_window',24)]:
        for scale in [.8,1.2]:cases[f'{key}_{scale}']={key:round(value*scale)}
    for scale in [.8,1.2]:cases[f'windows_joint_{scale}']={k:round(v*scale) for k,v in [('normalization',720),('macro',2160),('risk_window',24)]}
    for hours in [1,2]:
        cases[f'derivatives_delay_{hours}h']={'data_delay':hours}
        cases[f'entry_delay_{hours}h']={'signal_delay':hours}
    for name in ['bull_only','funding_only','divergence_only']:cases['ablation_'+name]={'ablation':name}
    report={'discovery_end':str(d.index.max()),'specification':'Original A unchanged; fixed delays/windows/ablations, no winning variant selected. OI-positive-weight sign cancels exactly.','cases':{}}
    out=ROOT/'output/a_robustness';out.mkdir(parents=True,exist_ok=True)
    keep=['return_pct','sharpe','sortino','max_dd_pct','profit_factor','win_rate','avg_trade_pnl','trades','net_without_best_5pct','folds']
    for name,kwargs in cases.items():
        r,eq,tr=walk(raw,a_features(d,**kwargs),'A',Spec())
        report['cases'][name]={k:r[k] for k in keep}
        if name=='base':
            eq.to_csv(out/'base_equity.csv');tr.to_csv(out/'base_trades.csv',index=False)
            report['bootstrap']=bootstrap(eq)
            positive=tr.net_pnl.clip(lower=0).sum();ranked=tr.sort_values('net_pnl',ascending=False)
            report['concentration']={'top3_share_of_gross_profit':float(ranked.net_pnl.head(3).sum()/positive),
                'top3':ranked.head(3).to_dict('records'),
                'pnl_without_each_fold':{str(k):float(tr.loc[tr.fold.ne(k),'net_pnl'].sum()) for k in range(1,9)},
                'year_pnl':tr.groupby(pd.to_datetime(tr.entry_time,utc=True).dt.year).net_pnl.agg(['count','sum']).to_dict('index')}
        print(name,round(r['return_pct'],4),round(r['sharpe'],4),r['trades'],flush=True)
    (ROOT/'research/a_robustness_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
