"""Fixed-universe daily breakout feasibility; no live trading or ranking hindsight."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SYMBOLS=['BTCUSDT','BNBUSDT','ETHUSDT','SOLUSDT','XRPUSDT','ADAUSDT']


def load():
    frames={}
    for symbol in SYMBOLS:
        d=pd.read_csv(ROOT/f'data/research_rotation/{symbol}.csv',index_col=0,parse_dates=True)
        d.index=pd.to_datetime(d.index,utc=True)
        g=d.resample('1D',label='left',closed='left')
        b=g.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum','quote_volume':'sum'})
        b=b.loc[g['close'].count().eq(24)]
        upper=b.high.rolling(20).max().shift()
        lower=b.low.rolling(10).min().shift()
        b['momentum']=b.close/b.close.shift(30)-1
        b['entry']=(b.close>upper)&(b.momentum>0)&(b.close>b.close.rolling(20).mean())
        b['entry']&=(b.quote_volume.rolling(20).mean()>=1e7)&((upper-lower)/b.close>=.01)
        b['exit']=b.close<lower
        frames[symbol]=b
    return frames


def simulate(frames,start='2023-01-01',end='2026-11-14',fee=.001,slip=.0005):
    # Common complete days; no interpolation and no fills across missing days.
    index=next(iter(frames.values())).index
    for df in frames.values():
        index=index.intersection(df.index)
    index=index[(index>=start)&(index<end)]
    cash=10000.;qty=0.;asset=None;entry_cash=0.;curve=[];trades=[];counts={s:0 for s in frames}
    held=0
    for i,stamp in enumerate(index):
        if i and stamp-index[i-1]==pd.Timedelta(days=1):
            previous=index[i-1]
            if asset and bool(frames[asset].loc[previous,'exit']):
                cash=qty*frames[asset].loc[stamp,'open']*(1-slip)*(1-fee)
                trades.append(cash/entry_cash-1);qty=0.;asset=None
            if asset is None:
                eligible=[s for s,d in frames.items() if bool(d.loc[previous,'entry'])]
                if eligible:
                    asset=max(eligible,key=lambda s:frames[s].loc[previous,'momentum'])
                    entry_cash=cash
                    qty=cash/(1+fee)/(frames[asset].loc[stamp,'open']*(1+slip));cash=0.
                    counts[asset]+=1
        held+=asset is not None
        curve.append(cash if asset is None else qty*frames[asset].loc[stamp,'close'])
    if asset:
        curve[-1]=qty*frames[asset].loc[index[-1],'close']*(1-slip)*(1-fee)
        trades.append(curve[-1]/entry_cash-1)
    eq=pd.Series(curve,index=index)
    returns=eq.pct_change().dropna()
    # Equal allocations held from the same second-day open to final close.
    bh=np.mean([d.loc[index[-1],'close']*(1-slip)*(1-fee)/(d.loc[index[1],'open']*(1+slip)*(1+fee)) for d in frames.values()])
    return {'return_pct':float((eq.iloc[-1]/10000-1)*100),
            'max_drawdown_pct':float((eq/eq.cummax().clip(lower=10000)-1).min()*100),
            'sharpe_daily':float(returns.mean()/returns.std()*np.sqrt(365)) if returns.std()>0 else 0.,
            'round_trips':len(trades),'entry_counts':counts,'exposure_pct':held/len(index)*100,
            'equal_weight_buy_hold_return_pct':float((bh-1)*100)}


def main():
    frames=load()
    report={'protocol':'FORUM_PROTOCOL.md','universes':{}}
    for name,subset in [('two_pairs',SYMBOLS[:2]),('six_pairs',SYMBOLS)]:
        data={s:frames[s] for s in subset}
        results={'full':simulate(data),'double_cost':simulate(data,fee=.002,slip=.001),'years':{}}
        for year in [2023,2024,2025,2026]:
            results['years'][str(year)]=simulate(data,start=f'{year}-01-01',end=f'{year+1}-01-01')
        report['universes'][name]=results
        print(name,results,flush=True)
    (ROOT/'research/rotation_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
