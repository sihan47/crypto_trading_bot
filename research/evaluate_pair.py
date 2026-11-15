"""Two fixed strategies, observed on 15m prices; 50/50 capital, no rebalancing."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from strategies.band_trend import band_features
from strategies.pullback_swing import pullback_features


def curve(df,a,b,fee=.001,slip=.0005):
    opens=df.open.to_numpy();closes=df.close.to_numpy();times=df.index.asi8
    buys=a.to_numpy(dtype=bool);sells=b.to_numpy(dtype=bool)
    equity=np.zeros(len(df));cash=10000.;qty=0.;trades=0
    step=pd.Timedelta(minutes=15).value
    for i in range(len(df)):
        if i and times[i]-times[i-1]==step:
            if qty and sells[i-1]:
                cash=qty*opens[i]*(1-slip)*(1-fee);qty=0.;trades+=1
            elif not qty and buys[i-1]:
                qty=cash/(1+fee)/(opens[i]*(1+slip));cash=0.
        equity[i]=cash+qty*closes[i]
    if qty:
        equity[-1]=qty*closes[-1]*(1-slip)*(1-fee);trades+=1
    return pd.Series(equity,index=df.index),trades


def metrics(eq):
    d=eq.resample('1D').last().ffill().pct_change(fill_method=None).dropna()
    return {'return_pct':float((eq.iloc[-1]/10000-1)*100),
            'max_drawdown_15m_pct':float((eq/eq.cummax().clip(lower=10000)-1).min()*100),
            'sharpe_daily':float(d.mean()/d.std()*np.sqrt(365)) if d.std()>0 else 0.}


def main():
    report={'method':'2023-01-01 through 2026-11-13; fixed BTC band50 and BNB daily RSI2; 15m close marks; 50/50 starting capital, no rebalance; excludes GPT','strategies':{}}
    eqs={}
    for symbol,tf,n,fn in [('BTCUSDT','4h',16,band_features),('BNBUSDT','1D',96,lambda d:pullback_features(d,1))]:
        raw=pd.read_csv(ROOT/f'data/research_15m/{symbol}.csv',index_col=0,parse_dates=True)
        raw.index=pd.to_datetime(raw.index,utc=True)
        g=raw.resample(tf,label='left',closed='left')
        d=g.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
        d=d.loc[g['close'].count().eq(n)]
        f=fn(d)
        # Runtime uses 1000 candles: compare sampled prefix/window signals.
        checked=0
        for end in np.linspace(1000,len(d),12,dtype=int):
            window=fn(d.iloc[end-1000:end])
            assert window.iloc[-1]['entries']==f.iloc[end-1]['entries']
            assert window.iloc[-1]['exits']==f.iloc[end-1]['exits']
            checked+=1
        f.index=f.index+pd.Timedelta(minutes=15*(n-1))
        raw=raw.loc[raw.index>='2023-01-01']
        a=f.entries.reindex(raw.index,fill_value=False);b=f.exits.reindex(raw.index,fill_value=False)
        eq,trades=curve(raw,a,b)
        stressed,_=curve(raw,a,b,fee=.002,slip=.001)
        eqs[symbol]=eq
        report['strategies'][symbol]={**metrics(eq),'round_trips':trades,'double_cost':metrics(stressed),'rolling_window_checks':checked}
    common=eqs['BTCUSDT'].index.intersection(eqs['BNBUSDT'].index)
    combo=.5*eqs['BTCUSDT'].loc[common]+.5*eqs['BNBUSDT'].loc[common]
    report['equal_capital_pair']=metrics(combo)
    daily=pd.DataFrame(eqs).resample('1D').last().ffill().pct_change(fill_method=None).dropna()
    report['daily_return_correlation']=float(daily.corr().iloc[0,1])
    (ROOT/'research/pair_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
