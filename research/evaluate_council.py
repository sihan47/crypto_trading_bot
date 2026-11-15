"""Frozen council technical proxy, individual experts, costs and correlations.

No historical GPT or news inference. Uses exactly the runtime feature code.
"""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from strategies.strategy_council import council_features, council_cards, council_decision, GROUPS
from research.evaluate_pair import curve, metrics


def main():
    report={'method':'Frozen COUNCIL_PROTOCOL.md; technical proxy excludes GPT; 2023-01-01 through 2026-11-13; each asset all-in; not fixed-lot live returns', 'assets':{}}
    for symbol in ['BTCUSDT','BNBUSDT']:
        raw=pd.read_csv(ROOT/f'data/research_15m/{symbol}.csv',index_col=0,parse_dates=True)
        raw.index=pd.to_datetime(raw.index,utc=True)
        g=raw.resample('4h')
        d=g.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
        d=d.loc[g.close.count().eq(16)]
        f=council_features(d)
        checks=0
        for end in np.linspace(1000,len(d),24,dtype=int):
            row=council_features(d.iloc[end-1000:end]).iloc[-1]
            for col in ['ready','regime']+[n+'_signal' for n in GROUPS]:
                assert row[col]==f.iloc[end-1][col], (symbol,end,col)
            checks+=1
        buy=[];sell=[];coverage=[]
        for _,row in f.iterrows():
            cards,market=council_cards(row)
            flat=council_decision(cards,market,False)
            held=council_decision(cards,market,True)
            buy.append(flat['action']=='BUY');sell.append(held['action']=='SELL');coverage.append(flat['coverage'])
        signals={'council_proxy':(pd.Series(buy,index=d.index),pd.Series(sell,index=d.index))}
        signals.update({name:(f[name+'_signal'].eq(1),f[name+'_signal'].eq(-1)) for name in GROUPS})
        raw=raw.loc[raw.index>='2023-01-01']
        result={'rolling_window_checks':checks,'strategies':{}}
        returns={}
        for name,(a,b) in signals.items():
            a=a.copy();b=b.copy();a.index+=pd.Timedelta(hours=3,minutes=45);b.index+=pd.Timedelta(hours=3,minutes=45)
            a=a.reindex(raw.index,fill_value=False);b=b.reindex(raw.index,fill_value=False)
            eq,trades=curve(raw,a,b)
            stress,_=curve(raw,a,b,fee=.002,slip=.001)
            yearly={}
            for year,part in raw.groupby(raw.index.year):
                e,t=curve(part,a.loc[part.index],b.loc[part.index])
                yearly[str(year)]={**metrics(e),'round_trips':t}
            result['strategies'][name]={**metrics(eq),'round_trips':trades,'double_cost':metrics(stress),'yearly_reset_cash':yearly}
            returns[name]=eq.resample('1D').last().ffill().pct_change(fill_method=None)
        r=pd.DataFrame(returns).dropna()
        result['daily_return_correlations']=r.corr().round(4).to_dict()
        bad=r[list(GROUPS)].lt(0).any(axis=1)
        result['correlations_on_any_expert_loss_days']=r.loc[bad].corr().round(4).to_dict()
        result['nonzero_coverage_fraction']=float(np.mean(np.array(coverage)>=.2))
        report['assets'][symbol]=result
        print(symbol,json.dumps(result['strategies']['council_proxy'],indent=2),flush=True)
    report['available_after']=pd.Timestamp.now(tz='UTC').isoformat()
    (ROOT/'research/council_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
