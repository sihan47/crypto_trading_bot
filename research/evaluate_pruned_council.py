"""One declared ablation, not a search over highest-return subsets."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from strategies.strategy_council import council_features,council_cards,council_decision,EXCLUDED
from research.evaluate_pair import curve,metrics


def main():
    report={'method':'Remove BNB Donchian only; preserve survivor relative weights; same historical sample, no GPT, no unseen validation.',
            'exclusions':EXCLUDED,'assets':{}}
    for symbol in ['BTCUSDT','BNBUSDT']:
        raw=pd.read_csv(ROOT/f'data/research_15m/{symbol}.csv',index_col=0,parse_dates=True)
        raw.index=pd.to_datetime(raw.index,utc=True)
        g=raw.resample('4h')
        d=g.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
        d=d.loc[g.close.count().eq(16)]
        f=council_features(d)
        results={}
        for label,asset in [('before',None),('after',symbol)]:
            buys=[];sells=[]
            for _,row in f.iterrows():
                cards,market=council_cards(row,asset)
                buys.append(council_decision(cards,market,False)['action']=='BUY')
                sells.append(council_decision(cards,market,True)['action']=='SELL')
            period=raw.loc[raw.index>='2023-01-01']
            index=d.index+pd.Timedelta(hours=3,minutes=45)
            a=pd.Series(buys,index=index).reindex(period.index,fill_value=False)
            b=pd.Series(sells,index=index).reindex(period.index,fill_value=False)
            eq,trades=curve(period,a,b)
            stress,_=curve(period,a,b,fee=.002,slip=.001)
            years={}
            for year,part in period.groupby(period.index.year):
                e,t=curve(part,a.loc[part.index],b.loc[part.index])
                years[str(year)]={**metrics(e),'round_trips':t}
            results[label]={**metrics(eq),'round_trips':trades,'double_cost':metrics(stress),'yearly_reset_cash':years}
        report['assets'][symbol]=results
        print(symbol,json.dumps(results,indent=2),flush=True)
    report['available_after']=pd.Timestamp.now(tz='UTC').isoformat()
    (ROOT/'research/pruned_council_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__': main()
