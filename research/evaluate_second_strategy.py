import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from strategies.pullback_swing import pullback_features
from backtesting.next_bar import simulate


def main():
    report={'protocol':'SECOND_STRATEGY_PROTOCOL.md','symbols':{}}
    for symbol in ['BTCUSDT','BNBUSDT']:
        raw=pd.read_csv(ROOT/f'data/research_15m/{symbol}.csv',index_col=0,parse_dates=True)
        raw.index=pd.to_datetime(raw.index,utc=True)
        results={}
        for tf,count,bpd in [('4h',16,6),('1D',96,1)]:
            g=raw.resample(tf,label='left',closed='left')
            df=g.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
            df=df.loc[g['close'].count().eq(count)]
            r=pullback_features(df,bpd)
            z=(df.close-df.close.ewm(span=65*bpd,adjust=False,min_periods=65*bpd).mean())/df.close.rolling(65*bpd).std(ddof=0).replace(0,np.nan)
            for name,a,b in [('rsi2_pullback',r.entries,r.exits),('zscore_trend',z>.5,z<-.5)]:
                block=df.loc[df.index>='2023-01-01']
                metrics={'full':simulate(block,a.loc[block.index],b.loc[block.index]),
                         'double_cost':simulate(block,a.loc[block.index],b.loc[block.index],fee=.002,slippage=.001),
                         'years':{}}
                for year in [2023,2024,2025,2026]:
                    block=df.loc[df.index.year==year]
                    metrics['years'][str(year)]=simulate(block,a.loc[block.index],b.loc[block.index])
                results[f'{tf}:{name}']=metrics
                print(symbol,tf,name,metrics,flush=True)
        report['symbols'][symbol]=results
    (ROOT/'research/second_strategy_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
