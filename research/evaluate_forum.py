import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from research.forum_breakout import hourly_features,simulate_forum


def main():
    report={'protocol':'FORUM_PROTOCOL.md','symbols':{}}
    for symbol in ['BTCUSDT','BNBUSDT']:
        df=pd.read_csv(ROOT/f'data/research_15m/{symbol}.csv',index_col=0,parse_dates=True)
        df.index=pd.to_datetime(df.index,utc=True)
        results={}
        for name,mode,variant in [('bb_only','session','bb'),('bb_vwap_session','session','combined'),
                                  ('bb_vwap_rolling24','rolling24','combined')]:
            f=hourly_features(df,mode)
            results[name]={'full':simulate_forum(df,f,variant),
                           'author_fee_sensitivity':simulate_forum(df,f,variant,fee=.00025,slippage=0),
                           'years':{}}
            for year in [2023,2024,2025,2026]:
                results[name]['years'][str(year)]=simulate_forum(df,f,variant,start=f'{year}-01-01',end=f'{year+1}-01-01')
            print(symbol,name,results[name]['full'],flush=True)
        report['symbols'][symbol]=results
    (ROOT/'research/forum_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
