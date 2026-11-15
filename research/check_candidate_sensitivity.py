"""Post-selection sensitivity diagnostic, NOT another parameter selection.

Freeze winner at 50 days / 1%. Check 40/60 days x 0.5/1.5% corners;
do not substitute a better-looking neighbour for the selected candidate.
"""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from backtesting.next_bar import simulate


def main():
    report={'purpose':'post-selection sensitivity, not blind validation or retuning','symbols':{}}
    for symbol in ['BTCUSDT','BNBUSDT']:
        path=ROOT/f'data/research_15m/{symbol}.csv'
        raw=pd.read_csv(path,index_col=0,parse_dates=True)
        raw.index=pd.to_datetime(raw.index,utc=True)
        g=raw.resample('4h',label='left',closed='left')
        df=g.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
        df=df.loc[g['close'].count().eq(16)]
        block=df.loc[df.index>='2023-01-01']
        rows={}
        for days,band in [(50,.01),(40,.005),(40,.015),(60,.005),(60,.015)]:
            avg=df.close.rolling(days*6).mean()
            a=(df.close>avg*(1+band)).loc[block.index]
            b=(df.close<avg*(1-band)).loc[block.index]
            rows[f'{days}d_{band:g}']={
                'normal_cost':simulate(block,a,b),
                'double_cost':simulate(block,a,b,fee=.002,slippage=.001),
            }
        report['symbols'][symbol]={'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'results':rows}
    out=ROOT/'research/candidate_sensitivity.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for s,d in report['symbols'].items():
        print(s,{k:{c:round(m['return_pct'],2) for c,m in v.items()} for k,v in d['results'].items()})


if __name__=='__main__':
    main()
