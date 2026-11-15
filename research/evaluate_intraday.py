"""Exploratory comparison, NOT a fresh blind holdout after earlier research."""
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from strategies.intraday_strategy import IntradayParams, intraday_features, signals_from_features
from backtesting.next_bar import simulate

PERIODS={
    '2023':('2023-01-01','2024-01-01'),
    '2024':('2024-01-01','2025-01-01'),
    '2025':('2025-01-01','2026-01-01'),
    '2026_later':('2026-05-02','2026-11-14'),
}


def main():
    params=IntradayParams()
    report={'method':{'params':asdict(params),'periods':PERIODS,
                      'execution':'next-open; single fully allocated long position; costs both sides',
                      'limitations':['Previously inspected historical periods: exploratory, not blind validation.',
                                     'No GPT replay; live fixed-lot sizing and polling delay differ.',
                                     'Stops evaluated at bar close, not guaranteed intrabar stop prices.']},'symbols':{}}
    for symbol in ['BTCUSDT','BNBUSDT']:
        df=pd.read_csv(ROOT/f'data/research_15m/{symbol}.csv',index_col=0,parse_dates=True)
        df.index=pd.to_datetime(df.index,utc=True)
        features=intraday_features(df,params)
        variants={}
        for name in ['pullback_only','squeeze_only','combined']:
            f=features.copy()
            if name=='pullback_only':
                f['squeeze_setup']=False
            if name=='squeeze_only':
                f['pullback_setup']=False
            variants[name]=signals_from_features(f,params)
        results={}
        periods={'full':(df.index[0],df.index[-1]+pd.Timedelta(minutes=15)),**PERIODS}
        for period,(start,end) in periods.items():
            block=df.loc[(df.index>=start)&(df.index<end)]
            results[period]={}
            for name,signals in variants.items():
                s=signals.loc[block.index]
                metrics=simulate(block,s['entries'],s['exits'])
                metrics['pullback_entries']=int((s['entries']&s['setup'].eq('pullback')).sum())
                metrics['squeeze_entries']=int((s['entries']&s['setup'].eq('squeeze')).sum())
                results[period][name]=metrics
            results[period]['buy_hold']=simulate(block,pd.Series(True,index=block.index),pd.Series(False,index=block.index))
        block=df.loc[(df.index>=PERIODS['2026_later'][0])&(df.index<PERIODS['2026_later'][1])]
        s=variants['combined'].loc[block.index]
        report['symbols'][symbol]={
            'data_range':{'first_open':df.index[0].isoformat(),'last_open':df.index[-1].isoformat(),
                          'bars':len(df),'gaps':int(df.index.to_series().diff().dropna().ne(pd.Timedelta(minutes=15)).sum())},
            'periods':results,'double_costs':simulate(block,s['entries'],s['exits'],fee=.002,slippage=.001)}
        print(symbol,json.dumps({'data_range':report['symbols'][symbol]['data_range'],'full':results['full']}),flush=True)
    path=ROOT/'research/intraday_evaluation.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(path)


if __name__=='__main__':
    main()
