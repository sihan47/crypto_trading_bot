"""Run the predeclared RESEARCH_PROTOCOL.md without contacting any account."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import yaml
from backtesting.next_bar import simulate
from research.candidate_rules import candidate_signals, RULES
from strategies.regime_ensemble import EnsembleParams, family_features, score_families
from strategies.macd_strategy import MACDParams, run_macd_strategy


def main():
    cfg=yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
    ep=EnsembleParams(**cfg['strategy']['params']['ensemble'])
    report={'protocol':'RESEARCH_PROTOCOL.md','rules':RULES,'symbols':{},'walk_forward':{}}
    cache={}
    for symbol in ['BTCUSDT','BNBUSDT']:
        raw=pd.read_csv(ROOT/f'data/research_15m/{symbol}.csv',index_col=0,parse_dates=True)
        raw.index=pd.to_datetime(raw.index,utc=True)
        if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
            raise ValueError('Unsorted/duplicate data')
        metrics={}
        cache[symbol]={}
        for tf,count,bpd in [('1h',4,24),('4h',16,6)]:
            groups=raw.resample(tf,label='left',closed='left')
            df=groups.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
            df=df.loc[groups['close'].count().eq(count)]
            signals=candidate_signals(df,bpd)
            old=run_macd_strategy(df['close'],MACDParams())
            signals['legacy_inverse_macd']=(old['entries'],old['exits'])
            if tf=='4h':
                e=score_families(family_features(df,ep),ep)
                signals['current_ensemble']=(e['entries'],e['exits'])
                signals['buy_hold']=(pd.Series(True,index=df.index),pd.Series(False,index=df.index))
            for name,(a,b) in signals.items():
                key=f'{tf}:{name}'
                cache[symbol][key]=(df,a,b)
                block=df.loc[df.index>='2023-01-01']
                result={'full':simulate(block,a.loc[block.index],b.loc[block.index]),'years':{}}
                for year in [2023,2024,2025,2026]:
                    block=df.loc[df.index.year==year]
                    result['years'][str(year)]=simulate(block,a.loc[block.index],b.loc[block.index])
                metrics[key]=result
            print('Completed',symbol,tf,flush=True)
        report['symbols'][symbol]=metrics
    names=[f'{tf}:{name}' for tf in ['1h','4h'] for name in RULES]
    for year in [2024,2025,2026]:
        training=list(range(2023,year))
        scores={'cash':0.}
        for name in names:
            if any(sum(report['symbols'][s][name]['years'][str(y)]['round_trips'] for y in training)<3
                   for s in report['symbols']):
                continue
            rows=[report['symbols'][s][name]['years'][str(y)] for s in report['symbols'] for y in training]
            scores[name]=sum(x['return_pct']+.5*x['max_drawdown_pct'] for x in rows)/len(rows)
        selected=max(scores,key=scores.get)
        fold={'training_years':training,'scores':scores,'selected':selected,'test':{}}
        for symbol in report['symbols']:
            key='4h:buy_hold' if selected=='cash' else selected
            df,a,b=cache[symbol][key]
            block=df.loc[df.index.year==year]
            if selected=='cash':
                a=b=pd.Series(False,index=df.index)
            fold['test'][symbol]={
                'normal_cost':simulate(block,a.loc[block.index],b.loc[block.index]),
                'double_cost':simulate(block,a.loc[block.index],b.loc[block.index],fee=.002,slippage=.001),
                'current_ensemble':report['symbols'][symbol]['4h:current_ensemble']['years'][str(year)],
                'buy_hold':report['symbols'][symbol]['4h:buy_hold']['years'][str(year)],
            }
        report['walk_forward'][str(year)]=fold
        print('Forward',year,selected,{s:round(v['normal_cost']['return_pct'],2) for s,v in fold['test'].items()},flush=True)
    path=ROOT/'research/candidate_evaluation.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(path,flush=True)


if __name__=='__main__':
    main()
