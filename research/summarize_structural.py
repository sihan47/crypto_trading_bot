"""Export research figures and descriptive, serial-dependence-aware uncertainty."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def bootstrap(eq,replicates=1000,block=7):
    d=eq.resample('1D').last().ffill().pct_change(fill_method=None)
    d.iloc[0]=eq.resample('1D').last().iloc[0]/10000-1
    x=d.to_numpy();rng=np.random.default_rng(20261115);values=[]
    for _ in range(replicates):
        starts=rng.integers(0,len(x),size=int(np.ceil(len(x)/block)))
        sample=x[(starts[:,None]+np.arange(block))%len(x)].ravel()[:len(x)]
        if sample.std(ddof=1)>0:
            values.append(sample.mean()/sample.std(ddof=1)*np.sqrt(365))
    return {'method':'Descriptive circular 7-day block bootstrap, 1000 resamples; not selection-adjusted inference.',
            'sharpe_95pct_interval':np.quantile(values,[.025,.975]).tolist(),'valid_resamples':len(values)}


def main():
    fig,axs=plt.subplots(3,2,figsize=(13,11),layout='constrained')
    reports={}
    names={'compression':'A: Compression expansion','recovery':'B: Sell-shock recovery','carry':'C: Spot / perpetual carry'}
    for i,(key,title) in enumerate(names.items()):
        folder=ROOT/'output'/f'structural_{key}'
        full=pd.read_csv(folder/'full_equity.csv',index_col=0,parse_dates=True).iloc[:,0]
        hold=pd.read_csv(folder/'holdout_equity.csv',index_col=0,parse_dates=True).iloc[:,0]
        tr=pd.read_csv(folder/'holdout_trades.csv')
        report=json.loads((ROOT/'research'/f'structural_{key}_evaluation.json').read_text(encoding='utf-8'))
        report['holdout_bootstrap']=bootstrap(hold)
        reports[key]=report
        axs[i,0].plot(full.index,(full/10000-1)*100,color=['#2378a8','#a84a35','#558236'][i])
        axs[i,0].axvspan(pd.Timestamp('2025-01-01',tz='UTC'),full.index[-1],color='#eee0a5',alpha=.45,label='Exposed-history holdout')
        axs[i,0].axhline(0,color='gray',lw=.7)
        axs[i,0].set_title(title+' | net equity');axs[i,0].set_ylabel('Return (%)');axs[i,0].grid(alpha=.2)
        axs[i,0].legend(fontsize=8)
        axs[i,1].hist(tr.net_pnl,bins=min(20,max(5,len(tr)//3)),color=['#2378a8','#a84a35','#558236'][i],alpha=.8)
        axs[i,1].axvline(0,color='black',lw=.8)
        h=report['periods']['holdout']
        axs[i,1].set_title(f"Holdout trades: n={h['trades']}, Sharpe={h['sharpe']:.2f}")
        axs[i,1].set_xlabel('Net trade PnL (USDT), initial equity 10,000');axs[i,1].set_ylabel('Count')
    fig.suptitle('BTC structural hypotheses: all rejected under predeclared criteria\nFees + slippage + actual funding for carry; no historical GPT',fontsize=14)
    fig.savefig(ROOT/'output/structural_research.png',dpi=170)
    plt.close(fig)
    summary={k:{'holdout':r['periods']['holdout'],'bootstrap':r['holdout_bootstrap'],'tests':r['tests'],
                'sensitivity_positive':sum(v['return_pct']>0 for v in r['sensitivity'].values()),
                'sensitivity_cases':len(r['sensitivity']),
                'double_cost':r.get('double_cost',r.get('double_cost_holdout'))} for k,r in reports.items()}
    (ROOT/'research/structural_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:{'bootstrap':v['bootstrap'],'positive_sensitivity':v['sensitivity_positive']} for k,v in summary.items()},indent=2))


if __name__=='__main__': main()
