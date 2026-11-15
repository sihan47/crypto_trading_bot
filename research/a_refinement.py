"""Fixed three-arm IS refinement, continuous inventory across reporting windows."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from research.impulse_scan import discovery
from research.a_robustness import a_features,read_prices,bootstrap
from research.archetype_tournament import simulate,metrics,walk,Spec
from research.mine_anomalies import engineer
ROOT=Path(__file__).resolve().parents[1]


def daily_atr(d,days=14):
    h=d[['spot_high','spot_low','spot_close']].copy()
    h.index-=pd.Timedelta(hours=1)
    day=h.resample('D').agg({'spot_high':'max','spot_low':'min','spot_close':'last'})
    counts=h.notna().all(axis=1).resample('D').sum()
    day.loc[counts.ne(24),:]=np.nan
    previous=day.spot_close.shift()
    tr=pd.concat([day.spot_high-day.spot_low,(day.spot_high-previous).abs(),(day.spot_low-previous).abs()],axis=1).max(axis=1)
    tr=tr.where(day.notna().all(axis=1)&previous.notna())
    atr=tr.rolling(days).mean();atr.index+=pd.Timedelta(days=1)
    return atr.reindex(d.index,method='ffill',tolerance=pd.Timedelta(hours=23))


def frames(d,scale=.5,multiple=3.,days=14,negative_z=1.):
    core=a_features(d,ablation='divergence_only')
    f,_,_=engineer(d)
    atr=daily_atr(d,days)
    # Same eligibility for all three arms, including complete daily ATR history.
    core['signal'] &= atr.notna().to_numpy()
    fixed=core.copy();sized=core.copy()
    sized['risk_scale']=np.where(f.oi_weighted_funding.lt(0),1.,scale)
    full=sized.copy();full['daily_atr']=atr.to_numpy()
    full['stop']=(d.spot_close-multiple*atr).to_numpy()
    full['signal'] &= full.stop.gt(0)
    full['exit_signal']=(core.regime.eq('bear').to_numpy()|f.position_account_divergence.le(-negative_z).to_numpy())
    return {'control':fixed,'sizing_only':sized,'full':full}


def summarize(eq,tr):
    out=metrics(eq,tr);pnl=tr.net_pnl if len(tr) else pd.Series(dtype=float)
    gain=pnl.clip(lower=0).sum();best=pnl.nlargest(3).clip(lower=0).sum()
    out.update({'top3_gross_profit_share':float(best/gain) if gain else None,
                'net_without_top3':float(pnl.sum()-best),
                'max_holding_hours':float(tr.hours.max()) if len(tr) else 0,
                'trades_over_7days':int(tr.hours.gt(168).sum()) if len(tr) else 0,
                'exit_reasons':tr.groupby('reason').net_pnl.agg(['count','sum']).to_dict('index') if len(tr) else {},
                'bootstrap_7d':bootstrap(eq),'bootstrap_28d':bootstrap(eq,block_days=28)})
    return out


def main():
    d=discovery();raw=read_prices(d);start=raw.index.min()+pd.Timedelta(days=365)
    evaluation=raw.loc[raw.index>=start]
    variants=frames(d);report={'start':str(start),'end':str(d.index.max()),'arms':{},'sensitivity':{},
        'method':'Frozen three-arm discovery-only evaluation. Continuous positions across 90d reporting boundaries; only final sample liquidation. No OOS or production change. Spot 10bp fees +5bp slippage each side.'}
    folder=ROOT/'output/a_refinement';folder.mkdir(parents=True,exist_ok=True)
    for name,frame in variants.items():
        options={'no_time_stop':True,'chandelier':True} if name=='full' else {}
        eq,tr,_=simulate(evaluation,frame,'A',**options)
        result=summarize(eq,tr)
        eq.to_csv(folder/f'{name}_equity.csv');tr.to_csv(folder/f'{name}_trades.csv',index=False)
        # Reporting windows do not reset inventory or attribute open PnL to trades.
        windows=[];capital=10000.
        for _,part in eq.groupby(pd.Grouper(freq='90D',origin=start)):
            if len(part):
                windows.append({'start':str(part.index[0]),'end':str(part.index[-1]),'return_pct':float((part.iloc[-1]/capital-1)*100)})
                capital=part.iloc[-1]
        result['reporting_windows']=windows
        stressed,st, _=simulate(evaluation,frame,'A',cost_factor=2.,**options)
        result['double_cost']=metrics(stressed,st)
        reset,_,_=walk(raw,frame,'A',**options)
        result['forced_fold_liquidation']={k:reset[k] for k in ['return_pct','sharpe','max_dd_pct','trades']}
        report['arms'][name]=result
        print(name,json.dumps({k:result[k] for k in ['return_pct','sharpe','max_dd_pct','trades','top3_gross_profit_share','max_holding_hours']}),flush=True)
    for key,value in [('scale',.5),('multiple',3.),('days',14),('negative_z',1.)]:
        for factor in [.8,1.2]:
            changed=round(value*factor) if key=='days' else value*factor
            f=frames(d,**{key:changed})['full']
            eq,tr,_=simulate(evaluation,f,'A',no_time_stop=True,chandelier=True,chandelier_multiple=changed if key=='multiple' else 3.)
            report['sensitivity'][f'{key}_{factor}']=metrics(eq,tr)
    for factor in [.8,1.2]:
        f=frames(d,scale=.5*factor,multiple=3*factor,days=round(14*factor),negative_z=factor)['full']
        eq,tr,_=simulate(evaluation,f,'A',no_time_stop=True,chandelier=True,chandelier_multiple=3*factor)
        report['sensitivity'][f'joint_{factor}']=metrics(eq,tr)
    (ROOT/'research/a_refinement_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
