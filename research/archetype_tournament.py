"""Frozen-spec chronological IS tournament. Reserved 30% is never loaded."""
from pathlib import Path
from dataclasses import dataclass,asdict,replace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import io
import numpy as np
import pandas as pd
from research.impulse_scan import discovery,impulses
from research.mine_anomalies import engineer
ROOT=Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Spec:
    risk:float=.005
    stop_vol:float=2.
    trail:float=2.
    deviation:float=1.
    impulse:float=1.
    tail:float=.1
    holding:float=1.
    liquidity:float=.001


def signals(d,p=Spec()):
    features,regime,ret=engineer(d)
    s=d.spot_close
    tr=pd.concat([d.spot_high-d.spot_low,(d.spot_high-s.shift()).abs(),(d.spot_low-s.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(24).mean()
    variance=(.5*np.log(d.spot_high/d.spot_low)**2-(2*np.log(2)-1)*np.log(s/d.spot_open)**2).clip(lower=0)
    daily_sigma=np.sqrt(variance.rolling(24).mean()*24)
    vol=np.sqrt(ret.pow(2).rolling(168).mean())
    vol_state=pd.Series(np.where(vol>vol.rolling(720).median().shift(1),'high','low'),index=d.index).where(vol.notna())
    core=pd.DataFrame({'sigma':daily_sigma,'regime':regime,'vol_regime':vol_state},index=d.index)
    a=core.copy()
    a['signal']=regime.eq('bull')&features.oi_weighted_funding.lt(0)&features.position_account_divergence.gt(0)
    a['stop']=s*np.exp(-p.stop_vol*daily_sigma);a['limit']=np.nan;a['target']=np.nan
    damping=pd.Series(False,index=d.index)
    for col in ['compression_x_flow_acceleration','signed_flow_low_impact']:
        x=features[col];lo=x.rolling(2160,min_periods=720).quantile(p.tail).shift(1)
        hi=x.rolling(2160,min_periods=720).quantile(1-p.tail).shift(1)
        damping |= x.le(lo)|x.ge(hi)
    center=(s*d.spot_volume).rolling(4).sum()/d.spot_volume.rolling(4).sum()
    b=core.copy();b['limit']=center-p.deviation*atr
    b['signal']=damping&s.lt(b['limit'])
    # Rest below current price, rather than assume a marketable quote is maker.
    b['limit']=np.minimum(b['limit'],s*(1-.0005))
    b['stop']=b['limit']-p.stop_vol*atr;b['target']=center
    ev=impulses(d,p.impulse);c=core.copy()
    c['signal']=ev.cascade_1h|ev.cascade_3h
    c['stop']=ev.low_3h.where(ev.cascade_3h,ev.low_1h)
    c['signal'] &= d.spot_low.gt(c['stop'])
    c['limit']=np.nan;c['target']=np.nan
    for frame in [a,b,c]:
        frame['signal'] &= frame.sigma.notna()&frame.stop.gt(0)&frame.regime.notna()
        frame.index-=pd.Timedelta(minutes=15)
    return {'A':a,'B':b,'C':c}


def simulate(raw,f,kind,p=Spec(),cost_factor=1.,maker_fee=.001,penetration=.0005,
             no_time_stop=False,chandelier=False,chandelier_multiple=3.):
    """Cash spot, next-bar decisions, stop-first ambiguity, no same-bar maker TP.

    Limit fills are conditional OHLC scenarios, not reconstructed queue positions.
    Primary quotes use conservative fees; a separate hypothetical discount is optional.
    """
    context=f.reindex(raw.index,method='ffill',tolerance=pd.Timedelta(minutes=45))
    events=f.signal.reindex(raw.index,fill_value=False).to_numpy(bool)
    exit_events=f.get('exit_signal',pd.Series(False,index=f.index)).reindex(raw.index,fill_value=False).to_numpy(bool)
    cols={col:context[col].to_numpy() for col in context}
    o=raw.open.to_numpy();low=raw.low.to_numpy();high=raw.high.to_numpy();close=raw.close.to_numpy()
    volume=raw.volume.to_numpy();times=raw.index;cash=10000.;qty=0.;pending=None;ledger=[];equity=[]
    fee=.001*cost_factor;slip=.0005*cost_factor
    hold={'A':168,'B':4,'C':12}[kind]*p.holding
    if no_time_stop:hold=float('inf')
    stop=0.;target=np.nan;peak=0.;entry_cost=0.;entry_fee=0.;entry_i=0;entry_ref=0.;ctx={};order_count=0;filled=0
    def exit_position(i,reference,reason):
        nonlocal cash,qty
        price=reference*(1-slip);exit_fee=qty*price*fee
        proceeds=qty*price-exit_fee
        ledger.append({**ctx,'entry_time':str(times[entry_i]),'exit_time':str(times[i]),'net_pnl':proceeds-entry_cost,
                       'reason':reason,'hours':(times[i]-times[entry_i]).total_seconds()/3600,
                       'entry_notional':qty*entry_ref,'fees':entry_fee+exit_fee,'entry_price':entry_ref,'exit_price':price})
        cash+=proceeds;qty=0.
    for i in range(len(raw)):
        gap=i>0 and times[i]-times[i-1]!=pd.Timedelta(minutes=15)
        exited=False;new_fill=False
        if gap:
            pending=None
            if qty:exit_position(i,o[i],'data_gap');exited=True
        if qty:
            if o[i]<=stop:exit_position(i,o[i],'gap_stop');exited=True
            elif i and exit_events[i-1]:exit_position(i,o[i],'regime_exit');exited=True
            elif (times[i]-times[entry_i]).total_seconds()/3600>=hold:
                exit_position(i,o[i],'time_exit');exited=True
        if not qty and not pending and not exited and not gap and i and events[i-1]:
            candidate=cols['stop'][i-1];limit=cols['limit'][i-1]
            if np.isfinite(candidate) and candidate<o[i]:
                ctx={'regime':str(cols['regime'][i-1]),'vol_regime':str(cols['vol_regime'][i-1])}
                if kind=='B':
                    if np.isfinite(limit) and candidate<limit<o[i]:
                        pending=(i,limit,candidate,cols['target'][i-1],ctx.copy());order_count+=1
                else:
                    pending=(i,o[i]*(1+slip),candidate,np.nan,ctx.copy());order_count+=1
        if pending and not qty:
            born,price,candidate,tgt,info=pending
            if i-born>=4 or gap:pending=None
            elif kind!='B' or low[i]<price*(1-penetration):
                # Quote must trade through, not merely touch. No favorable gap price improvement.
                rate=(maker_fee*cost_factor+slip) if kind=='B' else fee
                risk_per=price*(1+rate)-candidate*(1-slip)*(1-fee)
                cap=p.liquidity*(.1 if kind=='B' else 1)*volume[i-1]*close[i-1]
                risk_scale=cols['risk_scale'][i-1] if 'risk_scale' in cols else 1.
                qty=min(cash*p.risk*risk_scale/risk_per,cash/(price*(1+rate)),cap/price) if risk_per>0 and np.isfinite(risk_scale) and 0<risk_scale<=1 else 0
                if qty>0:
                    entry_fee=qty*price*rate;entry_cost=qty*price+entry_fee;cash-=entry_cost
                    entry_ref=price;entry_i=i;stop=candidate;target=tgt;peak=price;ctx=info;filled+=1;new_fill=True
                pending=None
        if qty:
            if low[i]<=stop:exit_position(i,min(o[i],stop),'intrabar_stop')
            elif kind=='B' and not new_fill and high[i]>=target:exit_position(i,target,'center_exit')
        if qty and kind=='A':
            peak=max(peak,high[i] if chandelier else close[i]);sigma=cols['sigma'][i]
            if chandelier:
                atr=cols['daily_atr'][i]
                if np.isfinite(atr):stop=max(stop,peak-chandelier_multiple*atr)
            elif np.isfinite(sigma):stop=max(stop,peak*np.exp(-p.trail*sigma))
        equity.append(cash+qty*close[i])
    if qty:exit_position(len(raw)-1,close[-1],'fold_end');equity[-1]=cash
    return pd.Series(equity,index=times),pd.DataFrame(ledger),{'orders':order_count,'fills':filled}


def metrics(eq,tr):
    daily=eq.resample('D').last().ffill().pct_change();daily.iloc[0]=eq.resample('D').last().iloc[0]/10000-1
    pnl=tr.net_pnl if len(tr) else pd.Series(dtype=float)
    downside=np.sqrt(np.mean(np.minimum(daily,0)**2));loss=-pnl.clip(upper=0).sum();gain=pnl.clip(lower=0).sum()
    return {'return_pct':float((eq.iloc[-1]/10000-1)*100),'sharpe':float(daily.mean()/daily.std()*np.sqrt(365)) if daily.std()>0 else None,
        'sortino':float(daily.mean()/downside*np.sqrt(365)) if downside>0 else None,
        'max_dd_pct':float((eq/eq.cummax().clip(lower=10000)-1).min()*100),
        'profit_factor':float(gain/loss) if loss else None,'win_rate':float((pnl>0).mean()) if len(pnl) else None,
        'avg_trade_pnl':float(pnl.mean()) if len(pnl) else None,'trades':len(pnl),
        'net_without_best_5pct':float(pnl.sum()-pnl.nlargest(max(1,int(np.ceil(len(pnl)*.05)))).sum())}


def walk(raw,f,kind,p=Spec(),**kwargs):
    first=raw.index.min()+pd.Timedelta(days=365)
    ends=pd.date_range(first,raw.index.max()+pd.Timedelta(minutes=15),freq='90D').tolist()
    if ends[-1]<=raw.index.max():ends.append(raw.index.max()+pd.Timedelta(minutes=15))
    equity=[];trades=[];folds=[];capital=10000.;counts={'orders':0,'fills':0}
    for number,(start,end) in enumerate(zip(ends[:-1],ends[1:]),1):
        part=raw.loc[(raw.index>=start)&(raw.index<end)]
        eq,tr,c=simulate(part,f,kind,p,**kwargs)
        folds.append({'fold':number,'start':str(start),'end_exclusive':str(end),**metrics(eq,tr),**c})
        factor=capital/10000;eq*=factor
        if len(tr):
            tr['net_pnl']*=factor;tr['entry_notional']*=factor;tr['fees']*=factor;tr['fold']=number;trades.append(tr)
        equity.append(eq);capital=eq.iloc[-1]
        for key in counts:counts[key]+=c[key]
    eq=pd.concat(equity);tr=pd.concat(trades,ignore_index=True) if trades else pd.DataFrame()
    report={**metrics(eq,tr),**counts,'folds':folds}
    dd=eq/eq.cummax().clip(lower=10000)-1;trough=dd.idxmin();peak=eq.loc[:trough].idxmax()
    report['worst_drawdown']={'peak':str(peak),'trough':str(trough),'drawdown_pct':float(dd.min()*100)}
    if len(tr):
        report['regime_pnl']=tr.groupby(['regime','vol_regime']).net_pnl.agg(['count','sum','mean']).reset_index().to_dict('records')
        mask=(pd.to_datetime(tr.exit_time,utc=True)>=peak)&(pd.to_datetime(tr.entry_time,utc=True)<=trough)
        report['drawdown_trades']=tr.loc[mask].to_dict('records')
        report['exit_diagnostics']=tr.groupby('reason').net_pnl.agg(['count','sum']).to_dict('index')
    return report,eq,tr


def main():
    d=discovery();boundary=d.index.max()
    # Stop on timestamp before parsing any reserved-period OHLCV values.
    lines=[]
    with (ROOT/'data/research_15m/BTCUSDT.csv').open() as source:
        lines.append(next(source))
        for line in source:
            stamp=pd.Timestamp(line.split(',',1)[0])
            if stamp+pd.Timedelta(minutes=15)>boundary:break
            if stamp>=d.index.min()-pd.Timedelta(hours=1):lines.append(line)
    raw=pd.read_csv(io.StringIO(''.join(lines)),index_col=0)
    raw.index=pd.to_datetime(raw.index,utc=True);raw=raw.loc[raw.volume>0]
    out=ROOT/'output/tournament_is';out.mkdir(parents=True,exist_ok=True)
    p=Spec();base=signals(d,p);report={'spec':asdict(p),'discovery_end':str(boundary),'method':'Frozen parameters; initial 365d calibration/history, successive 90d forward evaluation windows inside previously explored IS. Cash spot only; no funding. No final OOS validation.','archetypes':{},'sensitivity':{}}
    for kind in ['A','B','C']:
        r,eq,tr=walk(raw,base[kind],kind,p);report['archetypes'][kind]=r
        eq.to_csv(out/f'{kind}_equity.csv');tr.to_csv(out/f'{kind}_trades.csv',index=False)
        print(kind,json.dumps({k:r[k] for k in ['return_pct','sharpe','sortino','max_dd_pct','profit_factor','win_rate','avg_trade_pnl','trades']}),flush=True)
    variants=[]
    for field in asdict(p):
        for scale in [.8,1.2]:variants.append((f'{field}_{scale}',replace(p,**{field:getattr(p,field)*scale})))
    for scale in [.8,1.2]:variants.append((f'all_{scale}',Spec(**{k:v*scale for k,v in asdict(p).items()})))
    for label,param in variants:
        frames=signals(d,param);report['sensitivity'][label]={}
        affected={'risk':'ABC','stop_vol':'AB','trail':'A','deviation':'B','impulse':'C','tail':'B','holding':'ABC','liquidity':'ABC'}
        kinds=affected.get(label.rsplit('_',1)[0],'ABC')
        for kind in kinds:
            r,_,_=walk(raw,frames[kind],kind,param)
            report['sensitivity'][label][kind]={k:r[k] for k in ['return_pct','sharpe','max_dd_pct','trades']}
        print('Sensitivity',label,flush=True)
    report['cost_stress']={}
    for kind in ['A','B','C']:
        r,_,_=walk(raw,base[kind],kind,p,cost_factor=2.)
        report['cost_stress'][kind]={k:r[k] for k in ['return_pct','sharpe','trades']}
    report['B_execution_scenarios']={}
    for label,kwargs in [('hypothetical_6bp_maker',{'maker_fee':.0006}),('10bp_penetration',{'penetration':.001})]:
        r,_,_=walk(raw,base['B'],'B',p,**kwargs)
        report['B_execution_scenarios'][label]={k:r[k] for k in ['return_pct','sharpe','trades','orders','fills']}
    (ROOT/'research/tournament_is_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
