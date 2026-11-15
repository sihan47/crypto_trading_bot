"""Research only: closed-bar structural hypotheses and friction-aware spot simulator.

No exchange, API, model, notifier, or production strategy imports.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Parameters:
    day: int=24
    week: int=168
    compression: float=.8
    shock: float=2.
    participation: float=2.
    trail: float=2.
    max_hours: int=168
    risk: float=.005
    liquidity_cap: float=.001


def validate(df):
    if not isinstance(df.index,pd.DatetimeIndex) or df.index.tz is None or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('Unique ordered timezone-aware bars required')
    x=df[['open','high','low','close','volume']]
    if not np.isfinite(x).all().all() or (x.iloc[:,:4]<=0).any().any() or (x.volume<0).any():
        raise ValueError('Invalid price/volume')
    if (x.high<x[['open','close','low']].max(axis=1)).any() or (x.low>x[['open','close','high']].min(axis=1)).any():
        raise ValueError('Invalid OHLC ranges')


def features(raw,p=Parameters()):
    validate(raw)
    g=raw.resample('1h')
    h=g.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    h.loc[g.close.count().ne(4),:]=np.nan
    v=(.5*np.log(h.high/h.low)**2-(2*np.log(2)-1)*np.log(h.close/h.open)**2).clip(lower=0)
    short=v.rolling(p.day).mean().shift(1)
    long=v.rolling(p.week).mean().shift(1)
    sigma=np.sqrt(short).replace(0,np.nan)
    dollar=h.close*h.volume
    f=pd.DataFrame(index=h.index)
    f['compression']=np.sqrt(short/long.replace(0,np.nan))
    f['shock']=np.log(h.close/h.close.shift())/sigma
    f['participation']=dollar/dollar.rolling(p.week).median().shift(1).replace(0,np.nan)
    f['sigma_day']=sigma*np.sqrt(p.day)
    f['initial_stop']=h.low.rolling(p.day).min().shift(1)
    f['signal']=(f.compression<=p.compression)&(f.shock>=p.shock)&(f.participation>=p.participation)
    f['signal'] &= h.close.notna().rolling(p.week+1).sum().eq(p.week+1)
    # Diagnostics only; never part of entry/sizing logic.
    weekly_sigma=np.sqrt(long*24)
    past_median=weekly_sigma.rolling(24*28,min_periods=24*7).median().shift(1)
    f['vol_regime']=np.where(weekly_sigma>past_median,'high','low')
    f['prior_day_direction']=np.where(h.close.shift()>h.close.shift(p.day+1),'up','down')
    f['session']=np.select([f.index.hour<8,f.index.hour<16],['Asia_UTC_00_08','EU_UTC_08_16'],default='US_UTC_16_24')
    f['weekend']=f.index.dayofweek>=5
    # A 00:00 hourly opening stamp becomes available at 01:00, represented
    # at the last 15m opening stamp 00:45. Simulator acts at 01:00 open.
    f.index+=pd.Timedelta(minutes=45)
    return f


def simulate(raw,f,p=Parameters(),fee=.001,slip=.0005):
    validate(raw)
    aligned=f.reindex(raw.index,method='ffill',tolerance=pd.Timedelta(minutes=45))
    event=f['signal'].reindex(raw.index,fill_value=False).to_numpy(dtype=bool)
    o=raw.open.to_numpy();lo=raw.low.to_numpy();c=raw.close.to_numpy();vol=raw.volume.to_numpy()
    sig=aligned.sigma_day.to_numpy();initial=aligned.initial_stop.to_numpy()
    t=raw.index.asi8;step=pd.Timedelta(minutes=15).value
    cash=10000.;qty=0.;stop=0.;peak=0.;entry_cost=0.;entry_i=0
    equity=np.empty(len(raw));ledger=[];context={};fees=0.;slippage=0.;gap_exits=0

    def close_position(i,reference,reason):
        nonlocal cash,qty,fees,slippage
        price=reference*(1-slip);exit_fee=qty*price*fee
        proceeds=qty*price-exit_fee
        fees+=exit_fee;slippage+=qty*(reference-price)
        ledger.append({**context,'entry_time':str(raw.index[entry_i]),'exit_time':str(raw.index[i]),
                       'net_pnl':float(proceeds-entry_cost),'reason':reason,
                       'hours':float((t[i]-t[entry_i])/3.6e12),'qty':float(qty),
                       'entry_notional':float(entry_cost),'exit_price':float(price)})
        cash+=proceeds;qty=0.

    for i in range(len(raw)):
        gap=i>0 and t[i]-t[i-1]!=step
        if qty and gap:
            close_position(i,o[i],'data_gap');gap_exits+=1
        if qty:
            if o[i]<=stop: close_position(i,o[i],'gap_stop')
            elif (t[i]-t[entry_i])/3.6e12>=p.max_hours: close_position(i,o[i],'time_exit')
        # Exit at this open never immediately re-enters on the same old signal.
        just_exited=bool(ledger and ledger[-1]['exit_time']==str(raw.index[i]))
        if not qty and not gap and not just_exited and i and event[i-1]:
            price=o[i]*(1+slip);candidate=initial[i-1]
            if np.isfinite(candidate) and 0<candidate<o[i] and np.isfinite(sig[i-1]):
                loss_per_unit=price*(1+fee)-candidate*(1-slip)*(1-fee)
                liquidity=p.liquidity_cap*c[i-1]*vol[i-1]
                qty=min(cash*p.risk/loss_per_unit,cash/(price*(1+fee)),liquidity/price)
                if qty>0:
                    entry_cost=qty*price*(1+fee);cash-=entry_cost
                    fees+=qty*price*fee;slippage+=qty*(price-o[i])
                    entry_i=i;stop=candidate;peak=price
                    prior=aligned.iloc[i-1]
                    context={k:(bool(prior[k]) if k=='weekend' else str(prior[k])) for k in ['vol_regime','prior_day_direction','session','weekend']}
        if qty and lo[i]<=stop:
            close_position(i,min(o[i],stop),'intrabar_stop')
        if qty:
            peak=max(peak,c[i])
            if np.isfinite(sig[i]):
                stop=max(stop,peak*np.exp(-p.trail*sig[i]))
        equity[i]=cash+qty*c[i]
    if qty:
        close_position(len(raw)-1,c[-1],'end_of_sample');equity[-1]=cash
    return pd.Series(equity,index=raw.index),pd.DataFrame(ledger),{'fees':fees,'slippage_estimate':slippage,'funding':0.,'gap_exits':gap_exits}


def metrics(eq,trades,costs):
    daily=eq.resample('1D').last().ffill().pct_change(fill_method=None)
    daily.iloc[0]=eq.resample('1D').last().iloc[0]/10000-1
    pnl=trades.net_pnl if len(trades) else pd.Series(dtype=float)
    gain=float(pnl.clip(lower=0).sum());loss=float(-pnl.clip(upper=0).sum())
    k=max(1,math.ceil(len(pnl)*.05))
    without=float(pnl.sum()-pnl.nlargest(k).sum())
    return {'return_pct':float((eq.iloc[-1]/10000-1)*100),
            'max_drawdown_pct':float((eq/eq.cummax().clip(lower=10000)-1).min()*100),
            'sharpe':float(daily.mean()/daily.std()*np.sqrt(365)) if daily.std()>0 else 0.,
            'profit_factor':gain/loss if loss else None,'trades':len(trades),
            'net_pnl_excluding_top_5pct':without,'top_5pct_share_of_gross_profit':float(pnl.nlargest(k).clip(lower=0).sum()/gain) if gain else None,
            'win_rate':float((pnl>0).mean()) if len(pnl) else 0.,**costs}


def sensitivity(p):
    yield 'base',p
    for field,value in asdict(p).items():
        for factor in [.8,1.2]:
            v=max(1,int(round(value*factor))) if isinstance(value,int) else value*factor
            yield f'{field}_{factor}',replace(p,**{field:v})
    for factor in [.8,1.2]:
        values={k:max(1,int(round(v*factor))) if isinstance(v,int) else v*factor for k,v in asdict(p).items()}
        yield f'all_{factor}',type(p)(**values)


def run_research(label='compression',feature_fn=features,base=Parameters()):
    path=ROOT/'data/research_15m/BTCUSDT.csv'
    raw=pd.read_csv(path,index_col=0,parse_dates=True);raw.index=pd.to_datetime(raw.index,utc=True)
    validate(raw)
    zero_volume=int(raw.volume.eq(0).sum())
    # A zero-trade candle is not executable liquidity. Treat it as a gap,
    # including when aggregating complete hourly bars.
    raw=raw.loc[raw.volume>0].copy()
    out=ROOT/'output'/f'structural_{label}';out.mkdir(parents=True,exist_ok=True)
    report={'hypothesis':label,'parameters':asdict(base),'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'zero_volume_bars_excluded':zero_volume,
            'method':'Spot only, next-15m-open fills, 10bp fee +5bp slippage EACH side, no funding. Exposed-history temporal holdout; not pristine OOS.',
            'periods':{},'sensitivity':{}}
    base_f=feature_fn(raw,base)
    periods={'full':('2023-01-01',None),'development':('2023-01-01','2025-01-01'),
             'holdout':('2025-01-01',None),'2023':('2023-01-01','2024-01-01'),
             '2024':('2024-01-01','2025-01-01'),'2025':('2025-01-01','2026-01-01'),'2026':('2026-01-01',None)}
    for name,(start,end) in periods.items():
        d=raw.loc[(raw.index>=start)&((raw.index<end) if end else True)]
        eq,tr,cost=simulate(d,base_f,base)
        report['periods'][name]=metrics(eq,tr,cost)
        if name in {'full','holdout'}:
            eq.to_csv(out/f'{name}_equity.csv');tr.to_csv(out/f'{name}_trades.csv',index=False)
            groups={}
            for col in ['vol_regime','prior_day_direction','session','weekend','reason']:
                groups[col]=tr.groupby(col).net_pnl.agg(['count','sum','mean']).to_dict('index') if len(tr) else {}
            report['periods'][name]['loss_diagnostics']=groups
    hold=raw.loc[raw.index>='2025-01-01']
    for name,p in sensitivity(base):
        f=feature_fn(raw,p)
        eq,tr,cost=simulate(hold,f,p)
        report['sensitivity'][name]=metrics(eq,tr,cost)
    eq,tr,cost=simulate(hold,base_f,base,fee=.002,slip=.001)
    report['double_cost_holdout']=metrics(eq,tr,cost)
    h=report['periods']['holdout']
    report['tests']={'sharpe_at_least_1_5':h['sharpe']>=1.5,'pf_at_least_1_3':h['profit_factor'] is not None and h['profit_factor']>=1.3,
                     'drawdown_at_most_15pct':h['max_drawdown_pct']>=-15,'at_least_30_trades':h['trades']>=30,
                     'positive_without_top_5pct':h['net_pnl_excluding_top_5pct']>0,
                     'all_sensitivity_positive':all(s['return_pct']>0 for s in report['sensitivity'].values()),
                     'double_cost_positive':report['double_cost_holdout']['return_pct']>0}
    report['passed']=all(report['tests'].values())
    target=ROOT/'research'/f'structural_{label}_evaluation.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'holdout':h,'tests':report['tests'],'passed':report['passed']},indent=2),flush=True)
    return report


if __name__=='__main__': run_research()
