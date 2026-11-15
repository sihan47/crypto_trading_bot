"""Research-only BTC spot/linear-perpetual carry with separately timed legs."""
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import math
import numpy as np
import pandas as pd
from research.structural_btc import metrics
ROOT=Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CarryParameters:
    lookback_days: int=7
    holding_days: int=30
    hurdle: float=1.25
    allocation: float=.4
    loss_budget: float=.01
    margin_buffer: float=.25
    liquidity_cap: float=.001


def load_data():
    folder=ROOT/'data/research_structural'
    def read(path):
        d=pd.read_csv(path,index_col=0);d.index=pd.to_datetime(d.index,utc=True,format='mixed')
        if d.index.has_duplicates or not d.index.is_monotonic_increasing: raise ValueError('Invalid source index')
        return d
    spot=read(ROOT/'data/research_rotation/BTCUSDT.csv')
    perp=read(folder/'klines.csv');mark=read(folder/'markPriceKlines.csv');fund=read(folder/'fundingRate.csv')
    if fund.index.to_series().diff().max()>pd.Timedelta(hours=8,minutes=1):
        raise ValueError('Funding history gap; do not impute zero funding')
    floored=fund.index.floor('1h')
    if ((fund.index-floored)>pd.Timedelta(seconds=60)).any(): raise ValueError('Funding not near hourly mark open')
    fallback=pd.to_numeric(fund.markPrice,errors='coerce').isna() | pd.to_numeric(fund.markPrice,errors='coerce').le(0)
    prices=pd.to_numeric(fund.markPrice,errors='coerce').to_numpy()
    prices[fallback]=mark.open.reindex(floored[fallback]).to_numpy()
    if not np.isfinite(prices).all(): raise ValueError('Unavailable funding mark price')
    rate=pd.to_numeric(fund.fundingRate)
    pay=pd.Series(rate.to_numpy()*prices,index=floored).groupby(level=0).sum()
    rates=pd.Series(rate.to_numpy(),index=floored).groupby(level=0).sum()
    d=spot[['open','high','low','close','volume']].add_prefix('s_').join(perp[['open','high','low','close','volume']].add_prefix('p_'),how='inner').join(mark[['open','high','low','close']].add_prefix('m_'),how='inner')
    no_trade=d.s_volume.eq(0)|d.p_volume.eq(0)
    halted_hours=int(no_trade.sum())
    d=d.loc[~no_trade].copy()
    for prefix in ['s_','p_','m_']:
        if (d[prefix+'high']<d[[prefix+'open',prefix+'close',prefix+'low']].max(axis=1)).any(): raise ValueError('Invalid OHLC')
    if not np.isfinite(d).all().all() or (d<=0).any().any(): raise ValueError('Invalid prices')
    # Funding during any missing spot hour still belongs to an existing short.
    allhours=pd.date_range(d.index[0],d.index[-1],freq='1h')
    cumulative=pay.reindex(allhours,fill_value=0).cumsum()
    d['pay']=pay.reindex(d.index,fill_value=0)
    d['gap_pay']=cumulative.reindex(d.index).diff().fillna(0)-d.pay
    d['rate']=rates.reindex(d.index,fill_value=0)
    audit={'rows':len(d),'zero_volume_hours_excluded':halted_hours,'funding_events':len(fund),'funding_mark_open_fallbacks':int(fallback.sum()),
           'max_funding_timestamp_offset_seconds':float((fund.index-floored).total_seconds().max()),
           'market_gap_count':int(d.index.to_series().diff().gt(pd.Timedelta(hours=1)).sum()),
           'funding_price_note':'Missing settlement mark uses nominal settlement-hour mark OPEN; timestamp jitter explicitly audited.'}
    return d,audit


def carry_features(d,p=CarryParameters(),cost_scale=1.):
    hourly=d.asfreq('1h')
    f=pd.DataFrame(index=hourly.index)
    # Missing market bars invalidate the forecast window, not zero-fill it.
    f['daily_funding']=hourly.rate.rolling(p.lookback_days*24).sum().shift(1)/p.lookback_days
    f['basis']=hourly.p_close/hourly.s_close-1
    cost=2*(.001+.0006+2*.0005)*cost_scale
    f['entry']=(f.daily_funding>0)&(f.basis>=0)&(f.daily_funding*p.holding_days+f.basis>p.hurdle*cost)
    return f.reindex(d.index)


def carry_simulate(d,f,p=CarryParameters(),cost_scale=1.,leg_delay=0):
    sf=.001*cost_scale;pf=.0006*cost_scale;slip=.0005*cost_scale
    cash=10000.;sq=0.;pq=0.;q=0.;entry_equity=0.;entry_i=0;perp_entry=0.;margin=0.;fund_since=0.
    short_due=None;cover_due=None;closing=False;last_exit=-1
    fees=0.;slippage=0.;fund_total=0.;margin_flags=0;gap_exits=0;trades=[];equity=[];reason=''
    so=d.s_open.to_numpy();po=d.p_open.to_numpy();sc=d.s_close.to_numpy();mc=d.m_close.to_numpy();mh=d.m_high.to_numpy()
    entry=f.entry.to_numpy();daily=f.daily_funding.to_numpy();times=d.index.asi8;hour=3.6e12

    def sell_spot(i,reference):
        nonlocal cash,sq,fees,slippage
        price=reference*(1-slip);fee=sq*price*sf
        cash+=sq*price-fee;fees+=fee;slippage+=sq*(reference-price);sq=0.

    def cover_short(i,reference):
        nonlocal cash,pq,fees,slippage,last_exit,cover_due,closing
        price=reference*(1+slip);fee=pq*price*pf
        cash+=pq*(perp_entry-price)-fee;fees+=fee;slippage+=pq*(price-reference);pq=0.
        trades.append({'entry_time':str(d.index[entry_i]),'exit_time':str(d.index[i]),'net_pnl':float(cash-entry_equity),
                       'funding':float(fund_since),'reason':reason,'hours':float((times[i]-times[entry_i])/hour),'qty':float(q)})
        last_exit=i;cover_due=None;closing=False

    for i in range(len(d)):
        gap=i>0 and times[i]-times[i-1]!=hour
        held_before=pq>0
        should_exit=False
        if sq and pq and not closing and i:
            previous=equity[-1]
            margin_ratio=(margin+fund_since+pq*(perp_entry-mc[i-1]))/(pq*mc[i-1])
            if gap: reason='data_gap';should_exit=True;gap_exits+=1
            elif daily[i-1]<0: reason='negative_funding';should_exit=True
            elif previous<=entry_equity*(1-p.loss_budget): reason='pair_loss';should_exit=True
            elif margin_ratio<p.margin_buffer: reason='margin_buffer';should_exit=True
            elif (times[i]-times[entry_i])/hour>=p.holding_days*24: reason='time_exit';should_exit=True
        if held_before:
            # Conservative boundary convention: never take positive funding on
            # an exit hour; negative funding is still charged.
            boundary=should_exit or (cover_due is not None and i>=cover_due)
            amount=pq*(float(d.gap_pay.iloc[i])+(min(0.,float(d.pay.iloc[i])) if boundary else float(d.pay.iloc[i])))
            cash+=amount;fund_since+=amount;fund_total+=amount
        if should_exit:
            sell_spot(i,so[i]);closing=True;cover_due=i+leg_delay
        if cover_due is not None and i>=cover_due:
            cover_short(i,po[i])
        if not sq and not pq and short_due is None and not closing and i and not gap and last_exit!=i and entry[i-1]:
            entry_equity=cash;entry_i=i;fund_since=0.;reason=''
            price=so[i]*(1+slip)
            liquidity=p.liquidity_cap*min(d.s_close.iloc[i-1]*d.s_volume.iloc[i-1],d.p_close.iloc[i-1]*d.p_volume.iloc[i-1])
            q=min(cash*p.allocation/max(price,po[i]),liquidity/max(price,po[i]))
            if q>0:
                fee=q*price*sf;cash-=q*price+fee;fees+=fee;slippage+=q*(price-so[i]);sq=q;short_due=i+leg_delay
        if short_due is not None and i>=short_due:
            perp_entry=po[i]*(1-slip);pq=q
            fee=pq*perp_entry*pf;cash-=fee;fees+=fee;slippage+=pq*(po[i]-perp_entry)
            margin=pq*perp_entry-fee;short_due=None
            if cash<margin: margin_flags+=1
            amount=pq*min(0.,float(d.pay.iloc[i]))
            cash+=amount;fund_since+=amount;fund_total+=amount
        if pq and margin+fund_since+pq*(perp_entry-mh[i])<.01*pq*mh[i]:
            # OHLC can prove potential breach, not reconstruct liquidation fills.
            # Exit at adverse per-leg extrema, flag and disqualify the run.
            margin_flags+=1;reason='possible_margin_breach'
            if sq: sell_spot(i,float(d.s_low.iloc[i]))
            cover_short(i,float(d.p_high.iloc[i]))
        equity.append(cash+sq*sc[i]+pq*(perp_entry-mc[i]))
    if sq or pq:
        i=len(d)-1;reason='end_of_sample'
        if sq: sell_spot(i,sc[i])
        if pq: cover_short(i,float(d.p_close.iloc[i]))
        else:
            trades.append({'entry_time':str(d.index[entry_i]),'exit_time':str(d.index[i]),'net_pnl':cash-entry_equity,
                           'funding':fund_since,'reason':'end_before_short_fill','hours':0.,'qty':q})
        equity[-1]=cash
    return pd.Series(equity,index=d.index),pd.DataFrame(trades),{'fees':fees,'slippage_estimate':slippage,'funding':fund_total,'gap_exits':gap_exits,'margin_breach_flags':margin_flags}


def variants(p):
    yield 'base',p
    for key,v in asdict(p).items():
        for a in [.8,1.2]:
            yield f'{key}_{a}',replace(p,**{key:max(1,round(v*a)) if isinstance(v,int) else v*a})
    for a in [.8,1.2]:
        yield f'all_{a}',CarryParameters(**{k:max(1,round(v*a)) if isinstance(v,int) else v*a for k,v in asdict(p).items()})


def main():
    d,audit=load_data();p=CarryParameters();f=carry_features(d,p)
    out=ROOT/'output/structural_carry';out.mkdir(parents=True,exist_ok=True)
    r={'method':'Spot long / equal-BTC USDT perpetual short; 40% initial capital per notional; explicit fees/funding; hourly marks understate possible intrahour pair DD. Exposed-history holdout.',
       'parameters':asdict(p),'audit':audit,'periods':{},'sensitivity':{}}
    for name,start,end in [('full','2023-01-01',None),('development','2023-01-01','2025-01-01'),('holdout','2025-01-01',None)]+[(str(y),f'{y}-01-01',f'{y+1}-01-01') for y in range(2023,2027)]:
        part=d.loc[(d.index>=start)&((d.index<end) if end else True)]
        eq,tr,cost=carry_simulate(part,f.loc[part.index],p)
        r['periods'][name]=metrics(eq,tr,cost)
        if name in {'full','holdout'}:
            eq.to_csv(out/f'{name}_equity.csv');tr.to_csv(out/f'{name}_trades.csv',index=False)
            r['periods'][name]['exit_diagnostics']=tr.groupby('reason').net_pnl.agg(['count','sum']).to_dict('index') if len(tr) else {}
    hold=d.loc[d.index>='2025-01-01']
    for name,v in variants(p):
        ff=carry_features(d,v).loc[hold.index]
        eq,tr,cost=carry_simulate(hold,ff,v)
        r['sensitivity'][name]=metrics(eq,tr,cost)
    for name,scale,delay in [('double_cost',2.,0),('double_cost_cost_aware',2.,0),('one_hour_leg_delay',1.,1)]:
        # Frozen entry evidence isolates execution-cost damage. Separately
        # report what happens if the entry hurdle is updated for higher costs.
        feature_scale=scale if name=='double_cost_cost_aware' else 1.
        ff=carry_features(d,p,cost_scale=feature_scale).loc[hold.index]
        eq,tr,cost=carry_simulate(hold,ff,p,cost_scale=scale,leg_delay=delay)
        r[name]=metrics(eq,tr,cost)
    eq=pd.read_csv(out/'holdout_equity.csv',index_col=0,parse_dates=True).iloc[:,0]
    days=eq.resample('1D').last().pct_change(fill_method=None).dropna()-(1.04**(1/365)-1)
    r['cash_hurdle_4pct_scenario_sharpe']=float(days.mean()/days.std()*np.sqrt(365)) if days.std()>0 else 0.
    h=r['periods']['holdout']
    r['tests']={'sharpe_at_least_1_5':h['sharpe']>=1.5,'pf_at_least_1_3':h['profit_factor'] is not None and h['profit_factor']>=1.3,
                'drawdown_at_most_15pct':h['max_drawdown_pct']>=-15,'at_least_30_trades':h['trades']>=30,
                'positive_without_top_5pct':h['net_pnl_excluding_top_5pct']>0,
                'all_sensitivity_positive':all(x['return_pct']>0 for x in r['sensitivity'].values()),
                'double_cost_positive':r['double_cost']['return_pct']>0,
                'leg_delay_positive':r['one_hour_leg_delay']['return_pct']>0,
                'no_margin_flags':all(x['margin_breach_flags']==0 for x in [*r['periods'].values(),*r['sensitivity'].values(),r['double_cost'],r['one_hour_leg_delay']])}
    r['passed']=all(r['tests'].values())
    (ROOT/'research/structural_carry_evaluation.json').write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'holdout':h,'tests':r['tests'],'cash_hurdle_sharpe':r['cash_hurdle_4pct_scenario_sharpe']},indent=2),flush=True)


if __name__=='__main__': main()
