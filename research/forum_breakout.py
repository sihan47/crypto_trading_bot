"""Long-only forum adaptation with 15m intrabar execution resolution."""
import numpy as np
import pandas as pd


def hourly_features(raw, vwap_mode='session'):
    g=raw.resample('1h',label='left',closed='left')
    h=g.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    h=h.loc[g['close'].count().eq(4)]
    c=h.close
    avg=c.rolling(42).mean()
    upper=avg+2.5*c.rolling(42).std(ddof=0)
    typical=(h.high+h.low+c)/3
    if vwap_mode=='session':
        group=h.index.normalize()
        vwap=(typical*h.volume).groupby(group).cumsum()/h.volume.groupby(group).cumsum().replace(0,np.nan)
    elif vwap_mode=='rolling24':
        vwap=(typical*h.volume).rolling(24).sum()/h.volume.rolling(24).sum().replace(0,np.nan)
    else:
        raise ValueError('Unknown VWAP anchor')
    delta=c.diff()
    gain=delta.clip(lower=0).ewm(alpha=.2,adjust=False,min_periods=5).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=.2,adjust=False,min_periods=5).mean()
    rsi=100*gain/(gain+loss).replace(0,np.nan)
    up=h.high.diff(); down=-h.low.diff()
    plus=up.where((up>down)&(up>0),0.).ewm(alpha=.2,adjust=False,min_periods=5).mean()
    minus=down.where((down>up)&(down>0),0.).ewm(alpha=.2,adjust=False,min_periods=5).mean()
    dx=100*(plus-minus).abs()/(plus+minus).replace(0,np.nan)
    adx=dx.ewm(alpha=.2,adjust=False,min_periods=5).mean()
    bb=c>=upper
    momentum=(c>vwap).rolling(6).sum().eq(6)&(rsi>55)&(adx>45)
    inside=(c.shift()>=upper.shift())&(c<upper)
    f=pd.DataFrame({'bb':bb,'combined':bb|momentum,'exit':inside},index=h.index)
    # Information becomes available only after the full hour has closed.
    f.index=f.index+pd.Timedelta(hours=1)
    f['signal_available_at']=f.index
    return f


def simulate_forum(raw,features,variant='combined',start='2023-01-01',end='2026-11-14',fee=.001,slippage=.0005):
    df=raw.loc[(raw.index>=start)&(raw.index<end)]
    # Entry only at a new closed-hour decision, never repeated each 15 minutes.
    signals=features.reindex(df.index)
    buy=signals[variant].eq(True).to_numpy(dtype=bool)
    close_exit=signals['exit'].eq(True).to_numpy(dtype=bool)
    timestamps=df.index.asi8
    step=pd.Timedelta(minutes=15).value
    opens,highs,lows,closes=[df[col].to_numpy() for col in ['open','high','low','close']]
    cash=10000.;qty=0.;entry=0.;entry_cash=0.;opened=0
    equities=[];trades=[];ambiguous=0;reasons={};held=0
    for i in range(len(df)):
        exited=False
        consecutive=i>0 and timestamps[i]-timestamps[i-1]==step
        if qty:
            price=None;reason=None
            if opens[i]<=entry*.985:
                price=opens[i];reason='gap_stop'
            elif opens[i]>=entry*1.03:
                price=entry*1.03;reason='target'
            elif close_exit[i]:
                price=opens[i];reason='back_inside_band'
            elif timestamps[i]-opened>=pd.Timedelta(minutes=1075).value:
                price=opens[i];reason='time'
            if price is not None:
                cash=qty*price*(1-slippage)*(1-fee)
                trades.append(cash/entry_cash-1);qty=0.;exited=True
                reasons[reason]=reasons.get(reason,0)+1
        if not qty and not exited and buy[i] and consecutive:
            entry_cash=cash;entry=opens[i]*(1+slippage)
            qty=cash/(1+fee)/entry;cash=0.;opened=timestamps[i]
        if qty:
            held+=1
            stop=lows[i]<=entry*.985;target=highs[i]>=entry*1.03
            if stop or target:
                ambiguous+=bool(stop and target)
                price=entry*(.985 if stop else 1.03)
                reason='stop' if stop else 'target'
                cash=qty*price*(1-slippage)*(1-fee)
                trades.append(cash/entry_cash-1);qty=0.
                reasons[reason]=reasons.get(reason,0)+1
        equities.append(cash+qty*closes[i])
    if qty:
        equities[-1]=qty*closes[-1]*(1-slippage)*(1-fee)
        trades.append(equities[-1]/entry_cash-1)
        reasons['final_liquidation']=1
    curve=pd.Series(equities,index=df.index)
    daily=curve.resample('1D').last().pct_change().dropna()
    return {'return_pct':float((curve.iloc[-1]/10000-1)*100),
            'max_drawdown_pct':float((curve/curve.cummax().clip(lower=10000)-1).min()*100),
            'sharpe_daily':float(daily.mean()/daily.std()*np.sqrt(365)) if daily.std()>0 else 0.,
            'round_trips':len(trades),'win_rate_pct':float(np.mean(np.array(trades)>0)*100) if trades else 0.,
            'exposure_pct':held/len(df)*100,'ambiguous_bars_stop_first':ambiguous,'exit_reasons':reasons}
