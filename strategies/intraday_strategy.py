"""15m pullbacks and squeeze breakouts, gated by COMPLETED 4h bars.

Research signals are causal and share their implementation with runtime.
Stops are evaluated at bar CLOSE, not exchange-hosted intrabar stop orders.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntradayParams:
    enable_pullback: bool = True
    enable_squeeze: bool = True
    trend_fast: int = 20
    trend_slow: int = 50
    rsi_oversold: float = 10.0
    squeeze_window: int = 192
    squeeze_percentile: float = .10
    volume_multiple: float = 1.2
    fee_per_side: float = .001
    slippage_per_side: float = .0005
    min_reward_cost: float = 1.5
    max_atr_pct: float = .018
    stop_atr: float = 2.0
    trailing_atr: float = 3.0
    pullback_max_bars: int = 8
    squeeze_max_bars: int = 24
    cooldown_bars: int = 4

    def __post_init__(self):
        if not 1 < self.trend_fast < self.trend_slow or self.squeeze_window < 20:
            raise ValueError('Invalid indicator windows')
        if not 0 < self.squeeze_percentile < 1 or not 0 < self.rsi_oversold < 50:
            raise ValueError('Invalid signal thresholds')
        if min(self.pullback_max_bars,self.squeeze_max_bars) < 1 or self.cooldown_bars < 0:
            raise ValueError('Invalid holding/cooldown periods')
        if not all(np.isfinite(x) and x > 0 for x in (self.volume_multiple,self.min_reward_cost,
                   self.max_atr_pct,self.stop_atr,self.trailing_atr)):
            raise ValueError('Invalid risk parameters')
        if not all(np.isfinite(x) and 0 <= x < 1 for x in (self.fee_per_side,self.slippage_per_side)):
            raise ValueError('Invalid costs')


def completed_trend(df, params):
    groups = df.resample('4h', label='left', closed='left')
    bars = groups.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    # An unfinished 4h group, or one containing missing 15m bars, is never used.
    bars = bars.loc[groups['close'].count().eq(16)]
    fast = bars['close'].rolling(params.trend_fast).mean()
    slow = bars['close'].rolling(params.trend_slow).mean()
    higher = pd.DataFrame({'trend_fast_4h':fast,'trend_slow_4h':slow,'close_4h':bars['close']})
    # A 12:00-16:00 bar becomes available at the close of the 15:45 signal bar.
    higher.index = higher.index + pd.Timedelta(hours=3, minutes=45)
    higher['trend_available_at'] = higher.index + pd.Timedelta(minutes=15)
    return higher.reindex(df.index, method='ffill')


def intraday_features(df, params=IntradayParams()):
    if not isinstance(df.index,pd.DatetimeIndex) or df.index.tz is None or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('Unique ordered timezone-aware bars required')
    values = df[['open','high','low','close','volume']]
    if not np.isfinite(values.to_numpy()).all() or (values.iloc[:,:4] <= 0).any().any():
        raise ValueError('Invalid prices')
    if any(t.minute % 15 or t.second for t in df.index):
        raise ValueError('15-minute opening timestamps required')
    delta_time = df.index.to_series().diff()
    if (delta_time.dropna() < pd.Timedelta(minutes=15)).any():
        raise ValueError('Input is not 15m data')
    close = df['close']
    higher = completed_trend(df,params)
    bull = (higher['trend_fast_4h'] > higher['trend_slow_4h']) & (higher['close_4h'] > higher['trend_slow_4h'])
    tr = pd.concat([df['high']-df['low'],(df['high']-close.shift()).abs(),(df['low']-close.shift()).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    basis = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    upper = basis + 2*std
    bandwidth = 4*std/basis
    narrow = bandwidth <= bandwidth.rolling(params.squeeze_window).quantile(params.squeeze_percentile)
    recent_squeeze = narrow.shift(1).rolling(8).max().eq(1)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(2).mean()
    loss = (-delta.clip(upper=0)).rolling(2).mean()
    rsi = (100*gain/(gain+loss).replace(0,np.nan)).fillna(50)
    round_trip_cost = 2*(params.fee_per_side+params.slippage_per_side)
    reward_floor = round_trip_cost*params.min_reward_cost
    ready = higher['trend_slow_4h'].notna() & atr.notna() & bandwidth.rolling(params.squeeze_window).count().ge(params.squeeze_window)
    # Recent gaps suppress new entries; exits remain available.
    contiguous = delta_time.eq(pd.Timedelta(minutes=15)).rolling(20).sum().eq(20)
    allowed = ready & contiguous & bull & (atr/close <= params.max_atr_pct)
    pullback = allowed & (rsi.shift(1).rolling(3).min() < params.rsi_oversold) & (close > close.shift(1)) & (close < basis)
    pullback &= ((basis-close)/close >= reward_floor)
    squeeze = allowed & recent_squeeze & (close > upper.shift(1)) & (close > df['high'].rolling(20).max().shift(1))
    squeeze &= df['volume'] > df['volume'].rolling(20).mean().shift(1)*params.volume_multiple
    squeeze &= 2*atr/close >= reward_floor
    return pd.DataFrame({
        **{col:higher[col] for col in higher.columns},
        'close':close,'open':df['open'],'atr':atr,'atr_pct':atr/close,'basis':basis,
        'rsi2':rsi,'bandwidth':bandwidth,'bull_4h':bull,'ready':ready,
        'entry_blocked':~allowed,'pullback_setup':pullback & params.enable_pullback,
        'squeeze_setup':squeeze & params.enable_squeeze,
        'gap':delta_time.gt(pd.Timedelta(minutes=15)),
    },index=df.index)


def signals_from_features(features, params=IntradayParams()):
    """Bounded holding periods, no averaging down; entry at next bar open."""
    n=len(features)
    entries=np.zeros(n,dtype=bool)
    exits=np.zeros(n,dtype=bool)
    active=np.zeros(n,dtype=bool)
    mandatory=np.zeros(n,dtype=bool)
    setup_names=[]
    reasons=[]
    position=False
    setup='none'
    entry_index=-1
    entry_price=entry_atr=peak=0.
    cooldown_until=-1
    last_exit='flat'
    for i,row in enumerate(features.itertuples()):
        if position:
            if i == entry_index+1:
                entry_price=float(row.open)
                peak=entry_price
            peak=max(peak,float(row.close))
            max_bars=params.pullback_max_bars if setup=='pullback' else params.squeeze_max_bars
            reason=None
            if row.gap:
                reason='data_gap'
            elif row.close <= entry_price-params.stop_atr*entry_atr:
                reason='close_stop'
            elif not row.bull_4h:
                reason='higher_trend_invalidated'
            elif i-entry_index >= max_bars:
                reason='time_exit'
            elif setup=='pullback' and (row.close >= row.basis or row.rsi2 >= 70):
                reason='pullback_recovered'
            elif setup=='squeeze' and (row.close < row.basis or row.close <= peak-params.trailing_atr*entry_atr):
                reason='breakout_failed_or_trailing_exit'
            if reason:
                position=False
                last_exit=reason
                cooldown_until=i+params.cooldown_bars
        elif i > cooldown_until and row.ready:
            # Mutually exclusive execution paths, not two votes requiring agreement.
            if row.squeeze_setup or row.pullback_setup:
                setup='squeeze' if row.squeeze_setup else 'pullback'
                position=True
                entries[i]=True
                entry_index=i
                entry_atr=float(row.atr)
                entry_price=peak=float(row.close)  # replaced by the next open on next iteration
        active[i]=position
        exits[i]=bool(row.ready and not position)
        mandatory[i]=bool(exits[i] and last_exit in {'close_stop','data_gap','higher_trend_invalidated','time_exit'})
        setup_names.append(setup if position else 'none')
        reasons.append(('entry_'+setup if entries[i] else 'holding_'+setup) if position else last_exit)
    result=features.copy()
    result['entries'],result['exits'],result['virtual_position']=entries,exits,active
    result['mandatory_exit'],result['setup'],result['signal_reason']=mandatory,setup_names,reasons
    return result


def run_intraday_strategy(df, params=IntradayParams()):
    return signals_from_features(intraday_features(df,params),params)


def intraday_decision(row, in_position):
    action='HOLD'
    if in_position and row['exits']:
        action='SELL'
    elif not in_position and row['entries']:
        action='BUY'
    evidence=100. if action=='SELL' and row['mandatory_exit'] else 80. if action!='HOLD' else 0.
    allowed=['HOLD']
    if in_position:
        allowed.append('SELL')
    elif row['ready'] and not row['entry_blocked']:
        allowed.append('BUY')
    return {
        'action':action,'confidence':int(evidence),'reason':str(row['signal_reason']),
        'scores':{'BUY':evidence if action=='BUY' else 0.,'SELL':evidence if action=='SELL' else 0.,'HOLD':100.-evidence},
        'allowed_actions':allowed,'entry_blocked':bool(row['entry_blocked']),
        'risk_blocked':not bool(row['ready']),'should_call_gpt':action!='HOLD' and not bool(row['mandatory_exit']),
        'mandatory_exit':bool(row['mandatory_exit']),'regime_label':'uptrend_4h' if row['bull_4h'] else 'not_uptrend_4h',
        'coverage':1.,'entry_threshold':60.,'exit_threshold':55.,'mixed_signals':False,
    }
