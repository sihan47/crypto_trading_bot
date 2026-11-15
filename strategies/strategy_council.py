"""Causal, grouped strategy evidence. Signals are advice, not separate orders.

Input timestamps are OPEN times of already closed UTC 4h candles. Daily
evidence becomes available at the 20:00 candle's close, never earlier.
Parameters are fixed before evaluation; no performance-based weight fitting.
"""
import numpy as np
import pandas as pd

GROUPS = {
    'band50': 'trend', 'momentum84': 'trend', 'zscore65': 'trend',
    'donchian20': 'breakout', 'keltner20': 'breakout', 'pullback100': 'pullback',
}
WEIGHTS = {
    'trend': {'trend': .50, 'breakout': .30, 'pullback': .20},
    'range': {'trend': .30, 'breakout': .25, 'pullback': .45},
    'mixed': {'trend': .40, 'breakout': .30, 'pullback': .30},
}
RULES = {
    'band50': 'BUY above 50-day SMA +1%; SELL below -1%; otherwise neutral.',
    'momentum84': 'BUY 84-day return >1%; SELL <-1%; otherwise neutral.',
    'zscore65': 'BUY (close - 65-day SMA)/65-day std >0.5; SELL <-0.5.',
    'donchian20': 'BUY above prior 20-day high; SELL below prior 10-day low.',
    'keltner20': 'BUY above rising daily EMA20 +2 ATR10; SELL below EMA20.',
    'pullback100': 'BUY daily RSI2 crosses below 15 above SMA100; SELL on rebound cross above 30 or below SMA100. Exit is advice, not proof this lot entered on a pullback.',
}


def finite_ema(series, alpha, window):
    """Finite exponential window: reproducible with a bounded runtime history."""
    weights = (1-alpha)**np.arange(window-1, -1, -1)
    weights /= weights.sum()
    return series.rolling(window).apply(lambda values: np.dot(values, weights), raw=True)


def council_features(df):
    idx = df.index
    if (not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or idx.has_duplicates
            or not idx.is_monotonic_increasing or len(df) == 0):
        raise ValueError('Ordered timezone-aware closed bars required')
    if any(t.hour % 4 or t.minute or t.second for t in idx.tz_convert('UTC')):
        raise ValueError('Council requires 4h UTC opening timestamps')
    numeric = df[['open', 'high', 'low', 'close', 'volume']]
    if (not np.isfinite(numeric).all().all() or (numeric.iloc[:, :4] <= 0).any().any()
            or (df.volume < 0).any() or (df.high < df[['open','close','low']].max(axis=1)).any()
            or (df.low > df[['open','close','high']].min(axis=1)).any()):
        raise ValueError('Invalid OHLCV')
    # Missing candles remain missing; do not silently turn observations into days.
    d = df.copy().tz_convert('UTC').asfreq('4h')
    c = d.close
    f = pd.DataFrame(index=d.index)
    f['close'] = c
    f['sma50'] = c.rolling(300).mean()
    f['momentum84'] = c / c.shift(504) - 1
    sd = c.rolling(390).std(ddof=0)
    f['z65'] = ((c-c.rolling(390).mean()) / sd.replace(0, np.nan)).where(sd.ne(0), 0.)
    f['channel_high'] = d.high.rolling(120).max().shift(1)
    f['channel_low'] = d.low.rolling(60).min().shift(1)
    daily_group = d.resample('1D')
    daily = daily_group.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    daily.loc[daily_group.close.count().ne(6), :] = np.nan
    dc = daily.close
    tr = pd.concat([daily.high-daily.low, (daily.high-dc.shift()).abs(),
                    (daily.low-dc.shift()).abs()], axis=1).max(axis=1)
    q = pd.DataFrame(index=daily.index)
    q['daily_close'] = dc
    q['daily_sma100'] = dc.rolling(100).mean()
    q['daily_ema20'] = finite_ema(dc, 2/21, 100)
    q['daily_atr10'] = tr.rolling(10).mean()
    change = dc.diff()
    gain = finite_ema(change.clip(lower=0), .5, 30)
    loss = finite_ema(-change.clip(upper=0), .5, 30)
    q['daily_rsi2'] = (100*gain/(gain+loss)).where((gain+loss).ne(0), 50.)
    q['previous_rsi2'] = q.daily_rsi2.shift()
    q['keltner_buy'] = (dc > q.daily_ema20+2*q.daily_atr10) & (q.daily_ema20 > q.daily_ema20.shift())
    q['keltner_sell'] = dc < q.daily_ema20
    q['pullback_buy'] = (dc > q.daily_sma100) & (q.daily_rsi2 < 15) & (q.previous_rsi2 >= 15)
    q['pullback_sell'] = ((q.daily_rsi2 > 30) & (q.previous_rsi2 <= 30)) | (dc < q.daily_sma100)
    q['daily_as_of'] = q.index + pd.Timedelta(days=1)
    q.index = q.index + pd.Timedelta(hours=20)
    # A daily event is evidence for the next 24h, then expires.
    f = f.join(q.reindex(f.index, method='ffill', limit=5))
    movement = c.diff().abs().rolling(120).sum()
    f['efficiency20'] = ((c-c.shift(120)).abs()/movement.replace(0,np.nan)).where(movement.ne(0),0.)
    f['regime'] = np.select([f.efficiency20 >= .25, f.efficiency20 <= .10], ['trend','range'], default='mixed')
    f['ready'] = (c.notna().rolling(601).sum().eq(601)
                  & f[['sma50','momentum84','z65','daily_sma100','daily_rsi2','daily_ema20']].notna().all(axis=1))
    conditions = {
        'band50': (c > f.sma50*1.01, c < f.sma50*.99),
        'momentum84': (f.momentum84 > .01, f.momentum84 < -.01),
        'zscore65': (f.z65 > .5, f.z65 < -.5),
        'donchian20': (c > f.channel_high, c < f.channel_low),
        'keltner20': (f.keltner_buy.eq(True), f.keltner_sell.eq(True)),
        'pullback100': (f.pullback_buy.eq(True), f.pullback_sell.eq(True)),
    }
    for name, (buy, sell) in conditions.items():
        f[name+'_signal'] = np.select([f.ready & buy, f.ready & sell], [1,-1], default=0)
    return f.reindex(idx)


def council_cards(row):
    regime = row['regime']
    weights = WEIGHTS[regime]
    evidence = {k: (None if pd.isna(row[k]) else float(row[k])) for k in
                ['close','sma50','momentum84','z65','channel_high','channel_low',
                 'daily_close','daily_sma100','daily_ema20','daily_atr10','daily_rsi2','previous_rsi2','efficiency20']}
    cards = {}
    for name, group in GROUPS.items():
        signal = {1:'BUY', -1:'SELL', 0:'HOLD'}[int(row[name+'_signal'])]
        weight = weights[group] / list(GROUPS.values()).count(group)
        cards[name] = {'signal': signal, 'confidence': 100 if signal != 'HOLD' else 0,
                       'reason': RULES[name], 'category': group, 'regime_fit': regime, 'weight': weight,
                       'backtest_performance': 'No forecast probability; historical technical-only research excludes GPT.',
                       'evidence': evidence, 'ready': bool(row.ready),
                       'timeframe': '1d' if name in {'keltner20','pullback100'} else '4h',
                       'as_of': str(row.daily_as_of) if name in {'keltner20','pullback100'} else str(row.name+pd.Timedelta(hours=4))}
    market = {'last_close': float(row.close), 'regime_label': regime, 'ready': bool(row.ready),
              'group_budgets': weights, 'daily_as_of': str(row.daily_as_of), 'evidence': evidence,
              'signal_semantics': 'BUY/SELL are entry/exit advice, HOLD means no evidence; no shorting or independent expert orders.'}
    return cards, market


def council_decision(cards, market, in_position, authority=.4):
    buy = sum(x['weight'] for x in cards.values() if x['signal']=='BUY')
    sell = sum(x['weight'] for x in cards.values() if x['signal']=='SELL')
    active = buy+sell
    scores = {'BUY': 100*buy/active if active else 0., 'SELL': 100*sell/active if active else 0.,
              'HOLD': 100. if not active else 0.}
    side = 'SELL' if in_position else 'BUY'
    threshold = 55. if in_position else 60.
    allowed = ['HOLD']
    # Minimum coverage prevents one weak fragment of a correlated group from trading.
    if market['ready'] and active >= .20-1e-9 and scores[side] > 0 and (1-authority)*scores[side]+authority*100 >= threshold:
        allowed.append(side)
    action = side if side in allowed and scores[side] >= threshold else 'HOLD'
    return {'action': action, 'confidence': int(round(scores[action])), 'scores': scores,
            'allowed_actions': allowed, 'should_call_gpt': len(allowed)>1,
            'coverage': active, 'regime_label': market['regime_label'],
            'entry_threshold':60., 'exit_threshold':55.,
            'reason': f"Grouped evidence buy={scores['BUY']:.1f}, sell={scores['SELL']:.1f}; coverage={active:.2f}; GPT chooses final eligible action."}


def council_blend(local, review, authority):
    scores = dict(local['scores'])
    action = review.get('action') if review else 'HOLD'
    confidence = review.get('confidence', 0) if review else 0
    final = 'HOLD'
    valid = isinstance(confidence, (int,float)) and not isinstance(confidence,bool) and np.isfinite(confidence) and 0<=confidence<=100
    score = 0.
    if valid and action in {'BUY','SELL'} and action in local['allowed_actions']:
        score = (1-authority)*scores[action] + authority*confidence
        if score >= local['entry_threshold' if action=='BUY' else 'exit_threshold']:
            final = action
        scores[action] = round(score,2)
    return {'action': final, 'confidence': int(round(score)), 'blended_scores': scores,
            'gpt_used': bool(review),
            'reason': f"GPT={action}; technical/GPT score={score:.1f}; final={final}. " + str(review.get('reason','') if review else local['reason'])}
