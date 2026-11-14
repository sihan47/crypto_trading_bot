"""Causal, long-only strategy families. No accounts, network or LLM calls.

Prices are closed bars indexed by their opening time. A signal at t can only
be executed at t+1. Scores are evidence scores, NOT calibrated probabilities.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.ema_adx_strategy import _compute_adx


@dataclass(frozen=True)
class EnsembleParams:
    trend_weight: float = 0.5
    breakout_weight: float = 0.3
    reversion_weight: float = 0.2
    entry_threshold: float = 60.0
    exit_threshold: float = 55.0
    min_coverage: float = 0.25
    min_margin: float = 10.0
    max_atr_pct: float = 0.018
    atr_percentile_limit: float = 0.85
    fast: int = 48
    slow: int = 192
    breakout_entry: int = 96
    breakout_exit: int = 48
    band_window: int = 20

    def __post_init__(self):
        weights = (self.trend_weight, self.breakout_weight, self.reversion_weight)
        if not all(np.isfinite(v) and v >= 0 for v in weights) or sum(weights) <= 0:
            raise ValueError('Family weights must be finite, nonnegative and sum to > 0')
        if not 0 < self.fast < self.slow or min(self.breakout_entry, self.breakout_exit, self.band_window) < 2:
            raise ValueError('Invalid indicator windows')
        if not 0 <= self.min_coverage <= 1 or not 0 < self.atr_percentile_limit <= 1:
            raise ValueError('Invalid coverage/percentile')
        if not all(0 <= v <= 100 for v in (self.entry_threshold, self.exit_threshold, self.min_margin)):
            raise ValueError('Scores must be on a 0-100 scale')
        if not np.isfinite(self.max_atr_pct) or self.max_atr_pct <= 0:
            raise ValueError('Invalid ATR cap')


def family_features(df: pd.DataFrame, params: EnsembleParams = EnsembleParams()) -> pd.DataFrame:
    """Vectorized prefix-invariant features shared by research and live use."""
    if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError('Expected unique, ascending datetime-indexed bars')
    values = df[['open', 'high', 'low', 'close', 'volume']]
    if not np.isfinite(values.to_numpy()).all() or (values[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError('Invalid OHLCV values')
    close, high, low = df['close'], df['high'], df['low']
    adx_pack = _compute_adx(high, low, close, 14)
    atr, adx = adx_pack['atr'], adx_pack['adx']
    fast = close.ewm(span=params.fast, adjust=False, min_periods=params.fast).mean()
    slow = close.ewm(span=params.slow, adjust=False, min_periods=params.slow).mean()
    regime = pd.Series(np.where(adx >= 25, 'trend', np.where(adx < 20, 'range', 'mixed')), index=df.index)
    trend = pd.Series(0.0, index=df.index)
    trend.loc[(adx >= 20) & (fast > slow) & (close > slow)] = 1
    trend.loc[(adx >= 20) & (fast < slow) & (close < slow)] = -1
    trend_conf = (60 + (fast - slow).abs().div(atr.replace(0, np.nan)).clip(0, 3) * 10).fillna(0)

    upper = high.rolling(params.breakout_entry).max().shift(1)
    lower = low.rolling(params.breakout_exit).min().shift(1)
    breakout_events = pd.Series(np.nan, index=df.index)
    breakout_events.loc[(close > upper) & (close > slow)] = 1
    breakout_events.loc[close < lower] = -1
    breakout = breakout_events.ffill().fillna(0)
    # Only confirmed up/down channel state; do not count a prior exit as a buy.
    breakout_conf = pd.Series(75.0, index=df.index)

    basis = close.rolling(params.band_window).mean()
    std = close.rolling(params.band_window).std(ddof=0)
    lower_band = basis - 2 * std
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 * gain / (gain + loss).replace(0, np.nan)
    # Confirmation: price returns inside the lower band in a range, not a falling knife.
    entry = (regime == 'range') & (close.shift(1) < lower_band.shift(1)) & (close >= lower_band) & (rsi < 45)
    exit_ = (close >= basis) | ((fast < slow) & (adx >= 25))
    reversion_events = pd.Series(np.nan, index=df.index)
    reversion_events.loc[entry] = 1
    reversion_events.loc[exit_] = -1
    reversion = reversion_events.ffill().fillna(0)
    # Do not let an old mean-reversion buy persist into a directional decline.
    reversion.loc[(regime != 'range') & (reversion > 0)] = 0
    reversion_conf = pd.Series(70.0, index=df.index)
    atr_pct = atr / close
    atr_rank = atr_pct.rolling(96).rank(pct=True)
    ready = slow.notna() & atr_rank.notna() & upper.notna() & adx.notna()
    return pd.DataFrame({
        'trend': trend, 'breakout': breakout, 'reversion': reversion,
        'trend_conf': trend_conf, 'breakout_conf': breakout_conf, 'reversion_conf': reversion_conf,
        'regime': regime, 'atr_pct': atr_pct, 'atr_rank': atr_rank, 'adx': adx,
        'ready': ready, 'close': close,
    }, index=df.index)


def score_families(features: pd.DataFrame, params: EnsembleParams = EnsembleParams()) -> pd.DataFrame:
    """Normalize on active family weight, with independent minimum coverage."""
    weights = pd.DataFrame({
        'trend': params.trend_weight,
        'breakout': params.breakout_weight,
        'reversion': params.reversion_weight,
    }, index=features.index)
    is_range = features['regime'].eq('range')
    is_trend = features['regime'].eq('trend')
    weights.loc[is_range, ['trend', 'breakout']] *= 0.5
    weights.loc[is_range, 'reversion'] *= 2.0
    weights.loc[is_trend, 'reversion'] *= 0.25
    directions = features[['trend', 'breakout', 'reversion']]
    confidence = features[['trend_conf', 'breakout_conf', 'reversion_conf']].copy()
    confidence.columns = directions.columns
    active = weights.where(directions.ne(0), 0)
    denominator = active.sum(axis=1).replace(0, np.nan)
    buy = (active * confidence).where(directions.gt(0), 0).sum(axis=1).div(denominator).fillna(0)
    sell = (active * confidence).where(directions.lt(0), 0).sum(axis=1).div(denominator).fillna(0)
    coverage = denominator.div(weights.sum(axis=1)).fillna(0)
    blocked = (features['atr_pct'] > params.max_atr_pct) | (features['atr_rank'] > params.atr_percentile_limit)
    entries = features['ready'] & ~blocked & (coverage >= params.min_coverage) & (buy >= params.entry_threshold) & (buy - sell >= params.min_margin)
    # Volatility NEVER vetoes a reduction of an existing position.
    exits = features['ready'] & (coverage >= params.min_coverage) & (sell >= params.exit_threshold) & (sell - buy >= params.min_margin)
    result = features.copy()
    result['buy_score'], result['sell_score'], result['coverage'] = buy, sell, coverage
    result['entry_blocked'], result['entries'], result['exits'] = blocked, entries, exits
    return result


def run_regime_ensemble(df: pd.DataFrame, params: EnsembleParams = EnsembleParams()) -> pd.DataFrame:
    return score_families(family_features(df, params), params)


def decision_from_row(row: pd.Series, in_position: bool, params: EnsembleParams) -> dict:
    buy, sell = float(row['buy_score']), float(row['sell_score'])
    allowed = ['HOLD']
    if bool(row['ready']):
        if not bool(row['entry_blocked']):
            allowed.append('BUY')
        if in_position:
            allowed.append('SELL')
    action = 'SELL' if bool(row['exits']) and in_position else 'BUY' if bool(row['entries']) else 'HOLD'
    if not bool(row['ready']):
        reason = 'Indicator warm-up incomplete.'
    elif action == 'HOLD' and bool(row['entry_blocked']):
        reason = f'New buys blocked by ATR; sells remain eligible. buy={buy:.1f}, sell={sell:.1f}.'
    else:
        reason = f'Family scores: buy={buy:.1f}, sell={sell:.1f}, coverage={row["coverage"]:.2f}; {action}.'
    return {
        'action': action, 'confidence': int(round(max(buy, sell))), 'reason': reason,
        'scores': {'BUY': round(buy, 2), 'SELL': round(sell, 2), 'HOLD': round(100 - max(buy, sell), 2)},
        'allowed_actions': allowed, 'entry_blocked': bool(row['entry_blocked']),
        'risk_blocked': not bool(row['ready']), 'mixed_signals': buy > 0 and sell > 0,
        'should_call_gpt': action != 'HOLD', 'regime_label': str(row['regime']),
        'coverage': round(float(row['coverage']), 3),
        'entry_threshold': params.entry_threshold, 'exit_threshold': params.exit_threshold,
    }
