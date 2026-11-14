"""Single-bot durable bar/order guard. Ambiguous orders require reconciliation."""
import json
import os
from pathlib import Path
import re

import pandas as pd


def timeframe_delta(timeframe):
    match = re.fullmatch(r'(\d+)([mhd])', timeframe)
    if not match or int(match[1]) <= 0:
        raise ValueError('Unsupported timeframe')
    return pd.Timedelta(int(match[1]), unit={'m':'min','h':'h','d':'D'}[match[2]])


def closed_bars(df, timeframe, now=None):
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None or df.index.tz is None:
        raise ValueError('UTC-aware timestamps required')
    delta = timeframe_delta(timeframe)
    result = df.loc[df.index + delta <= now]
    if not result.empty and now - (result.index[-1] + delta) > delta + pd.Timedelta(minutes=2):
        raise ValueError('Latest closed bar is stale')
    return result


class DecisionState:
    def __init__(self, path):
        self.path = Path(path)

    def _read(self):
        if not self.path.exists():
            return {'bars':{}, 'orders':{}}
        state = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(state.get('bars'),dict) or not isinstance(state.get('orders'),dict):
            raise ValueError('Invalid decision journal; reconcile instead of resetting')
        return state

    def _write(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8') as handle:
            json.dump(state,handle,indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(self.path)

    def claim_bar(self, symbol, timeframe, stamp):
        state = self._read()
        key = f'{symbol}:{timeframe}'
        stamp = pd.Timestamp(stamp)
        previous = state['bars'].get(key)
        if previous and stamp <= pd.Timestamp(previous):
            return False
        state['bars'][key] = stamp.isoformat()
        self._write(state)
        return True

    def begin_order(self, symbol, side, stamp):
        if side not in ('BUY','SELL'):
            raise ValueError('Unexpected order side')
        state = self._read()
        record = state['orders'].get(symbol,{})
        if record.get('pending'):
            return False, 'Previous order outcome is unresolved; reconcile with exchange first'
        if record.get('last_side') == side:
            return False, 'Same-side signal already traded; waiting for opposite executed signal'
        record['pending'] = {'side':side, 'bar':str(stamp)}
        state['orders'][symbol] = record
        self._write(state)
        return True, 'Order intent persisted'

    def complete_order(self, symbol, order):
        if not isinstance(order,dict) or not order.get('orderId'):
            raise ValueError('Confirmed exchange order required')
        state = self._read()
        record = state['orders'][symbol]
        record['last_side'] = record.pop('pending')['side']
        record['order_id'] = order['orderId']
        self._write(state)

    def record_shadow(self, symbol, side, stamp):
        state = self._read()
        shadow = state.setdefault('shadow', {})
        if shadow.get(symbol, {}).get('side') == side:
            return False
        shadow[symbol] = {'side': side, 'bar': str(stamp)}
        self._write(state)
        return True
