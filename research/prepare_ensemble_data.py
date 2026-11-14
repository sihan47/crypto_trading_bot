"""Export complete local 15m bars, then append public mainnet klines. No keys."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'research_15m'


def prepare(symbol, end):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{symbol}.csv'
    if path.exists():
        bars = pd.read_csv(path, index_col=0, parse_dates=True)
        bars.index = pd.to_datetime(bars.index, utc=True)
    else:
        with sqlite3.connect(f'file:{(ROOT / "data/local_data.db").as_posix()}?mode=ro', uri=True) as conn:
            raw = pd.read_sql_query(
                'SELECT timestamp,open,high,low,close,volume FROM klines WHERE symbol=? AND interval=? ORDER BY timestamp',
                conn, params=(symbol, '1m'))
        raw.index = pd.to_datetime(raw.pop('timestamp'), unit='ms', utc=True)
        groups = raw.resample('15min', closed='left', label='left')
        bars = groups.agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
        bars = bars.loc[groups['close'].count() == 15]
        del raw
    step = 15 * 60 * 1000
    cursor = int((bars.index[-1] + pd.Timedelta(minutes=15)).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz='UTC').timestamp() * 1000)
    chunks = []
    with requests.Session() as session:
        while cursor < end_ms:
            response = session.get('https://api.binance.com/api/v3/klines', params={
                'symbol':symbol, 'interval':'15m', 'startTime':cursor, 'endTime':end_ms-1, 'limit':1000}, timeout=30)
            response.raise_for_status()
            rows = response.json()
            if not rows:
                raise RuntimeError(f'{symbol}: unexpected end of market data')
            frame = pd.DataFrame([r[:6] for r in rows], columns=['timestamp','open','high','low','close','volume'])
            frame.index = pd.to_datetime(frame.pop('timestamp'), unit='ms', utc=True)
            chunks.append(frame.astype(float))
            next_cursor = int(rows[-1][0]) + step
            if next_cursor <= cursor:
                raise RuntimeError('Pagination did not advance')
            cursor = next_cursor
            time.sleep(.1)
    bars = pd.concat([bars, *chunks]).sort_index()
    bars = bars.loc[~bars.index.duplicated(keep='last')]
    bars = bars.loc[bars.index + pd.Timedelta(minutes=15) <= pd.Timestamp(end, tz='UTC')]
    bars.index.name = 'open_time'
    bars.to_csv(path)
    print(symbol, len(bars), str(bars.index[0]), str(bars.index[-1]),
          'gaps', int((bars.index.to_series().diff().dropna() != pd.Timedelta(minutes=15)).sum()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--end', required=True, help='Exclusive UTC date, e.g. 2026-11-14')
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda s: prepare(s, args.end), ['BTCUSDT','BNBUSDT']))
