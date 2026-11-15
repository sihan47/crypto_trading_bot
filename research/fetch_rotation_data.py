"""Public Binance spot candles only. Does not load dotenv or account keys."""
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]


def fetch(symbol):
    path=ROOT/f'data/research_rotation/{symbol}.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        print('Cached',symbol,flush=True)
        return
    start=int(pd.Timestamp('2022-09-01',tz='UTC').timestamp()*1000)
    end=int(pd.Timestamp('2026-11-14',tz='UTC').timestamp()*1000)
    rows=[]
    with requests.Session() as session:
        while start<end:
            response=session.get('https://api.binance.com/api/v3/klines',params={
                'symbol':symbol,'interval':'1h','startTime':start,'endTime':end-1,'limit':1000},timeout=30)
            response.raise_for_status()
            data=response.json()
            if not data:
                raise ValueError('Unexpected end of public candle data')
            rows.extend([x[:8] for x in data])
            nxt=int(data[-1][0])+3600000
            if nxt<=start:
                raise ValueError('Pagination stalled')
            start=nxt
            time.sleep(.12)
    df=pd.DataFrame(rows,columns=['timestamp','open','high','low','close','volume','close_time','quote_volume'])
    df.index=pd.to_datetime(df.pop('timestamp'),unit='ms',utc=True)
    df=df.drop(columns='close_time').astype(float)
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('Invalid candles')
    df.to_csv(path)
    print(symbol,len(df),flush=True)


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch,['BTCUSDT','BNBUSDT','ETHUSDT','SOLUSDT','XRPUSDT','ADAUSDT']))
