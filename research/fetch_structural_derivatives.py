"""Public market-data download only; never loads credentials or account SDKs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/research_structural';OUT.mkdir(parents=True,exist_ok=True)
START=int(pd.Timestamp('2022-09-01',tz='UTC').timestamp()*1000)
END=int(pd.Timestamp('2026-11-14',tz='UTC').timestamp()*1000)


def fetch(route):
    target=OUT/f'{route}.csv'
    if target.exists():
        print('Cached',route,flush=True);return
    cursor=START;rows=[]
    with requests.Session() as session:
        while cursor<END:
            params={'symbol':'BTCUSDT','startTime':cursor,'endTime':END-1,'limit':1000}
            if route!='fundingRate': params['interval']='1h'
            r=session.get('https://fapi.binance.com/fapi/v1/'+route,params=params,timeout=30)
            r.raise_for_status();part=r.json()
            if not isinstance(part,list) or not part: raise ValueError(f'Incomplete {route} history at {cursor}')
            rows.extend(part)
            nxt=int(part[-1]['fundingTime'])+1 if route=='fundingRate' else int(part[-1][0])+3600000
            if nxt<=cursor: raise ValueError('Pagination stalled')
            cursor=nxt
            if route=='fundingRate' and len(part)<1000: break
            time.sleep(.15)
    if route=='fundingRate':
        d=pd.DataFrame(rows)
        d.index=pd.to_datetime(d.pop('fundingTime'),unit='ms',utc=True)
        d=d[['fundingRate','markPrice']]
    else:
        d=pd.DataFrame([x[:6] for x in rows],columns=['timestamp','open','high','low','close','volume'])
        d.index=pd.to_datetime(d.pop('timestamp'),unit='ms',utc=True)
    if d.index.has_duplicates or not d.index.is_monotonic_increasing: raise ValueError('Invalid timestamps')
    d.to_csv(target)
    print(route,len(d),str(d.index[0]),str(d.index[-1]),flush=True)


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch,['fundingRate','klines','markPriceKlines']))
    manifest={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in OUT.glob('*.csv')}
    manifest['source']='https://fapi.binance.com/fapi/v1/ public endpoints; symbol BTCUSDT; 2022-09-01 through 2026-11-13 UTC'
    (ROOT/'research/structural_derivatives_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
