"""Download official public Binance daily BTC metrics with checksum verification.

No credentials, account calls, trading endpoints, or notifications.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/research_structural'
CACHE=OUT/'metrics_archives'
BASE='https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/'
FIELDS=['sum_open_interest','sum_open_interest_value','count_toptrader_long_short_ratio',
        'sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio']


def fetch_day(day):
    name=f'BTCUSDT-metrics-{day}.zip';path=CACHE/name;check=path.with_suffix('.zip.CHECKSUM')
    url=BASE+name
    for attempt in range(3):
        try:
            if path.exists() and check.exists():
                body=path.read_bytes();expected=check.read_text().split()[0]
            else:
                with requests.Session() as session:
                    r=session.get(url,timeout=25)
                    if r.status_code==404: return {'day':day,'status':'missing_404','url':url}
                    r.raise_for_status()
                    c=session.get(url+'.CHECKSUM',timeout=25);c.raise_for_status()
                    body=r.content;expected=c.text.split()[0]
                if hashlib.sha256(body).hexdigest()!=expected: raise ValueError('Checksum mismatch')
                path.write_bytes(body);check.write_text(expected+'  '+name+'\n',encoding='ascii')
            actual=hashlib.sha256(body).hexdigest()
            if actual!=expected: raise ValueError('Cached checksum mismatch')
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                members=[n for n in z.namelist() if n.endswith('.csv')]
                if len(members)!=1: raise ValueError('Unexpected archive structure')
                frame=pd.read_csv(z.open(members[0]))
            if not set(['create_time','symbol']+FIELDS)<=set(frame.columns): raise ValueError('Missing metrics fields')
            if not frame.symbol.eq('BTCUSDT').all(): raise ValueError('Unexpected symbol')
            return {'day':day,'status':'verified','sha256':actual,'rows':len(frame),'url':url}
        except (requests.RequestException,ValueError,zipfile.BadZipFile,IndexError) as exc:
            if attempt==2: return {'day':day,'status':'error','error_type':type(exc).__name__,'url':url}
            time.sleep(1+attempt)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',default='2022-09-01');parser.add_argument('--end',default='2026-11-14',help='Exclusive UTC date')
    args=parser.parse_args();CACHE.mkdir(parents=True,exist_ok=True)
    days=[d.strftime('%Y-%m-%d') for d in pd.date_range(args.start,args.end,inclusive='left',freq='1D')]
    manifest=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(fetch_day,day) for day in days]
        for task in as_completed(tasks):
            manifest.append(task.result())
            if len(manifest)%100==0: print('Processed',len(manifest),'/',len(days),flush=True)
    manifest.sort(key=lambda x:x['day'])
    report={'symbol':'BTCUSDT','start':args.start,'end_exclusive':args.end,'files':manifest,
            'retrieved_at':pd.Timestamp.now(tz='UTC').isoformat()}
    (OUT/'metrics_download_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    counts=pd.Series([x['status'] for x in manifest]).value_counts().to_dict()
    print(json.dumps(counts),flush=True)
    if counts.get('error',0): raise SystemExit('Some downloads failed; manifest retained, rerun to retry.')


if __name__=='__main__': main()
