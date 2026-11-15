"""Public Binance archives + previously acquired public funding/metrics; no credentials."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import io
import json
import time
import zipfile
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/research_structural'
CACHE = DATA/'flow_archives'
FIELDS = ['open_time','open','high','low','close','volume','close_time','quote_volume',
          'trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']


def fetch_archive(job):
    frequency, stamp = job
    name = f'BTCUSDT-1h-{stamp}.zip'
    url = f'https://data.binance.vision/data/futures/um/{frequency}/klines/BTCUSDT/1h/{name}'
    path = CACHE/name; check = CACHE/(name+'.CHECKSUM')
    for attempt in range(3):
        try:
            if path.exists() and check.exists():
                body = path.read_bytes(); expected = check.read_text().split()[0]
            else:
                r = requests.get(url, timeout=40); r.raise_for_status()
                c = requests.get(url+'.CHECKSUM', timeout=40); c.raise_for_status()
                body = r.content; expected = c.text.split()[0]
            digest = hashlib.sha256(body).hexdigest()
            if digest != expected: raise ValueError('Archive checksum mismatch')
            path.write_bytes(body); check.write_text(expected+'  '+name, encoding='ascii')
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                members = [x for x in z.namelist() if x.endswith('.csv')]
                if len(members) != 1: raise ValueError('Unexpected archive members')
                frame = pd.read_csv(z.open(members[0]), header=None, names=FIELDS)
            # Some Futures archives contain a header; remove it only by name.
            frame = frame.loc[~frame.open_time.astype(str).isin(['open_time','openTime'])].copy()
            for col in FIELDS: frame[col] = pd.to_numeric(frame[col], errors='raise')
            unit = 'us' if frame.open_time.median() > 1e14 else 'ms'
            frame.index = pd.to_datetime(frame.open_time, unit=unit, utc=True)+pd.Timedelta(hours=1)
            return frame, {'url':url,'sha256':digest,'rows':len(frame),'timestamp_unit':unit}
        except (requests.RequestException, ValueError, zipfile.BadZipFile):
            if attempt == 2: raise
            time.sleep(attempt+1)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [('monthly', x.strftime('%Y-%m')) for x in pd.date_range('2022-09-01','2026-10-01',freq='MS')]
    jobs += [('daily', x.strftime('%Y-%m-%d')) for x in pd.date_range('2026-11-01','2026-11-13')]
    frames=[]; manifest=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f, record in pool.map(fetch_archive, jobs):
            frames.append(f); manifest.append(record)
            if len(frames)%10 == 0: print('Verified flow archives',len(frames),'/',len(jobs),flush=True)
    flow=pd.concat(frames).sort_index()
    if flow.index.has_duplicates: raise ValueError('Duplicate candle timestamps')
    expected=pd.date_range('2022-09-01 01:00','2026-11-14 00:00',freq='h',tz='UTC')
    flow=flow.reindex(expected)
    valid=(flow.volume.gt(0)&flow.taker_buy_base_asset_volume.between(0,flow.volume)
           &flow[['open','high','low','close']].gt(0).all(axis=1)
           &flow.high.ge(flow[['open','close','low']].max(axis=1))
           &flow.low.le(flow[['open','close','high']].min(axis=1)))
    flow.loc[~valid,:]=np.nan
    flow.index.name='decision_time'
    flow.to_csv(DATA/'btc_futures_flow_1h.csv')
    panel=pd.read_csv(DATA/'btc_hourly_research.csv',index_col=0)
    panel.index=pd.to_datetime(panel.index,utc=True)
    panel=panel.reindex(expected)
    for col in ['open','high','low','close','volume','quote_volume','trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume']:
        panel['futures_'+col]=flow[col]
    panel['flow_imbalance']=(2*flow.taker_buy_base_asset_volume-flow.volume)/flow.volume
    # Verify the cached official metrics archives again, not just the panel hash.
    old=json.loads((DATA/'metrics_download_manifest.json').read_text())
    for item in old['files']:
        if item['status'] != 'verified': raise ValueError('Incomplete metrics archive inventory')
        p=DATA/'metrics_archives'/f"BTCUSDT-metrics-{item['day']}.zip"
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']: raise ValueError('Metrics cache changed')
    audit=json.loads((ROOT/'research/btc_derivatives_data_audit.json').read_text())
    if hashlib.sha256((DATA/'btc_hourly_research.csv').read_bytes()).hexdigest()!=audit['sha256']['btc_hourly_research.csv']:
        raise ValueError('Existing panel differs from audited build')
    fields=['futures_close','futures_volume','futures_taker_buy_base_asset_volume','sum_open_interest',
            'funding_rate','sum_toptrader_long_short_ratio','count_toptrader_long_short_ratio','count_long_short_ratio']
    panel.index.name='decision_time'; panel.to_csv(DATA/'btc_orthogonal_1h.csv')
    report={'start':str(panel.index.min()),'end':str(panel.index.max()),'rows':len(panel),
            'invalid_or_missing_flow_hours':int((~valid).sum()),'metrics_archives_reverified':len(old['files']),
            'missing':panel[fields].isna().sum().to_dict(),
            'complete_all_dimensions':int(panel[fields].notna().all(axis=1).sum()),
            'yearly_complete_fraction':{str(y):float(g[fields].notna().all(axis=1).mean()) for y,g in panel.groupby(panel.index.year)},
            'funding_source':'Previously fetched actual settlements: https://fapi.binance.com/fapi/v1/fundingRate; +1h assumed availability',
            'metrics_availability':'Source label +65min; assumption, not historical publication logs',
            'semantics':'Top position=sum_toptrader_long_short_ratio; top account=count_toptrader_long_short_ratio; global account=count_long_short_ratio. Global is not pure retail. Flow imbalance is aggressive executed-volume imbalance, not order-book OFI.',
            'sha256':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['btc_orthogonal_1h.csv','fundingRate.csv','btc_metrics_5m.csv']},
            'flow_archives':manifest}
    (ROOT/'research/orthogonal_data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='flow_archives'},indent=2),flush=True)


if __name__=='__main__': main()
