"""Causal research panel, preserving raw archives and explicit missing values."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from research.fetch_btc_metrics import FIELDS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/research_structural'


def clean_metrics(raw,delay_minutes=65):
    d=raw.copy()
    d['create_time']=pd.to_datetime(d.create_time,utc=True,format='mixed',errors='coerce')
    invalid_time=int(d.create_time.isna().sum());d=d.loc[d.create_time.notna()].copy()
    invalid={}
    for col in FIELDS:
        d[col]=pd.to_numeric(d[col],errors='coerce')
        bad=~np.isfinite(d[col]) | (d[col]<=0 if col.startswith('sum_open_interest') else d[col]<0)
        invalid[col]=int(bad.sum());d.loc[bad,col]=np.nan
    before=len(d)
    d=d.drop_duplicates(subset=['create_time']+FIELDS)
    exact_duplicates=before-len(d)
    conflicting=d.create_time.duplicated(keep=False)
    conflict_times=int(d.loc[conflicting,'create_time'].nunique())
    # No last-wins or averages for conflicting source snapshots.
    d=d.loc[~conflicting].sort_values('create_time').reset_index(drop=True)
    d['metric_available_at']=d.create_time+pd.Timedelta(minutes=delay_minutes)
    gaps=d.create_time.diff()
    audit={'input_rows':len(raw),'clean_rows':len(d),'invalid_timestamps':invalid_time,
           'invalid_or_missing_by_field':invalid,'exact_duplicates_removed':exact_duplicates,
           'conflicting_timestamps_quarantined':conflict_times,
           'gaps_over_5_minutes':int(gaps.gt(pd.Timedelta(minutes=5)).sum()),
           'max_gap_minutes':float(gaps.max()/pd.Timedelta(minutes=1)) if len(d)>1 else None,
           'assumed_availability_delay_minutes':delay_minutes,
           'availability_note':'Label +5min interval uncertainty +60min publication buffer; assumption, not verified point-in-time publication log.'}
    return d,audit


def align_metrics(prices,metrics):
    return pd.merge_asof(prices.sort_values('decision_time'),metrics.sort_values('metric_available_at'),
                         left_on='decision_time',right_on='metric_available_at',direction='backward',
                         tolerance=pd.Timedelta(minutes=10))


def read_market(path):
    d=pd.read_csv(path,index_col=0)
    d.index=pd.to_datetime(d.index,utc=True,format='mixed')
    if d.index.has_duplicates or not d.index.is_monotonic_increasing: raise ValueError('Invalid market index')
    return d


def main():
    manifest=json.loads((OUT/'metrics_download_manifest.json').read_text(encoding='utf-8'))
    frames=[]
    for item in manifest['files']:
        if item['status']!='verified': continue
        path=OUT/'metrics_archives'/f"BTCUSDT-metrics-{item['day']}.zip"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']: raise ValueError('Archive changed after verification')
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith('.csv')]
            df=pd.read_csv(z.open(names[0]))
        df['source_day']=item['day'];frames.append(df)
    raw=pd.concat(frames,ignore_index=True)
    cleaned,audit=clean_metrics(raw)
    cleaned.to_csv(OUT/'btc_metrics_5m.csv',index=False)
    spot=read_market(ROOT/'data/research_rotation/BTCUSDT.csv')
    perp=read_market(OUT/'klines.csv');mark=read_market(OUT/'markPriceKlines.csv')
    d=spot.add_prefix('spot_').join(perp.add_prefix('perp_'),how='outer').join(mark[['open','high','low','close']].add_prefix('mark_'),how='outer')
    d['candle_open_time']=d.index;d['decision_time']=d.index+pd.Timedelta(hours=1)
    panel=align_metrics(d.reset_index(drop=True),cleaned)
    funding=read_market(OUT/'fundingRate.csv').reset_index()
    funding=funding.rename(columns={funding.columns[0]:'funding_time','fundingRate':'funding_rate','markPrice':'funding_mark_price'})
    funding['funding_available_at']=funding.funding_time+pd.Timedelta(hours=1)
    panel=pd.merge_asof(panel.sort_values('decision_time'),funding.sort_values('funding_available_at'),
                        left_on='decision_time',right_on='funding_available_at',direction='backward',tolerance=pd.Timedelta(hours=12))
    panel['perp_spot_basis_bps']=(panel.perp_close/panel.spot_close-1)*10000
    panel['mark_spot_basis_bps']=(panel.mark_close/panel.spot_close-1)*10000
    panel['metrics_age_minutes']=(panel.decision_time-panel.create_time)/pd.Timedelta(minutes=1)
    panel['spot_tradeable']=panel.spot_volume.gt(0)&panel.spot_close.notna()
    panel['core_data_complete']=panel[['sum_open_interest','count_long_short_ratio','funding_rate','perp_spot_basis_bps']].notna().all(axis=1)&panel.spot_tradeable
    panel=panel.set_index('decision_time')
    panel.to_csv(OUT/'btc_hourly_research.csv')
    audit.update({'requested_start':manifest['start'],'requested_end_exclusive':manifest['end_exclusive'],
        'verified_files':sum(x['status']=='verified' for x in manifest['files']),
        'missing_file_dates':[x['day'] for x in manifest['files'] if x['status']!='verified'],
        'metrics_start':str(cleaned.create_time.min()),'metrics_end':str(cleaned.create_time.max()),
        'hourly_rows':len(panel),'core_complete_rows':int(panel.core_data_complete.sum()),
        'hourly_missing_by_field':panel[FIELDS+['funding_rate','perp_spot_basis_bps']].isna().sum().to_dict(),
        'metric_snapshot_age_minutes_max':float(panel.metrics_age_minutes.max()),
        'yearly_core_coverage':{str(y):float(g.core_data_complete.mean()) for y,g in panel.groupby(panel.index.year)},
        'sources':['https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/',
                   'https://fapi.binance.com/fapi/v1/fundingRate','https://fapi.binance.com/fapi/v1/klines',
                   'https://api.binance.com/api/v3/klines'],
        'timestamp_warning':'Community issue #491 reports label-convention changes. Uniform conservative delay used; no claim of complete independent verification.',
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'btc_metrics_5m.csv',OUT/'btc_hourly_research.csv',OUT/'metrics_download_manifest.json']}})
    (ROOT/'research/btc_derivatives_data_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'README.md').write_text('''# BTC derivatives research data

All timestamps UTC. No private API credentials required or stored.

* metrics_archives/: original daily ZIP + official SHA-256 CHECKSUM files.
* metrics_download_manifest.json: per-day URL, availability, hash, row count.
* btc_metrics_5m.csv: sorted, validated snapshots; conflicting duplicates excluded.
* btc_hourly_research.csv: index is DECISION time, not candle OPEN time. Price columns are already closed; metric/funding columns obey explicit availability delays.
* fundingRate.csv: actual settlement events; not predicted future funding.
* klines.csv / markPriceKlines.csv: perpetual trade/mark OHLC.

OI quantity and OI USDT value are different: a price change alone moves OI value.
count_long_short_ratio = account-count ratio, NOT net BTC exposure.
count_toptrader_long_short_ratio = top-trader account-count ratio.
sum_toptrader_long_short_ratio = top-trader position ratio.
sum_taker_long_short_vol_ratio = taker buy/sell volume ratio; do NOT average ratios as though they were additive volumes.
perp_spot_basis_bps uses aligned hourly CLOSE prices, NOT executable bid/ask spreads.

Metrics available_at = raw create_time +65 minutes (5min label uncertainty +60min assumed publication buffer). Funding available_at = actual funding_time +1 hour. These are conservative assumptions, not verified historical publication logs.
No backward fill; unmatched/stale fields remain NaN. Inspect core_data_complete AND the exact fields your hypothesis uses. Missing data is not zero OI, zero funding, or neutral positioning.
Do not feed this full historical file directly into a past-time GPT decision: subset by decision time first.

Rebuild: python research/fetch_btc_metrics.py ; python research/build_btc_derivatives_panel.py
Audit: research/btc_derivatives_data_audit.json. Files in data/ are intentionally excluded from Git.
''',encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['verified_files','missing_file_dates','hourly_rows','core_complete_rows','yearly_core_coverage','conflicting_timestamps_quarantined']},indent=2),flush=True)


if __name__=='__main__': main()
