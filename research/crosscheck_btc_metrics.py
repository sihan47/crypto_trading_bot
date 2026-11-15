"""Archive / REST snapshot comparison. No automatic timestamp correction."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/research_structural'
MAPPING={
    'openInterestHist':{'sumOpenInterest':'sum_open_interest','sumOpenInterestValue':'sum_open_interest_value'},
    'globalLongShortAccountRatio':{'longShortRatio':'count_long_short_ratio'},
    'topLongShortAccountRatio':{'longShortRatio':'count_toptrader_long_short_ratio'},
    'topLongShortPositionRatio':{'longShortRatio':'sum_toptrader_long_short_ratio'},
    'takerlongshortRatio':{'buySellRatio':'sum_taker_long_short_vol_ratio'},
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--day',default='2026-11-13');args=parser.parse_args()
    archive=pd.read_csv(OUT/'btc_metrics_5m.csv')
    archive.index=pd.to_datetime(archive.create_time,utc=True)
    cache=OUT/'rest_crosscheck';cache.mkdir(exist_ok=True)
    start=int(pd.Timestamp(args.day,tz='UTC').timestamp()*1000);end=start+86400000-1
    report={'date':args.day,'purpose':'Independent REST/archive field/time comparison; offsets are diagnostics, never fitted trading parameters.','fields':{}}
    for route,fields in MAPPING.items():
        path=cache/(route+'.json')
        if path.exists():
            rows=json.loads(path.read_text(encoding='utf-8'))
            if not all(start-300000<=x['timestamp']<=end+300000 for x in rows):
                raise ValueError('Cached REST date differs beyond one interval; preserve cache explicitly')
        else:
            r=requests.get('https://fapi.binance.com/futures/data/'+route,params={'symbol':'BTCUSDT','period':'5m','startTime':start,'endTime':end,'limit':500},timeout=25)
            r.raise_for_status();rows=r.json();path.write_text(json.dumps(rows,indent=2),encoding='utf-8')
        d=pd.DataFrame(rows);d.index=pd.to_datetime(d.timestamp,unit='ms',utc=True)
        report.setdefault('response_ranges',{})[route]={'rows':len(d),'start':str(d.index.min()),'end':str(d.index.max()),
            'outside_requested_day':int(((d.timestamp<start)|(d.timestamp>end)).sum())}
        for rest_col,archive_col in fields.items():
            comparisons={}
            for offset in [-5,0,5]:
                reference=pd.to_numeric(d[rest_col]).to_numpy()
                observed=archive[archive_col].reindex(d.index+pd.Timedelta(minutes=offset)).to_numpy()
                valid=np.isfinite(reference)&np.isfinite(observed)&(reference!=0)
                relative=np.abs(observed[valid]-reference[valid])/np.abs(reference[valid])
                comparisons[str(offset)]={'pairs':int(valid.sum()),'median_relative_error':float(np.median(relative)) if len(relative) else None,
                                           'max_relative_error':float(relative.max()) if len(relative) else None,
                                           'within_1e_minus_5_fraction':float((relative<=1e-5).mean()) if len(relative) else None}
            report['fields'][archive_col]=comparisons
    (ROOT/'research/btc_metrics_rest_crosscheck.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
