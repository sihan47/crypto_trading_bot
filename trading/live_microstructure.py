"""Public Binance microstructure context; closed-hour, causal publication buffers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import io
import json
import re
import zipfile
import numpy as np
import pandas as pd
import requests
from trading.macro_calendar import utc

ROOT=Path(__file__).resolve().parents[1]
BASE='https://fapi.binance.com'
FIELDS=['sum_open_interest','sum_open_interest_value','sum_toptrader_long_short_ratio',
        'count_toptrader_long_short_ratio','count_long_short_ratio']
ROUTES={'openInterestHist':{'sumOpenInterest':'sum_open_interest','sumOpenInterestValue':'sum_open_interest_value'},
        'topLongShortPositionRatio':{'longShortRatio':'sum_toptrader_long_short_ratio'},
        'topLongShortAccountRatio':{'longShortRatio':'count_toptrader_long_short_ratio'},
        'globalLongShortAccountRatio':{'longShortRatio':'count_long_short_ratio'}}


def api(path,params):
    response=requests.get(BASE+path,params=params,timeout=(5,12))
    response.raise_for_status();result=response.json()
    if not isinstance(result,list):raise ValueError('Expected Binance list')
    return result


def archive(symbol,day,cache):
    name=f'{symbol}-metrics-{day}.zip';path=cache/name
    # Reuse previously verified local archives without requiring them on VPS.
    old=ROOT/'data/research_structural/metrics_archives'/name
    if old.exists():path=old
    check=path.with_suffix('.zip.CHECKSUM')
    url=f'https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{name}'
    try:
        if path.exists() and check.exists():body=path.read_bytes();expected=check.read_text().split()[0]
        else:
            r=requests.get(url,timeout=(4,6));r.raise_for_status()
            c=requests.get(url+'.CHECKSUM',timeout=(4,6));c.raise_for_status()
            body=r.content;expected=c.text.split()[0]
        if hashlib.sha256(body).hexdigest()!=expected:raise ValueError('Archive checksum mismatch')
        if not path.exists():path.write_bytes(body);check.write_text(expected,encoding='ascii')
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            names=[x for x in z.namelist() if x.endswith('.csv')]
            if len(names)!=1:raise ValueError('Unexpected archive members')
            d=pd.read_csv(z.open(names[0]),usecols=['create_time','symbol']+FIELDS)
        if not d.symbol.eq(symbol).all():raise ValueError('Wrong archive symbol')
        d['create_time']=pd.to_datetime(d.create_time,utc=True,errors='raise')
        return d,None
    except Exception as exc:return None,{'day':day,'error_type':type(exc).__name__}


def features(d):
    def z(s):
        return (s-s.rolling(720).mean().shift(1))/s.rolling(720).std().shift(1).replace(0,np.nan)
    oi=np.log(d.sum_open_interest.where(d.sum_open_interest>0))
    divergence=np.log(d.sum_toptrader_long_short_ratio.where(d.sum_toptrader_long_short_ratio>0))-np.log(d.count_long_short_ratio.where(d.count_long_short_ratio>0))
    ret=np.log(d.close).diff()
    gk=(.5*np.log(d.high/d.low)**2-(2*np.log(2)-1)*np.log(d.close/d.open)**2).clip(lower=0)
    compression=np.sqrt(gk.rolling(24).mean()/gk.rolling(168).mean().replace(0,np.nan))
    flow=(2*d.taker_buy_base_asset_volume-d.volume)/d.volume.replace(0,np.nan)
    f=pd.DataFrame(index=d.index)
    f['squeeze_anomaly']=z(d.funding_rate)*np.exp((oi-oi.rolling(720).mean().shift(1)).clip(-3,3))
    f['smart_money_floor']=z(divergence)
    f['compression_x_flow_acceleration']=flow.diff().diff()/compression.replace(0,np.nan)
    response=ret/ret.rolling(168).std().shift(1).replace(0,np.nan)
    f['signed_flow_low_impact']=flow/(1+response.abs())
    return f.replace([np.inf,-np.inf],np.nan)


def build_panel(symbol,now,cache):
    start=now.floor('D')-pd.Timedelta(days=130)
    end=now.floor('h');cursor=int(start.timestamp()*1000);end_ms=int(end.timestamp()*1000)-1
    prices=[]
    while cursor<=end_ms:
        rows=api('/fapi/v1/klines',{'symbol':symbol,'interval':'1h','startTime':cursor,'endTime':end_ms,'limit':1500})
        if not rows:break
        prices.extend(rows)
        next_cursor=int(rows[-1][0])+3600000
        if next_cursor<=cursor:raise ValueError('Kline pagination did not advance')
        cursor=next_cursor
    columns=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base_asset_volume','taker_buy_quote_volume','ignore']
    p=pd.DataFrame(prices,columns=columns)
    if p.empty:raise ValueError('No price history')
    for col in columns:p[col]=pd.to_numeric(p[col],errors='coerce')
    p.index=pd.to_datetime(p.open_time,unit='ms',utc=True)+pd.Timedelta(hours=1)
    if p.index.has_duplicates:raise ValueError('Duplicate prices')
    grid=pd.date_range(start+pd.Timedelta(hours=1),end,freq='h')
    p=p.reindex(grid)
    bad=~p[['open','high','low','close','volume','taker_buy_base_asset_volume']].apply(np.isfinite).all(axis=1)|p.volume.le(0)|~p.taker_buy_base_asset_volume.between(0,p.volume)
    bad |= p[['open','high','low','close']].le(0).any(axis=1) | p.high.lt(p[['open','close','low']].max(axis=1)) | p.low.gt(p[['open','close','high']].min(axis=1))
    p.loc[bad,:]=np.nan
    days=pd.date_range(start,now.floor('D')-pd.Timedelta(days=2),freq='D')
    frames=[];errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for frame,error in pool.map(lambda day:archive(symbol,day.strftime('%Y-%m-%d'),cache),days):
            if frame is not None:frames.append(frame)
            if error:errors.append(error)
    recent=[]
    for route,mapping in ROUTES.items():
        rows=api('/futures/data/'+route,{'symbol':symbol,'period':'1h','startTime':int((now-pd.Timedelta(days=3)).timestamp()*1000),'limit':100})
        frame=pd.DataFrame(rows)
        if frame.empty:raise ValueError('Missing recent positioning data')
        frame.index=pd.to_datetime(frame.timestamp,unit='ms',utc=True)-pd.Timedelta(minutes=5)
        if frame.index.has_duplicates:raise ValueError('Duplicate recent snapshots')
        recent.append(frame[list(mapping)].rename(columns=mapping))
    rest=pd.concat(recent,axis=1);rest.index.name='create_time';rest=rest.reset_index()
    historic=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['create_time']+FIELDS)
    # Official archive takes precedence for overlapping periods; REST fills recent gap.
    metrics=pd.concat([historic,rest],ignore_index=True).drop_duplicates('create_time',keep='first').sort_values('create_time')
    for col in FIELDS:
        metrics[col]=pd.to_numeric(metrics[col],errors='coerce')
        metrics.loc[~np.isfinite(metrics[col])|metrics[col].le(0),col]=np.nan
    metrics['available_at']=metrics.create_time+pd.Timedelta(minutes=65)
    p.index.name='decision_time'
    joined=pd.merge_asof(p.reset_index(),metrics[['create_time','available_at']+FIELDS],left_on='decision_time',right_on='available_at',direction='backward',tolerance=pd.Timedelta(minutes=10))
    funding=api('/fapi/v1/fundingRate',{'symbol':symbol,'startTime':int((start-pd.Timedelta(days=1)).timestamp()*1000),'endTime':end_ms,'limit':1000})
    if len(funding)>=1000:raise ValueError('Funding history needs pagination; refuse truncated sample')
    fund=pd.DataFrame(funding)
    fund['funding_time']=pd.to_datetime(fund.fundingTime,unit='ms',utc=True)
    fund['funding_available_at']=fund.funding_time+pd.Timedelta(hours=1)
    fund['funding_rate']=pd.to_numeric(fund.fundingRate,errors='coerce')
    joined=pd.merge_asof(joined,fund[['funding_time','funding_available_at','funding_rate']].sort_values('funding_available_at'),
                         left_on='decision_time',right_on='funding_available_at',direction='backward',tolerance=pd.Timedelta(hours=12))
    return joined.set_index('decision_time'),errors


def summarize(panel,symbol,now,errors):
    if panel.empty:raise ValueError('Empty panel')
    last=panel.iloc[-1];age=(now-panel.index[-1]).total_seconds()
    if not 0<=age<=5400:raise ValueError('Stale closed-hour prices')
    if pd.isna(last.create_time) or not 0<=(now-last.create_time).total_seconds()<=3*3600:raise ValueError('Stale positioning')
    if pd.isna(last.funding_time) or not 0<=(now-last.funding_time).total_seconds()<=13*3600:raise ValueError('Stale funding')
    f=features(panel);items={}
    for name in f:
        value=f[name].iloc[-1];history=f[name].iloc[-2161:-1].dropna()
        valid=np.isfinite(value) and len(history)>=720
        items[name]={'value':float(value) if np.isfinite(value) else None,
                     'percentile':float((history.le(value).sum()/len(history))*100) if valid else None,
                     'reference_n':len(history),'reference_start':str(history.index.min()) if len(history) else None,
                     'reference_end':str(history.index.max()) if len(history) else None}
    macro=None
    if len(panel)>2160 and np.isfinite(panel.close.iloc[-2161]) and np.isfinite(last.close):
        macro='bull' if last.close>panel.close.iloc[-2161] else 'bear'
    complete=all(x['percentile'] is not None for x in items.values()) and macro is not None
    delta=f.smart_money_floor.iloc[-1]-f.smart_money_floor.iloc[-5]
    return {'status':'available' if complete else 'warming_up','symbol':symbol,'as_of':now.isoformat(),
            'price_close_time':str(panel.index[-1]),'positioning_source_time':str(last.create_time),
            'actual_settled_funding_rate':float(last.funding_rate) if np.isfinite(last.funding_rate) else None,
            'funding_settlement_time':str(last.funding_time),'macro_trend_regime':macro,
            'positioning_snapshot':{name:float(last[name]) if np.isfinite(last[name]) else None for name in FIELDS},
            'smart_money_divergence_change_4h':float(delta) if np.isfinite(delta) else None,
            **items,'volatility_damping':{'components':['compression_x_flow_acceleration','signed_flow_low_impact'],
                'extreme':any(x['percentile'] is not None and (x['percentile']<10 or x['percentile']>90) for k,x in items.items() if k in ['compression_x_flow_acceleration','signed_flow_low_impact'])},
            'priors':{'validation':'Exploratory, NOT validated trading alpha. BTC-specific findings do not transfer automatically to BNB.',
                      'applicable_asset':symbol=='BTCUSDT','squeeze':'Bull-only historical association; funding z<0 is not necessarily negative actual funding.',
                      'positioning':'Top POSITION vs GLOBAL ACCOUNT ratio; not identified smart money versus pure retail.',
                      'damping':'Observed contraction association; confounding controls weakened significance; not a guarantee of mean reversion.'},
            'archive_failures':errors,'sources':[BASE+'/fapi/v1/klines',BASE+'/fapi/v1/fundingRate',
                *[BASE+'/futures/data/'+x for x in ROUTES],f'https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/'],
            'availability_note':'Closed-hour prices; metrics label+65m and funding settlement+1h. Conservative assumptions, not verified publication logs. REST labels mapped back 5m to archive convention.'}


def fetch_telemetry(symbol='BTCUSDT',now=None,cache_dir=None):
    now=utc(now)
    if not re.fullmatch('[A-Z0-9]{3,20}USDT',symbol):raise ValueError('Invalid symbol')
    cache=Path(cache_dir) if cache_dir else ROOT/'logs/context_microstructure'/symbol
    cache.mkdir(parents=True,exist_ok=True);snapshot=cache/'snapshot.json'
    if snapshot.exists():
        try:
            saved=json.loads(snapshot.read_text(encoding='utf-8'))
            if 0<=(now-utc(saved['as_of'])).total_seconds()<300:return saved
        except (ValueError,KeyError):pass
    try:
        panel,errors=build_panel(symbol,now,cache)
        result=summarize(panel,symbol,now,errors)
        panel.to_csv(cache/'latest_panel.csv')
    except Exception as exc:result={'status':'unavailable','symbol':symbol,'as_of':now.isoformat(),'error_type':type(exc).__name__}
    temp=snapshot.with_suffix('.tmp');temp.write_text(json.dumps(result,allow_nan=False),encoding='utf-8');temp.replace(snapshot)
    return result
