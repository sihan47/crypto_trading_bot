"""Statistical EDA only: no positions, execution rules, PnL or model selection."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.stats import norm, skew, kurtosis

ROOT=Path(__file__).resolve().parents[1]
WINDOW=24*30
REGIME=24*90
HORIZONS=(4,12,24)


def zscore(s):
    past=s.rolling(WINDOW,min_periods=WINDOW).agg(['mean','std']).shift(1)
    return (s-past['mean'])/past['std'].replace(0,np.nan)


def engineer(d):
    """All predictors/regime labels use data available at the timestamp only."""
    price=d.futures_close.where(d.futures_volume.gt(0))
    ret=np.log(price).diff()
    oi=np.log(d.sum_open_interest.where(d.sum_open_interest.gt(0)))
    top=np.log(d.sum_toptrader_long_short_ratio.where(d.sum_toptrader_long_short_ratio.gt(0)))
    account=np.log(d.count_toptrader_long_short_ratio.where(d.count_toptrader_long_short_ratio.gt(0)))
    global_ratio=np.log(d.count_long_short_ratio.where(d.count_long_short_ratio.gt(0)))
    flow=d.flow_imbalance
    gk=(.5*np.log(d.futures_high/d.futures_low)**2-(2*np.log(2)-1)*np.log(d.futures_close/d.futures_open)**2).clip(lower=0)
    compression=np.sqrt(gk.rolling(24).mean()/gk.rolling(168).mean().replace(0,np.nan))
    divergence=top-global_ratio
    f=pd.DataFrame(index=d.index)
    f['position_account_divergence']=zscore(divergence)
    f['account_account_divergence']=zscore(account-global_ratio)
    f['oi_weighted_funding']=zscore(d.funding_rate)*np.exp((oi-oi.rolling(WINDOW).mean().shift(1)).clip(-3,3))
    f['divergence_x_oi_change']=zscore(divergence)*zscore(oi.diff(24))
    f['compression_x_flow_acceleration']=flow.diff().diff()/compression.replace(0,np.nan)
    # Large signed flow with small contemporaneous price response. Not depth replenishment.
    response=ret/ret.rolling(168).std().shift(1).replace(0,np.nan)
    f['signed_flow_low_impact']=flow/(1+response.abs())
    regime=pd.Series(np.where(price>price.shift(REGIME),'bull','bear'),index=d.index)
    regime=regime.where(price.notna()&price.shift(REGIME).notna())
    return f.replace([np.inf,-np.inf],np.nan), regime, ret


def outcomes(ret,h):
    future=ret.shift(-1).iloc[::-1].rolling(h,min_periods=h).sum().iloc[::-1]
    variance=ret.pow(2).shift(-1).iloc[::-1].rolling(h,min_periods=h).sum().iloc[::-1]/h
    past=ret.pow(2).rolling(168,min_periods=168).mean().replace(0,np.nan)
    scale=np.sqrt(past*h)
    return pd.DataFrame({'return':future,
                         'vol_expansion':.5*np.log(variance/past),
                         'upper_tail':(future>2*scale).astype(float).where(future.notna()&scale.notna()),
                         'lower_tail':(future< -2*scale).astype(float).where(future.notna()&scale.notna())})


def contrast(y,event,valid,lag=168):
    """Conditional minus same-regime complement mean, calendar-time Newey-West SE.

    Influence scores retain zeros at non-regime/missing timestamps: no compression
    of time gaps. Bartlett HAC handles overlapping horizons and weekly dependence.
    """
    valid=valid & y.notna() & event.notna()
    a=valid & event.eq(True); b=valid & event.eq(False)
    na=int(a.sum()); nb=int(b.sum())
    if min(na,nb)<100: return None
    av=y[a].to_numpy(); bv=y[b].to_numpy()
    diff=float(av.mean()-bv.mean())
    influence=np.zeros(len(y))
    influence[a.to_numpy()]=(av-av.mean())/na
    influence[b.to_numpy()]=-(bv-bv.mean())/nb
    ac=fftconvolve(influence,influence[::-1],mode='full')[len(y)-1:len(y)+lag]
    variance=ac[0]+2*np.dot(1-np.arange(1,len(ac))/(lag+1),ac[1:])
    if variance<=0: return None
    t=diff/np.sqrt(variance)
    return {'n_event':na,'n_control':nb,'conditional_mean':float(av.mean()),'control_mean':float(bv.mean()),
            'difference':diff,'t_hac':float(t),'p':float(2*norm.sf(abs(t))),
            'conditional_skew':float(skew(av)) if np.std(av)>0 else None,
            'conditional_excess_kurtosis':float(kurtosis(av)) if np.std(av)>0 else None,
            'conditional_q05':float(np.quantile(av,.05)),'conditional_q95':float(np.quantile(av,.95))}


def bh(p):
    p=np.asarray(p); order=np.argsort(p); n=len(p)
    adjusted=np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1]
    out=np.empty(n);out[order]=np.minimum(adjusted,1);return out


def main():
    path=ROOT/'data/research_structural/btc_orthogonal_1h.csv'
    d=pd.read_csv(path,index_col=0);d.index=pd.to_datetime(d.index,utc=True)
    if d.index.has_duplicates or not d.index.equals(pd.date_range(d.index.min(),d.index.max(),freq='h')):
        raise ValueError('Complete unique hourly grid required; missing observations must be NaN rows')
    split=int(len(d)*.7); boundary=d.index[split]
    # The last 30% is never used for features, thresholds, labels or tests here.
    discovery=d.iloc[:split].copy()
    f,regime,ret=engineer(discovery)
    records=[]
    for name in f:
        s=f[name];quant=s.rolling(REGIME,min_periods=WINDOW).quantile(.1).shift(1)
        high=s.rolling(REGIME,min_periods=WINDOW).quantile(.9).shift(1)
        for side,cut in [('low',quant),('high',high)]:
            event=(s.le(cut) if side=='low' else s.ge(cut)).where(s.notna()&cut.notna())
            for h in HORIZONS:
                targets=outcomes(ret,h)
                for target in targets:
                    for state in ['bull','bear']:
                        result=contrast(targets[target],event,regime.eq(state))
                        if result:
                            stamps=d.index[:split][(event.eq(True)&regime.eq(state)&targets[target].notna()).to_numpy()]
                            independent=[]
                            for stamp in stamps:
                                if not independent or stamp-independent[-1]>=pd.Timedelta(hours=h): independent.append(stamp)
                            records.append({'feature':name,'side':side,'horizon_h':h,'target':target,'regime':state,
                                            'nonoverlap_event_count':len(independent),**result})
    tests=pd.DataFrame(records)
    tests['q_bh']=bh(tests.p)
    tests['q_by']=np.minimum(tests.q_bh*np.sum(1/np.arange(1,len(tests)+1)),1)
    ranked=[]
    for key,g in tests.groupby(['feature','side','horizon_h','target']):
        if set(g.regime)!= {'bull','bear'}: continue
        same_sign=g.difference.prod()>0
        raw=bool(same_sign and ((g.t_hac.abs()>2.5)|(g.p<.01)).all())
        qualified=bool(raw and (g.q_by<.01).all() and (g.nonoverlap_event_count>=30).all())
        row=dict(zip(['feature','side','horizon_h','target'],key))
        row.update({'raw_both_regimes':raw,'fdr_qualified':qualified,'worst_q':float(g.q_by.max()),
                    'min_abs_t':float(g.t_hac.abs().min()),'same_direction':bool(same_sign)})
        for _,item in g.iterrows():
            for col in ['difference','conditional_mean','t_hac','p','q_bh','q_by','n_event','nonoverlap_event_count']:
                row[item.regime+'_'+col]=item[col]
        ranked.append(row)
    ranked=pd.DataFrame(ranked).sort_values(['fdr_qualified','raw_both_regimes','worst_q','min_abs_t'],ascending=[False,False,True,False])
    out=ROOT/'output/anomalies';out.mkdir(parents=True,exist_ok=True)
    tests.to_csv(out/'all_tests.csv',index=False);ranked.to_csv(out/'ranked_anomalies.csv',index=False)
    f.to_csv(out/'discovery_features.csv')
    qualified=ranked.loc[ranked.fdr_qualified]
    report={'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'discovery_start':str(discovery.index.min()),'discovery_end':str(discovery.index.max()),
            'reserved_start':str(boundary),'reserved_rows':len(d)-split,'discovery_rows':split,
            'regime_counts':regime.value_counts().to_dict(),'feature_nonmissing':f.notna().sum().to_dict(),
            'tests':len(tests),'comparisons':len(ranked),'raw_both_regimes_count':int(ranked.raw_both_regimes.sum()),
            'fdr_qualified_count':len(qualified),'qualified':qualified.to_dict('records'),
            'ranked_top20':ranked.head(20).to_dict('records'),
            'method':'Conditional-minus-complement contrasts within trailing-90d bull/bear regimes; 168h Bartlett HAC normal-approximation p; BH and dependence-robust BY across all estimable features/tails/horizons/outcomes/regimes; qualification requires BY q<0.01 in both regimes and same direction. 10/90% rolling past-only thresholds; >=100 event/control hours and >=30 nonoverlapping event windows. Descriptive skew/kurtosis are not separately significance-tested. No trading rules or PnL.',
            'limitations':['Economic dimensions are distinct, not statistically independent.',
                           'Repeated historical research means reserved data is not globally pristine OOS.',
                           'BY handles arbitrary test dependence only if individual p-values are valid; HAC asymptotics can be inaccurate with sparse tails. Findings require replication.',
                           'Global accounts are not pure retail; position/account ratios have different denominators.',
                           'HAC asymptotic p-values and regime labels do not establish causation or trade profitability.']}
    (ROOT/'research/anomaly_findings.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['discovery_start','discovery_end','reserved_start','regime_counts','tests','raw_both_regimes_count','fdr_qualified_count']},indent=2))
    print(ranked.head(12).to_string(index=False))


if __name__=='__main__': main()
