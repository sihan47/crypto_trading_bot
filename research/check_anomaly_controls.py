"""Post-discovery confounding diagnostics, not an independent validation sample."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.stats import norm
from research.mine_anomalies import engineer,outcomes,REGIME,WINDOW,bh
ROOT=Path(__file__).resolve().parents[1]


def main():
    d=pd.read_csv(ROOT/'data/research_structural/btc_orthogonal_1h.csv',index_col=0)
    d.index=pd.to_datetime(d.index,utc=True);d=d.iloc[:int(len(d)*.7)]
    f,regime,ret=engineer(d)
    f.corr(method='spearman').to_csv(ROOT/'output/anomalies/feature_spearman.csv')
    controls=pd.DataFrame({'log_prior_vol':np.log(ret.pow(2).rolling(168).mean().shift(1)),
        'log_volume':np.log(d.futures_volume.where(d.futures_volume.gt(0))),
        'abs_current_return':ret.abs(),'past_24h_return':ret.rolling(24).sum(),
        'hour_sin':np.sin(2*np.pi*d.index.hour/24),'hour_cos':np.cos(2*np.pi*d.index.hour/24),
        'weekend':(d.index.dayofweek>=5).astype(float)},index=d.index)
    rows=[]
    choices=[('compression_x_flow_acceleration','high','vol_expansion'),
             ('signed_flow_low_impact','low','vol_expansion'),
             ('signed_flow_low_impact','low','lower_tail')]
    for name,side,target in choices:
        s=f[name];cut=s.rolling(REGIME,min_periods=WINDOW).quantile(.9 if side=='high' else .1).shift(1)
        event=(s.ge(cut) if side=='high' else s.le(cut)).astype(float).where(s.notna()&cut.notna())
        y=outcomes(ret,4)[target]
        for state in ['bull','bear']:
            valid=controls.notna().all(axis=1)&event.notna()&y.notna()&regime.eq(state)
            c=controls.loc[valid].to_numpy();c=(c-c.mean(axis=0))/c.std(axis=0)
            c=np.column_stack([np.ones(len(c)),c]);x=event[valid].to_numpy();yy=y[valid].to_numpy()
            xr=x-c@np.linalg.lstsq(c,x,rcond=None)[0]
            yr=yy-c@np.linalg.lstsq(c,yy,rcond=None)[0]
            coef=float(xr@yr/(xr@xr));residual=yr-coef*xr
            influence=np.zeros(len(d));influence[valid.to_numpy()]=xr*residual/(xr@xr)
            lag=168;ac=fftconvolve(influence,influence[::-1],mode='full')[len(d)-1:len(d)+lag]
            variance=ac[0]+2*np.dot(1-np.arange(1,len(ac))/(lag+1),ac[1:])
            t=coef/np.sqrt(variance)
            rows.append({'feature':name,'side':side,'target':target,'horizon_h':4,'regime':state,
                         'n':len(x),'adjusted_difference':coef,'t_hac':float(t),'p':float(2*norm.sf(abs(t)))})
    original=pd.read_csv(ROOT/'output/anomalies/all_tests.csv')
    combined=np.r_[original.p,[x['p'] for x in rows]]
    q=np.minimum(bh(combined)*np.sum(1/np.arange(1,len(combined)+1)),1)[-len(rows):]
    for row,value in zip(rows,q):row['q_by_expanded_family']=float(value)
    report={'selection':'Post-discovery diagnostics of 3 reported comparisons (2 feature families), same discovery sample; NOT independent confirmation.',
            'controls':list(controls.columns),'results':rows,'expanded_family_tests':len(combined)}
    (ROOT/'research/anomaly_control_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
