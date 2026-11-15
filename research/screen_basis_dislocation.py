"""Economic screen only; NOT a backtest, no PnL or Sharpe claims."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def event_counts(series,threshold):
    eligible=series.dropna().loc[series.dropna()>=threshold]
    spaced=[]
    for stamp in eligible.index:
        if not spaced or stamp-spaced[-1]>=pd.Timedelta(hours=24): spaced.append(stamp)
    return {'candidate_hours':len(eligible),'day_separated_events':len(spaced),
            'first':str(eligible.index.min()) if len(eligible) else None,
            'last':str(eligible.index.max()) if len(eligible) else None}


def main():
    d=pd.read_csv(ROOT/'data/research_structural/btc_hourly_research.csv',index_col=0)
    d.index=pd.to_datetime(d.index,utc=True)
    basis=d.perp_spot_basis_bps.where(d.spot_volume.gt(0)&d.perp_volume.gt(0))
    report={'method':'Necessary cost feasibility screen only; no orders/funding simulation, no portfolio returns. Positive basis only; no spot borrowing.',
            'round_trip_bps':52,'base_buffer':1.25,'periods':{}}
    for name,start,end in [('development','2023-01-01','2025-01-01'),('holdout','2025-01-01',None),('full','2023-01-01',None)]:
        s=basis.loc[(basis.index>=start)&((basis.index<end) if end else True)]
        result={'max_positive_basis_bps':float(s.max()),'min_basis_bps':float(s.min()),
                'scenarios':{}}
        for key,threshold in [('base',65),('hurdle_0.8',52),('hurdle_1.2',78),('double_friction',130)]:
            result['scenarios'][key]={'threshold_bps':threshold,**event_counts(s,threshold)}
        report['periods'][name]=result
    report['enough_events_for_full_backtest']=all(x['day_separated_events']>=30 for x in report['periods']['holdout']['scenarios'].values())
    (ROOT/'research/basis_dislocation_screen.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
