import unittest
import numpy as np
import pandas as pd
from research.build_btc_derivatives_panel import clean_metrics,align_metrics
from research.fetch_btc_metrics import FIELDS


class PanelTests(unittest.TestCase):
    def raw(self):
        rows=[]
        for i in range(4):
            rows.append({'create_time':f'2026-01-01 00:{i*5:02d}:00','symbol':'BTCUSDT',
                         'source_day':'2026-01-01',**{k:float(i+1) for k in FIELDS}})
        return pd.DataFrame(rows)

    def test_conflicting_duplicates_quarantined_exact_duplicates_collapsed(self):
        d=self.raw()
        same=d.iloc[[0]].copy();conflict=d.iloc[[1]].copy();conflict[FIELDS[0]]=999.
        clean,a=clean_metrics(pd.concat([d,same,conflict]))
        self.assertEqual(a['exact_duplicates_removed'],1)
        self.assertEqual(a['conflicting_timestamps_quarantined'],1)
        self.assertEqual(len(clean),3)

    def test_invalid_oi_not_zero_signal_and_zero_taker_ratio_valid(self):
        d=self.raw();d.loc[0,'sum_open_interest']=0.;d.loc[0,'sum_taker_long_short_vol_ratio']=0.
        d.loc[1,'count_long_short_ratio']=float('inf')
        clean,_=clean_metrics(d)
        self.assertTrue(np.isnan(clean.sum_open_interest.iloc[0]))
        self.assertEqual(clean.sum_taker_long_short_vol_ratio.iloc[0],0.)
        self.assertTrue(np.isnan(clean.count_long_short_ratio.iloc[1]))

    def test_no_future_or_stale_metrics(self):
        clean,_=clean_metrics(self.raw())
        times=pd.to_datetime(['2026-01-01 01:00Z','2026-01-01 01:05Z','2026-01-01 01:30Z','2026-01-01 01:31Z'])
        out=align_metrics(pd.DataFrame({'decision_time':times}),clean)
        self.assertTrue(pd.isna(out.sum_open_interest.iloc[0]))
        self.assertEqual(out.sum_open_interest.iloc[1],1.)
        self.assertEqual(out.sum_open_interest.iloc[2],4.)
        self.assertTrue(pd.isna(out.sum_open_interest.iloc[3]))
        valid=out.dropna(subset='metric_available_at')
        self.assertTrue((valid.metric_available_at<=valid.decision_time).all())


if __name__=='__main__': unittest.main()
