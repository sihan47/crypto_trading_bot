import unittest
import numpy as np
import pandas as pd
from research.mine_anomalies import outcomes, engineer, bh, contrast


class MiningTests(unittest.TestCase):
    def test_forward_labels_and_missing_horizon(self):
        r=pd.Series(np.arange(220)/10000.)
        y=outcomes(r,4)
        self.assertAlmostEqual(y['return'].iloc[180],r.iloc[181:185].sum())
        self.assertTrue(y['return'].iloc[-4:].isna().all())
        r.iloc[182]=np.nan
        self.assertTrue(np.isnan(outcomes(r,4)['return'].iloc[180]))

    def test_bh_order_and_bounds(self):
        np.testing.assert_allclose(bh([.03,.001,.5]),[.045,.003,.5])

    def test_contrast_is_relative_to_control(self):
        rng=np.random.default_rng(42); n=5000
        e=pd.Series(np.arange(n)%2==0)
        y=pd.Series(rng.normal(size=n))+e.astype(float)*2
        result=contrast(y,e,pd.Series(True,index=y.index))
        self.assertGreater(result['difference'],1.8)
        self.assertLess(result['p'],.01)

    def test_features_prefix_causality(self):
        ix=pd.date_range('2023-01-01',periods=3000,freq='h',tz='UTC')
        x=np.arange(len(ix));price=100+np.sin(x/10)+x*.01
        d=pd.DataFrame({'futures_close':price,'futures_open':price-.1,'futures_high':price+1,
                        'futures_low':price-1,'futures_volume':100.,'sum_open_interest':1000+x,
                        'sum_toptrader_long_short_ratio':1.2+np.sin(x/30)*.1,
                        'count_toptrader_long_short_ratio':1.1+np.sin(x/20)*.1,
                        'count_long_short_ratio':1.3+np.sin(x/40)*.1,
                        'flow_imbalance':np.sin(x),'funding_rate':np.sin(x/100)*.0001},index=ix)
        full=engineer(d);prefix=engineer(d.iloc[:2500])
        for a,b in zip(full,prefix):
            if isinstance(a,pd.DataFrame): pd.testing.assert_frame_equal(a.iloc[:2500],b)
            else: pd.testing.assert_series_equal(a.iloc[:2500],b)


if __name__=='__main__': unittest.main()
