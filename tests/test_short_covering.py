import unittest
import numpy as np
import pandas as pd
from research.structural_short_covering import CoverParameters,covering_signals


class CoverTests(unittest.TestCase):
    def panel(self):
        idx=pd.date_range('2025-01-01',periods=800,freq='1h',tz='UTC')
        return pd.DataFrame({'create_time':idx-pd.Timedelta(minutes=65),'metric_available_at':idx,
            'spot_close':100+np.sin(np.arange(800)/7),'spot_low':98.,'spot_tradeable':True,
            'sum_open_interest':10000+100*np.sin(np.arange(800)/5),'sum_taker_long_short_vol_ratio':1.},index=idx)

    def test_three_conditions_and_invalid_data(self):
        d=self.panel();d.loc[d.index[400],'sum_open_interest']*=.8
        d.loc[d.index[399],'spot_close']*=1.2
        d.loc[d.index[400],'sum_taker_long_short_vol_ratio']=3.
        self.assertTrue(covering_signals(d).signal.iloc[400])
        d.loc[d.index[400],'sum_taker_long_short_vol_ratio']=np.nan
        self.assertFalse(covering_signals(d).signal.iloc[400])
        d.loc[d.index[400],'sum_taker_long_short_vol_ratio']=3.
        d.loc[d.index[400],'sum_open_interest']=0.
        self.assertFalse(covering_signals(d).signal.iloc[400])

    def test_prefix_and_future_availability(self):
        d=self.panel();f=covering_signals(d)
        pd.testing.assert_frame_equal(f.iloc[:600],covering_signals(d.iloc[:600]))
        d['metric_available_at']+=pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError): covering_signals(d)


if __name__=='__main__': unittest.main()
