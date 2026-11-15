import unittest
import numpy as np
import pandas as pd
from research.structural_deleveraging import signals_from_panel,DeleveragingParameters,deleveraging_features


class DeleveragingTests(unittest.TestCase):
    def panel(self,n=800):
        idx=pd.date_range('2025-01-01',periods=n,freq='1h',tz='UTC')
        d=pd.DataFrame(index=idx)
        d['create_time']=idx-pd.Timedelta(minutes=65)
        d['metric_available_at']=idx
        d['spot_close']=100+np.sin(np.arange(n)/7)
        d['spot_low']=d.spot_close-1
        d['spot_tradeable']=True
        d['sum_open_interest']=10000+100*np.sin(np.arange(n)/5)
        d['count_long_short_ratio']=1.5+.1*np.sin(np.arange(n)/3)
        return d

    def test_prefix_causality_and_event_price_alignment(self):
        d=self.panel();p=DeleveragingParameters()
        f=signals_from_panel(d,p)
        short=signals_from_panel(d.iloc[:600],p)
        pd.testing.assert_frame_equal(short,f.iloc[:600])
        returns=np.log(d.spot_close).diff()
        expected=(returns/returns.rolling(p.week).std().shift(1)).iloc[399]
        self.assertAlmostEqual(f.price_z.iloc[400],expected)
        changed=d.copy();changed.iloc[600:,changed.columns.get_loc('sum_open_interest')]*=5
        pd.testing.assert_frame_equal(signals_from_panel(changed,p).iloc[:600],short)

    def test_actual_three_conditions_trigger_and_zero_oi_is_not_accepted(self):
        d=self.panel();p=DeleveragingParameters()
        d.loc[d.index[400],'sum_open_interest']*=.8
        d.loc[d.index[399],'spot_close']*=.8
        d.loc[d.index[399],'spot_low']=70.
        d.loc[d.index[399],'count_long_short_ratio']=10.
        self.assertTrue(signals_from_panel(d,p).signal.iloc[400])
        d.loc[d.index[400],'sum_open_interest']=0.
        self.assertFalse(signals_from_panel(d,p).signal.iloc[400])

    def test_future_available_metric_rejected(self):
        d=self.panel();d['metric_available_at']+=pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError,'Future'):
            signals_from_panel(d,DeleveragingParameters())


if __name__=='__main__': unittest.main()
