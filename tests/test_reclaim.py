import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from research.structural_reclaim import reclaim_signals, ReclaimParameters


class ReclaimTests(unittest.TestCase):
    def fixture(self):
        ix = pd.date_range('2025-01-01', periods=12, freq='h', tz='UTC')
        panel = pd.DataFrame({'create_time': ix-pd.Timedelta(minutes=65),
                              'spot_open': 100., 'spot_close': 97.,
                              'spot_low': 96., 'spot_tradeable': True}, index=ix)
        evidence = pd.DataFrame({'oi_z': 0., 'price_z': 0., 'initial_stop': 90.}, index=ix)
        evidence.loc[ix[2], ['oi_z', 'price_z']] = -3.
        panel.loc[ix[4], 'spot_close'] = 101.
        return panel, evidence

    def run_signal(self, panel, evidence, **kwargs):
        with patch('research.structural_reclaim.covering_signals', return_value=evidence.copy()):
            return reclaim_signals(panel, **kwargs)

    def test_wait_for_reclaim_and_prefix_causality(self):
        d, e = self.fixture()
        result = self.run_signal(d, e)
        self.assertEqual(list(result.index[result.signal]), [d.index[4]])
        self.assertEqual(result.initial_stop.iloc[4], 90.)
        pd.testing.assert_frame_equal(result.iloc[:5], self.run_signal(d.iloc[:5], e.iloc[:5]))

    def test_breach_before_availability_and_before_confirmation(self):
        for i in [2, 3]:
            d, e = self.fixture(); d.loc[d.index[i], 'spot_low'] = 89.
            self.assertFalse(self.run_signal(d, e).signal.any())

    def test_expiry_and_delayed_availability(self):
        d, e = self.fixture()
        self.assertFalse(self.run_signal(d, e, p=ReclaimParameters(recovery_hours=1)).signal.any())
        result = self.run_signal(d, e, extra_delay_hours=4)
        self.assertFalse(result.signal.any())

    def test_missing_price_disarms(self):
        d, e = self.fixture(); d.loc[d.index[3], 'spot_low'] = np.nan
        self.assertFalse(self.run_signal(d, e).signal.any())


if __name__ == '__main__': unittest.main()
