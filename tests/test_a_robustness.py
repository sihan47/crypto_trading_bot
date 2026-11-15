import unittest
import pandas as pd
from research.impulse_scan import discovery
from research.a_robustness import a_features
from research.archetype_tournament import signals


class ARobustnessTests(unittest.TestCase):
    def test_original_equivalence_and_oi_sign_cancellation(self):
        d=discovery().iloc[:6000]
        pd.testing.assert_series_equal(a_features(d).signal,signals(d)['A'].signal)
        changed=d.copy();changed['sum_open_interest']*=3
        pd.testing.assert_series_equal(a_features(d).signal,a_features(changed).signal)

    def test_prefix_and_delayed_signal(self):
        d=discovery().iloc[:6000]
        for kwargs in [{},{'data_delay':2},{'signal_delay':2},{'normalization':576,'macro':1728,'risk_window':19}]:
            pd.testing.assert_frame_equal(a_features(d,**kwargs).iloc[:5500],a_features(d.iloc[:5500],**kwargs))
        base=a_features(d);delayed=a_features(d,signal_delay=2)
        expected=base.signal.shift(2,fill_value=False)&delayed.sigma.notna()
        pd.testing.assert_series_equal(expected.rename('signal'),delayed.signal)


if __name__=='__main__':unittest.main()
