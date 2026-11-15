import unittest
import numpy as np
import pandas as pd
from research.archetype_tournament import simulate,Spec,signals
from research.impulse_scan import discovery


class TournamentTests(unittest.TestCase):
    def fixture(self):
        ix=pd.date_range('2024-01-01',periods=12,freq='15min',tz='UTC')
        raw=pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.,'volume':1000.},index=ix)
        f=pd.DataFrame({'signal':[True,False,False],'stop':95.,'limit':99.,'target':101.,'sigma':.02,
                        'regime':'bull','vol_regime':'low'},index=ix[[0,4,8]])
        return raw,f

    def test_next_bar_market_and_cash_accounting(self):
        raw,f=self.fixture();eq,tr,_=simulate(raw,f,'C')
        self.assertEqual(tr.entry_time.iloc[0],str(raw.index[1]))
        self.assertAlmostEqual(eq.iloc[-1]-10000,tr.net_pnl.sum())
        self.assertLess(tr.net_pnl.iloc[0],0)

    def test_touch_does_not_fill_limit(self):
        raw,f=self.fixture();eq,tr,c=simulate(raw,f,'B')
        self.assertEqual(c['orders'],1);self.assertEqual(c['fills'],0)
        self.assertEqual(eq.iloc[-1],10000)

    def test_limit_stop_first_and_no_optimistic_target(self):
        raw,f=self.fixture();raw.loc[raw.index[1],'low']=94.
        eq,tr,c=simulate(raw,f,'B')
        self.assertEqual(c['fills'],1)
        self.assertEqual(tr.reason.iloc[0],'intrabar_stop')
        self.assertLess(tr.net_pnl.iloc[0],0)
        self.assertAlmostEqual(eq.iloc[-1]-10000,tr.net_pnl.sum())

    def test_signal_prefix_causality(self):
        d=discovery().iloc[:5000]
        full=signals(d);prefix=signals(d.iloc[:4500])
        for key in full:pd.testing.assert_frame_equal(full[key].iloc[:4500],prefix[key])


if __name__=='__main__':unittest.main()
