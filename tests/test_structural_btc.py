import unittest
import numpy as np
import pandas as pd
from research.structural_btc import Parameters,features,simulate,metrics,sensitivity
from research.structural_recovery import RecoveryParameters,recovery_features


class StructuralTests(unittest.TestCase):
    def bars(self,n=3000):
        c=100+np.arange(n)*.001+np.sin(np.arange(n)/7)*.2
        return pd.DataFrame({'open':c,'high':c+.3,'low':c-.3,'close':c,'volume':1000.},
                            index=pd.date_range('2024-01-01',periods=n,freq='15min',tz='UTC'))

    def evidence(self,d):
        f=pd.DataFrame({'signal':[True]+[False]*(len(d)-1),'sigma_day':.1,'initial_stop':95.,
                        'vol_regime':'low','prior_day_direction':'up','session':'test','weekend':False},index=d.index)
        return f

    def test_prefix_and_future_mutation(self):
        d=self.bars();f=features(d)
        short=features(d.iloc[:2000])
        pd.testing.assert_frame_equal(short,f.loc[short.index])
        changed=d.copy();changed.iloc[2000:,:4]*=4
        pd.testing.assert_frame_equal(features(changed).loc[short.index],short)

    def test_partial_hour_cannot_signal(self):
        f=features(self.bars(2002))
        self.assertFalse(f.signal.iloc[-1])

    def test_entry_next_bar_costs_gap_and_accounting(self):
        d=self.bars(4);d[['open','close']]=100.;d.high=101.;d.low=99.
        d.iloc[2]=[90.,91.,89.,90.,1000.]
        f=self.evidence(d)
        eq,tr,cost=simulate(d,f,Parameters(trail=100.,risk=.005))
        self.assertEqual(tr.iloc[0].entry_time,str(d.index[1]))
        self.assertEqual(tr.iloc[0].reason,'gap_stop')
        self.assertAlmostEqual(tr.iloc[0].exit_price,90*(1-.0005))
        self.assertGreater(cost['fees'],0)
        self.assertGreater(cost['slippage_estimate'],0)
        self.assertEqual(cost['funding'],0)
        self.assertAlmostEqual(eq.iloc[-1]-10000,tr.net_pnl.sum())

    def test_entry_bar_stop_and_no_same_bar_high_ratchet(self):
        d=self.bars(3);d[['open','close']]=100.;d.high=200.;d.low=99.
        f=self.evidence(d)
        _,tr,_=simulate(d,f)
        self.assertEqual(tr.iloc[0].reason,'end_of_sample')
        d.iloc[1,d.columns.get_loc('low')]=94.
        _,tr,_=simulate(d,f)
        self.assertEqual(tr.iloc[0].reason,'intrabar_stop')
        self.assertAlmostEqual(tr.iloc[0].exit_price,95*(1-.0005))

    def test_missing_bar_no_entry_and_existing_position_liquidates(self):
        d=self.bars(5);d[['open','close']]=100.;d.high=101.;d.low=99.
        f=self.evidence(d)
        gap=d.drop(d.index[2])
        _,tr,cost=simulate(gap,f)
        self.assertEqual(tr.iloc[0].reason,'data_gap')
        self.assertEqual(cost['gap_exits'],1)
        _,tr,_=simulate(d.drop(d.index[1]),f)
        self.assertEqual(len(tr),0)

    def test_sensitivity_covers_every_parameter_and_corners(self):
        variants=list(sensitivity(Parameters()))
        self.assertEqual(len(variants),21)
        self.assertEqual(variants[-1][0],'all_1.2')

    def test_recovery_signal_uses_shock_volume_and_reclaim(self):
        d=self.bars();d[['open','close']]=100.;d.high=100.1;d.low=99.9;d.volume=10.
        d.iloc[-4:]=[100.,100.,94.,98.,1000.]
        f=recovery_features(d)
        self.assertTrue(f.signal.iloc[-1])
        self.assertEqual(f.initial_stop.iloc[-1],94.)
        self.assertNotIn('compression',f.columns)
        self.assertEqual(len(list(sensitivity(RecoveryParameters()))),21)
        short=recovery_features(d.iloc[:2000])
        pd.testing.assert_frame_equal(short,f.loc[short.index])

    def test_zero_cost_and_friction_accounting_agree_with_manual_round_trip(self):
        d=self.bars(3);d[['open','close']]=100.;d.high=101.;d.low=99.
        f=self.evidence(d)
        eq,tr,cost=simulate(d,f)
        qty=tr.qty.iloc[0]
        expected=qty*(100*(1-.0005)*(1-.001)-100*(1+.0005)*(1+.001))
        self.assertAlmostEqual(eq.iloc[-1]-10000,expected)
        clean,_,_=simulate(d,f,fee=0,slip=0)
        self.assertAlmostEqual(clean.iloc[-1],10000.)


if __name__=='__main__': unittest.main()
