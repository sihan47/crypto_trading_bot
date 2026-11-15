import unittest
import numpy as np
import pandas as pd
from research.structural_carry import CarryParameters,carry_features,carry_simulate,variants


class CarryTests(unittest.TestCase):
    def data(self,n=12):
        d=pd.DataFrame(index=pd.date_range('2024-01-01',periods=n,freq='1h',tz='UTC'))
        for prefix in ['s_','p_','m_']:
            for col in ['open','close']: d[prefix+col]=100.
            d[prefix+'high']=101.;d[prefix+'low']=99.
        d['s_volume']=d['p_volume']=100000.
        d['rate']=0.;d['pay']=0.;d['gap_pay']=0.
        return d

    def signal(self,d):
        return pd.DataFrame({'entry':[True]+[False]*(len(d)-1),'daily_funding':.001,'basis':0.},index=d.index)

    def test_four_leg_fees_and_self_financing_accounting(self):
        d=self.data();f=self.signal(d)
        eq,tr,cost=carry_simulate(d,f)
        self.assertEqual(len(tr),1)
        q=tr.qty.iloc[0]
        expected=q*(100*(1-.0005)*(1-.001)-100*(1+.0005)*(1+.001))
        expected+=q*(100*(1-.0005)*(1-.0006)-100*(1+.0005)*(1+.0006))
        self.assertAlmostEqual(eq.iloc[-1]-10000,expected)
        self.assertAlmostEqual(tr.net_pnl.sum(),eq.iloc[-1]-10000)
        self.assertEqual(cost['margin_breach_flags'],0)

    def test_funding_sign_boundary_and_delay(self):
        d=self.data();f=self.signal(d)
        d.loc[d.index[1],'pay']=1. # Entry-hour positive funding must not be collected.
        d.loc[d.index[2],'pay']=2.
        d.loc[d.index[3],'pay']=-.5
        _,tr,cost=carry_simulate(d,f)
        self.assertAlmostEqual(cost['funding'],tr.qty.iloc[0]*1.5)
        _,tr,cost=carry_simulate(d,f,leg_delay=1)
        self.assertAlmostEqual(cost['funding'],tr.qty.iloc[0]*-.5)

    def test_future_funding_never_changes_prior_forecasts(self):
        d=self.data(800);d.rate=.0001
        f=carry_features(d)
        changed=d.copy();changed.loc[changed.index[600]:,'rate']=.9
        pd.testing.assert_frame_equal(f.iloc[:600],carry_features(changed).iloc[:600])
        self.assertAlmostEqual(f.daily_funding.iloc[300],.0001*24)
        self.assertEqual(len(list(variants(CarryParameters()))),17)

    def test_legging_stress_exposes_directional_risk(self):
        d=self.data();f=self.signal(d)
        d.loc[d.index[2],'p_open']=90.;d.loc[d.index[2],'p_low']=89.
        normal,_,_=carry_simulate(d,f)
        delay,_,_=carry_simulate(d,f,leg_delay=1)
        self.assertLess(delay.iloc[-1],normal.iloc[-1])

    def test_intrahour_margin_breach_is_not_hidden(self):
        d=self.data();f=self.signal(d)
        d.loc[d.index[4],'m_high']=250.
        _,tr,cost=carry_simulate(d,f)
        self.assertGreater(cost['margin_breach_flags'],0)
        self.assertEqual(tr.iloc[0].reason,'possible_margin_breach')


if __name__=='__main__': unittest.main()
