import unittest
import numpy as np
import pandas as pd
from research.candidate_rules import candidate_signals
from backtesting.next_bar import simulate


class ResearchCandidateTests(unittest.TestCase):
    def prices(self,n=700):
        rng=np.random.default_rng(42)
        close=100*np.exp(np.cumsum(rng.normal(.001,.02,n)))
        return pd.DataFrame({'open':close,'close':close,'high':close*1.01,
                             'low':close*.99,'volume':1.},
                            index=pd.date_range('2023-01-01',periods=n,freq='4h',tz='UTC'))

    def test_all_rules_are_causal(self):
        df=self.prices()
        full=candidate_signals(df,6)
        prefix=candidate_signals(df.iloc[:650],6)
        for name,(a,b) in full.items():
            pd.testing.assert_series_equal(a.iloc[:650],prefix[name][0])
            pd.testing.assert_series_equal(b.iloc[:650],prefix[name][1])

    def test_no_momentum_before_lookback(self):
        df=self.prices()
        a,b=candidate_signals(df,6)['momentum84']
        self.assertFalse(a.iloc[:504].any())
        self.assertFalse(b.iloc[:504].any())

    def test_standard_macd_buys_only_bullish_cross(self):
        df=self.prices()
        a,b=candidate_signals(df,6)['macd_standard']
        c=df.close
        line=c.ewm(span=12,adjust=False,min_periods=12).mean()-c.ewm(span=26,adjust=False,min_periods=26).mean()
        hist=line-line.ewm(span=9,adjust=False,min_periods=9).mean()
        self.assertTrue(a.any() and b.any())
        self.assertTrue(((hist[a]>0)&(hist.shift()[a]<=0)).all())
        self.assertTrue(((hist[b]<0)&(hist.shift()[b]>=0)).all())

    def test_simulator_next_open_costs_hand_calculation(self):
        df=self.prices(4)
        df['open']=[50.,100.,120.,200.]
        df['close']=[90.,110.,130.,210.]
        a=pd.Series([True,False,False,False],index=df.index)
        b=pd.Series([False,True,False,False],index=df.index)
        result=simulate(df,a,b,fee=.001,slippage=.0005)
        expected=(120*(1-.0005)*(1-.001)/(100*(1+.0005)*(1+.001))-1)*100
        self.assertAlmostEqual(result['return_pct'],expected)
        self.assertEqual(result['round_trips'],1)


if __name__=='__main__':
    unittest.main()
