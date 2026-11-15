import unittest
import pandas as pd
import numpy as np
from research.forum_breakout import hourly_features,simulate_forum
from research.rotation_breakout import simulate as simulate_rotation


class ForumTests(unittest.TestCase):
    def prices(self,n=500):
        rng=np.random.default_rng(12)
        c=100*np.exp(np.cumsum(rng.normal(0,.005,n)))
        return pd.DataFrame({'open':c,'high':c*1.002,'low':c*.998,'close':c,'volume':1.},
                            index=pd.date_range('2023-01-01',periods=n,freq='15min',tz='UTC'))

    def test_hourly_causality_and_available_time(self):
        df=self.prices()
        for mode in ['session','rolling24']:
            a=hourly_features(df,mode)
            b=hourly_features(df.iloc[:403],mode)
            pd.testing.assert_frame_equal(a.loc[b.index],b)
            self.assertEqual(a.index[0],df.index[0]+pd.Timedelta(hours=1))

    def test_stop_first_when_both_hit(self):
        df=self.prices(8)
        df[['open','close']]=100.;df['high']=101.;df['low']=99.
        df.iloc[4,df.columns.get_loc('high')]=105.
        df.iloc[4,df.columns.get_loc('low')]=95.
        f=pd.DataFrame({'combined':[True],'exit':[False]},index=df.index[[4]])
        r=simulate_forum(df,f,fee=0,slippage=0)
        self.assertAlmostEqual(r['return_pct'],-1.5)
        self.assertEqual(r['ambiguous_bars_stop_first'],1)

    def test_gap_stop_uses_worse_open(self):
        df=self.prices(8)
        df[['open','close']]=100.;df['high']=101.;df['low']=99.
        df.iloc[5,df.columns.get_loc('open')]=90.
        f=pd.DataFrame({'combined':[True],'exit':[False]},index=df.index[[4]])
        r=simulate_forum(df,f,fee=0,slippage=0)
        self.assertAlmostEqual(r['return_pct'],-10.)
        self.assertEqual(r['exit_reasons']['gap_stop'],1)

    def test_rotation_uses_previous_day_ranking_and_next_open(self):
        idx=pd.date_range('2023-01-01',periods=4,freq='1D',tz='UTC')
        a=pd.DataFrame({'open':[50.,100.,120.,120.],'close':[90.,110.,120.,120.],
                        'entry':[True,False,False,False],'exit':[False,True,False,False],
                        'momentum':[.2,0.,0.,0.]},index=idx)
        b=a.copy();b['momentum']=[.1,99.,99.,99.]
        r=simulate_rotation({'A':a,'B':b},fee=0,slip=0)
        self.assertAlmostEqual(r['return_pct'],20.)
        self.assertEqual(r['entry_counts'],{'A':1,'B':0})


if __name__=='__main__':
    unittest.main()
