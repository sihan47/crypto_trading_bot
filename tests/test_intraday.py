import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from strategies.intraday_strategy import (IntradayParams,completed_trend,intraday_features,
                                         run_intraday_strategy,signals_from_features,intraday_decision)


class IntradayTests(unittest.TestCase):
    def prices(self,n=1500):
        rng=np.random.default_rng(47)
        close=100*np.exp(np.cumsum(rng.normal(.0003,.004,n)))
        return pd.DataFrame({'open':close,'close':close,'high':close*1.005,'low':close*.995,'volume':100.},
                            index=pd.date_range('2025-01-01',periods=n,freq='15min',tz='UTC'))

    def features(self,n=12):
        return pd.DataFrame({'open':100.,'close':100.,'atr':1.,'basis':105.,'bull_4h':True,
                             'ready':True,'rsi2':30.,'gap':False,'pullback_setup':False,'squeeze_setup':False,
                             'entry_blocked':False},index=pd.date_range('2025-01-01',periods=n,freq='15min',tz='UTC'))

    def test_higher_bar_only_available_on_its_last_signal_bar(self):
        df=self.prices(64)
        p=IntradayParams(trend_fast=2,trend_slow=3)
        result=completed_trend(df,p)
        self.assertTrue(pd.isna(result.iloc[14]['close_4h']))
        self.assertEqual(result.iloc[15]['close_4h'],df.iloc[15]['close'])
        self.assertEqual(result.iloc[30]['close_4h'],df.iloc[15]['close'])
        self.assertEqual(result.iloc[31]['close_4h'],df.iloc[31]['close'])
        valid=result['trend_available_at'].notna()
        self.assertTrue((result.loc[valid,'trend_available_at']<=result.index[valid]+pd.Timedelta(minutes=15)).all())

    def test_future_and_partial_higher_candle_cannot_change_history(self):
        df=self.prices()
        a=run_intraday_strategy(df)
        b=run_intraday_strategy(df.iloc[:1205])
        pd.testing.assert_frame_equal(a.iloc[:1205],b)
        df.iloc[1205:,:4]*=2
        pd.testing.assert_frame_equal(run_intraday_strategy(df).iloc[:1205],b)

    def test_timeout_uses_next_open_and_does_not_reenter_same_bar(self):
        f=self.features()
        f.iloc[0,f.columns.get_loc('pullback_setup')]=True
        f['pullback_setup']=True
        r=signals_from_features(f,IntradayParams(pullback_max_bars=3,cooldown_bars=2))
        self.assertTrue(r.iloc[0]['entries'])
        self.assertFalse(r.iloc[2]['exits'])
        self.assertTrue(r.iloc[3]['exits'])
        self.assertTrue(r.iloc[3]['mandatory_exit'])
        self.assertFalse(r.iloc[5]['entries'])
        self.assertTrue(r.iloc[6]['entries'])

    def test_close_stop_reference_is_actual_next_open(self):
        f=self.features(4)
        f.iloc[0,f.columns.get_loc('squeeze_setup')]=True
        f.iloc[1,f.columns.get_loc('open')]=110.
        f.iloc[1,f.columns.get_loc('close')]=107.
        r=signals_from_features(f)
        self.assertEqual(r.iloc[1]['signal_reason'],'close_stop')
        view=intraday_decision(r.iloc[1],True)
        self.assertEqual(view['action'],'SELL')
        self.assertFalse(view['should_call_gpt'])

    def test_no_exit_of_unmanaged_holdings(self):
        row=signals_from_features(self.features()).iloc[-1]
        self.assertEqual(intraday_decision(row,False)['action'],'HOLD')
        self.assertNotIn('SELL',intraday_decision(row,False)['allowed_actions'])

    def test_cost_floor_and_downtrend_block_entries(self):
        df=self.prices()
        f=intraday_features(df,IntradayParams(min_reward_cost=1000.))
        self.assertFalse(f['pullback_setup'].any())
        self.assertFalse(f['squeeze_setup'].any())
        f=intraday_features(df)
        self.assertFalse(f.loc[~f['bull_4h'],'pullback_setup'].any())
        self.assertFalse(f.loc[~f['bull_4h'],'squeeze_setup'].any())


if __name__=='__main__':
    unittest.main()
