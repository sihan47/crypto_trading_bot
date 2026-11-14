import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from backtesting.next_bar import simulate
from strategies.regime_ensemble import EnsembleParams, run_regime_ensemble, score_families, decision_from_row


class EnsembleTests(unittest.TestCase):
    def fixture(self, n=700):
        rng = np.random.default_rng(47)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.003, n)))
        return pd.DataFrame({'open':close,'high':close*1.004,'low':close*.996,
                             'close':close,'volume':np.ones(n)},
                            index=pd.date_range('2025-01-01',periods=n,freq='15min',tz='UTC'))

    def row_features(self, **updates):
        row = {'trend':-1.,'breakout':-1.,'reversion':0.,'trend_conf':80.,
               'breakout_conf':75.,'reversion_conf':70.,'regime':'trend',
               'atr_pct':.03,'atr_rank':1.,'adx':35.,'ready':True,'close':100.}
        row.update(updates)
        return pd.DataFrame([row],index=pd.date_range('2025-01-01',periods=1,tz='UTC'))

    def test_future_prices_cannot_change_past_signals(self):
        df = self.fixture()
        whole = run_regime_ensemble(df)
        prefix = run_regime_ensemble(df.iloc[:450])
        pd.testing.assert_frame_equal(whole.iloc[:450],prefix)
        changed = df.copy()
        changed.iloc[450:, :4] *= 3
        pd.testing.assert_frame_equal(run_regime_ensemble(changed).iloc[:450],prefix)

    def test_atr_blocks_buys_but_not_exits(self):
        params = EnsembleParams()
        result = score_families(self.row_features(),params).iloc[0]
        self.assertTrue(result['exits'])
        self.assertFalse(result['entries'])
        view = decision_from_row(result,True,params)
        self.assertEqual(view['action'],'SELL')
        self.assertIn('SELL',view['allowed_actions'])
        self.assertNotIn('BUY',view['allowed_actions'])
        flat = decision_from_row(result,False,params)
        self.assertEqual(flat['action'],'HOLD')
        self.assertNotIn('SELL',flat['allowed_actions'])

    def test_inactive_indicators_do_not_dilute_active_evidence(self):
        result = score_families(self.row_features(trend=1.,breakout=0.,atr_pct=.005,atr_rank=.5)).iloc[0]
        self.assertAlmostEqual(result['buy_score'],80.)
        self.assertTrue(result['entries'])

    def test_tiny_weight_cannot_create_buy_by_itself(self):
        params = EnsembleParams(.01,.49,.5)
        result = score_families(self.row_features(trend=1.,breakout=0.,atr_pct=.005,atr_rank=.5),params).iloc[0]
        self.assertFalse(result['entries'])

    def test_warmup_and_invalid_weights(self):
        result = run_regime_ensemble(self.fixture(100))
        self.assertFalse(result['entries'].any())
        self.assertFalse(result['exits'].any())
        with self.assertRaises(ValueError):
            EnsembleParams(-1,1,1)

    def test_next_open_and_two_sided_costs(self):
        df = self.fixture(3)
        df['open'] = [100,200,300]
        df['close'] = [100,200,300]
        buy = pd.Series([True,False,False],index=df.index)
        sell = pd.Series([False,True,False],index=df.index)
        result = simulate(df,buy,sell,fee=.001,slippage=.0005)
        expected = (300*(1-.0005)*(1-.001)/(200*(1+.0005)*(1+.001))-1)*100
        self.assertAlmostEqual(result['return_pct'],expected)
        self.assertEqual(result['orders'],2)
        self.assertEqual(result['round_trips'],1)

    def test_no_same_bar_fill(self):
        df = self.fixture(3)
        buy = pd.Series([False,False,True],index=df.index)
        result = simulate(df,buy,pd.Series(False,index=df.index))
        self.assertEqual(result['orders'],0)
        self.assertEqual(result['return_pct'],0)


if __name__ == '__main__':
    unittest.main()
