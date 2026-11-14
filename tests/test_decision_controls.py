import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from trading.decision_state import DecisionState, closed_bars
from strategies.gpt_strategy import GPTParams, _blend_with_gpt
from trading import order_executor
import strategy_loader
from strategies import gpt_strategy


class DecisionControlsTests(unittest.TestCase):
    def test_closed_bar_and_stale_data(self):
        df = pd.DataFrame({'close':[1.,2.,3.]}, index=pd.date_range('2026-01-01',periods=3,freq='4h',tz='UTC'))
        result = closed_bars(df,'4h',pd.Timestamp('2026-01-01T09:00Z'))
        self.assertEqual(len(result),2)
        with self.assertRaises(ValueError):
            closed_bars(df,'4h',pd.Timestamp('2026-01-02T09:00Z'))

    def test_restart_same_bar_and_order_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.json'
            state = DecisionState(path)
            stamp = pd.Timestamp('2026-01-01',tz='UTC')
            self.assertTrue(state.claim_bar('BTCUSDT','4h',stamp))
            state = DecisionState(path)
            self.assertFalse(state.claim_bar('BTCUSDT','4h',stamp))
            self.assertTrue(state.begin_order('BTCUSDT','BUY',stamp)[0])
            # An interrupted request blocks BOTH sides until reconciled.
            state = DecisionState(path)
            self.assertFalse(state.begin_order('BTCUSDT','BUY',stamp)[0])
            self.assertFalse(state.begin_order('BTCUSDT','SELL',stamp)[0])
            state.complete_order('BTCUSDT',{'orderId':123})
            self.assertFalse(state.begin_order('BTCUSDT','BUY',stamp)[0])
            self.assertTrue(state.begin_order('BTCUSDT','SELL',stamp)[0])

    def test_corrupt_journal_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json'
            path.write_text('not json')
            with self.assertRaises(ValueError):
                DecisionState(path).claim_bar('BTCUSDT','4h',pd.Timestamp('2026-01-01',tz='UTC'))

    def test_shadow_does_not_write_live_order_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = DecisionState(Path(tmp)/'state.json')
            self.assertTrue(state.record_shadow('BTCUSDT','SELL','2026-01-01'))
            self.assertFalse(state.record_shadow('BTCUSDT','SELL','2026-01-02'))
            self.assertEqual(state._read()['orders'],{})
            self.assertTrue(state.begin_order('BTCUSDT','SELL','2026-01-03')[0])

    @patch.object(strategy_loader, 'run_gpt_strategy')
    def test_config_reaches_actual_strategy_runner(self, run):
        runner = strategy_loader.load_strategy(symbol_override='BTCUSDT')
        values = 100 + np.arange(1000)*.01 + np.sin(np.arange(1000))
        df = pd.DataFrame({'open':values,'high':values+1,'low':values-1,'close':values,'volume':1.},
                          index=pd.date_range('2025-01-01',periods=1000,freq='4h',tz='UTC'))
        runner(df)
        actual = run.call_args.kwargs
        self.assertEqual(actual['timeframe'],'4h')
        self.assertEqual(actual['params'].engine,'regime_v2')
        self.assertEqual(actual['params'].ensemble['trend_weight'],.4)
        self.assertEqual(actual['params'].model,'gpt-6.1-sol')

    def view(self, action='BUY'):
        return {'action':action, 'confidence':80, 'reason':'test',
                'scores':{'BUY':80.,'SELL':10.,'HOLD':20.},
                'allowed_actions':['BUY','SELL','HOLD'],'entry_threshold':60.,'exit_threshold':55.}

    def test_agreement_does_not_get_stuck_in_hold(self):
        result=_blend_with_gpt(self.view(),{'action':'BUY','confidence':80},GPTParams(engine='regime_v2',gpt_authority=.2))
        self.assertEqual(result['action'],'BUY')

    def test_gpt_cannot_invent_trade_or_override_atr(self):
        result=_blend_with_gpt(self.view('HOLD'),{'action':'BUY','confidence':100},GPTParams(engine='regime_v2'))
        self.assertEqual(result['action'],'HOLD')
        view=self.view()
        view['allowed_actions']=['SELL','HOLD']
        self.assertEqual(_blend_with_gpt(view,{'action':'BUY','confidence':100},GPTParams(engine='regime_v2'))['action'],'HOLD')

    def test_strong_sell_survives_weak_review(self):
        view=self.view('SELL')
        view['scores']={'BUY':0.,'SELL':75.,'HOLD':25.}
        view['allowed_actions']=['SELL','HOLD']
        self.assertEqual(_blend_with_gpt(view,{'action':'HOLD','confidence':80},GPTParams(engine='regime_v2',gpt_authority=.2))['action'],'SELL')

    @patch.object(gpt_strategy, 'OpenAI')
    def test_malformed_model_output_is_rejected(self, client):
        from types import SimpleNamespace
        client.return_value.chat.completions.create.return_value=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{"action":"BUY","confidence":999}'))],
            usage=None)
        with self.assertRaises(ValueError):
            gpt_strategy._query_openai('test','test',False,notify=False)

    def test_api_failure_produces_hold_in_actual_strategy(self):
        view=self.view()
        view['should_call_gpt']=True
        with patch.object(gpt_strategy,'_position_state',return_value={'available':True,'in_position':True}), \
             patch.object(gpt_strategy,'_summaries_from_outputs',return_value=({},{})), \
             patch.object(gpt_strategy,'_local_decision',return_value=view), \
             patch.object(gpt_strategy,'_gpt_prompt',return_value='test'), \
             patch.object(gpt_strategy,'_query_openai',side_effect=RuntimeError('API unavailable')), \
             patch.dict(gpt_strategy.os.environ,{'OPENAI_API_KEY':'test-not-a-real-key'}):
            df=pd.DataFrame({'close':[100.]},index=pd.date_range('2026-01-01',periods=1,tz='UTC'))
            result=gpt_strategy.run_gpt_strategy('BTCUSDT',df,{},'4h',GPTParams(engine='regime_v2'))
            self.assertEqual(result['last_signal'],'HOLD')
            self.assertIn('unavailable',result['reason'])

    @patch.object(order_executor,'_get_client')
    @patch.object(order_executor,'get_portfolio_value',side_effect=RuntimeError('balance endpoint down'))
    def test_order_success_is_not_lost_to_balance_logging(self, portfolio, get_client):
        get_client.return_value.create_order.return_value={'orderId':123,'status':'FILLED'}
        self.assertEqual(order_executor.execute_order('BUY','BTCUSDT',.0005)['orderId'],123)
        portfolio.assert_not_called()


if __name__ == '__main__':
    unittest.main()
