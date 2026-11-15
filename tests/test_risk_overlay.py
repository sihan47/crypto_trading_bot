import json
import unittest
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from strategies.risk_overlay import validate_response,adapt_response,entry_multiplier,OVERLAY_SYSTEM,overlay_context
from strategies import gpt_strategy as gpt


class RiskOverlayTests(unittest.TestCase):
    def response(self,**changes):
        return dict(red_team_counterargument='The flow may be uninformed.',
                    catalyst_and_flow_alignment='Calendar and percentile feeds are unavailable.',
                    veto_triggered=False,size_multiplier=1.0,execution_note='Approve within configured cap.',**changes)

    def payload(self):
        return {'proposed_action':'BUY','allowed_actions':['BUY','HOLD'],'risk_overlay_context':overlay_context()}

    def test_strict_types_and_veto_consistency(self):
        good=self.response();validate_response(good)
        for change in [{'size_multiplier':True},{'size_multiplier':float('nan')},{'size_multiplier':2},
                       {'veto_triggered':'false'},{'size_multiplier':0},{'veto_triggered':True},
                       {'execution_note':''},{'action':'BUY'}]:
            with self.assertRaises(ValueError):validate_response({**good,**change})

    def test_veto_reduce_and_unknown_feed_boost_cap(self):
        abort=adapt_response({**self.response(),'veto_triggered':True,'size_multiplier':0},self.payload())
        self.assertEqual(abort['action'],'HOLD');self.assertEqual(abort['size_multiplier'],0)
        reduced=adapt_response({**self.response(),'size_multiplier':.25},self.payload())
        self.assertEqual(reduced['action'],'BUY');self.assertEqual(reduced['size_multiplier'],.25)
        boost=adapt_response({**self.response(),'size_multiplier':1.5},self.payload())
        self.assertEqual(boost['size_multiplier'],1.)

    def test_no_direction_invention_or_partial_managed_exit(self):
        p=self.payload();p['proposed_action']='SELL'
        self.assertEqual(adapt_response(self.response(),p)['action'],'HOLD')
        p['allowed_actions']=['SELL','HOLD']
        r=adapt_response({**self.response(),'size_multiplier':.25},p)
        self.assertEqual(r['action'],'SELL');self.assertEqual(r['size_multiplier'],1)

    def test_veto_overrides_strong_technical_score(self):
        local={'action':'BUY','confidence':100,'scores':{'BUY':100,'SELL':0,'HOLD':0},'allowed_actions':['BUY','HOLD']}
        review=adapt_response({**self.response(),'veto_triggered':True,'size_multiplier':0},self.payload())
        for engine in ['regime_v2','council_v1']:
            self.assertEqual(gpt._blend_with_gpt(local,review,gpt.GPTParams(engine=engine,gpt_authority=.4))['action'],'HOLD')

    @patch.object(gpt,'OpenAI')
    def test_system_policy_sent_and_exact_schema_adapted(self,client):
        client.return_value.chat.completions.create.return_value=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(self.response())))],usage=None)
        r=gpt._query_openai(json.dumps(self.payload()),'test-model',False,notify=False)
        self.assertEqual(r['action'],'BUY');self.assertEqual(r['confidence'],100)
        self.assertEqual(client.return_value.chat.completions.create.call_args.kwargs['messages'][0]['content'],OVERLAY_SYSTEM)

    def test_execution_rejects_invalid_multipliers(self):
        self.assertEqual(entry_multiplier({'size_multiplier':.25}),.25)
        for bad in [True,-1,2,float('inf'),'0.25']:
            with self.assertRaises(ValueError):entry_multiplier({'size_multiplier':bad})

    def test_runner_scales_entry_before_exchange_rounding(self):
        import pandas as pd
        from decimal import Decimal
        from trading.decision_state import DecisionState
        with patch('binance.client.Client'),patch('trading.notifier.send_message'),patch('trading.order_executor.execute_order'):
            spec=importlib.util.spec_from_file_location('overlay_runner_test',Path(__file__).resolve().parents[1]/'run_bot.py')
            runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
            runner.RUNTIME['trading_cfg']['execution_mode']='live'
            now=pd.Timestamp.now(tz='UTC').floor('4h')
            df=pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.,'volume':1.},
                            index=pd.date_range(end=now,periods=1000,freq='4h'))
            portfolio={'total_usdt':1000.,'breakdown':{'BTC':{'balance':0.,'value_usdt':0.},'USDT':{'balance':1000.,'value_usdt':1000.}}}
            cfg={'symbol':'BTCUSDT','order_quantity':.1,'engine':'council_v1',
                 'strategy_func':lambda *a,**k:{'last_signal':'BUY','size_multiplier':.25}}
            with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'fetch_live_ohlcv',return_value=df),\
                 patch.object(runner,'get_portfolio_value',return_value=portfolio),\
                 patch.object(runner,'market_quantity',return_value='0.025') as rounding,\
                 patch.object(runner,'execute_order',return_value=None) as execute:
                runner.DECISION_STATE=DecisionState(Path(tmp)/'state.json')
                runner.run_one_symbol(cfg)
                self.assertEqual(rounding.call_args.args[1],Decimal('0.025'))
                execute.assert_called_once_with('BUY',symbol='BTCUSDT',quantity='0.025')


if __name__=='__main__':unittest.main()
