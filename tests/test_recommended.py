import importlib.util
import os
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd
from strategies.band_trend import BandParams,band_features,band_decision
from strategies.pullback_swing import pullback_features
from strategies import gpt_strategy as gpt
from strategy_loader import load_strategy
from trading.decision_state import DecisionState
from trading.managed_position import managed_position,market_quantity,confirm_managed_order

INFO={'status':'TRADING','filters':[{'filterType':'LOT_SIZE','stepSize':'.00001','minQty':'.00001','maxQty':'100'},
                                     {'filterType':'MIN_NOTIONAL','minNotional':'5','applyToMarket':True}]}


def fill(side,qty,commission='0',asset='BTC'):
    return {'orderId':123,'status':'FILLED','side':side,'executedQty':qty,
            'fills':[{'qty':qty,'commission':commission,'commissionAsset':asset}]}


class RecommendedTests(unittest.TestCase):
    def prices(self):
        c=100+np.arange(1000)*.1
        return pd.DataFrame({'open':c,'high':c+1,'low':c-1,'close':c,'volume':1.},
                            index=pd.date_range('2025-01-01',periods=1000,freq='4h',tz='UTC'))

    def test_band_causality_and_buffer(self):
        df=self.prices()
        a=band_features(df)
        pd.testing.assert_frame_equal(a.iloc[:800],band_features(df.iloc[:800]))
        self.assertFalse(a.iloc[:299]['entries'].any())
        self.assertEqual(band_decision(a.iloc[-1],False)['action'],'BUY')
        self.assertEqual(band_decision(a.iloc[-1],True)['action'],'HOLD')
        df['close']=100.
        row=band_features(df).iloc[-1]
        self.assertEqual(band_decision(row,True)['action'],'HOLD')
        self.assertEqual(band_decision(row,False)['action'],'HOLD')

    def test_config_selects_terra_40_and_band(self):
        with patch('strategy_loader.run_gpt_strategy') as run:
            load_strategy('config.recommended.yaml')(self.prices(),managed_position=False)
            args=run.call_args.kwargs
            self.assertEqual(args['params'].engine,'band_trend')
            self.assertEqual(args['params'].model,'gpt-6.1-terra')
            self.assertEqual(args['params'].gpt_authority,.4)
            self.assertEqual(args['params'].band_trend,{'days':50,'band':.01})
            self.assertIs(args['managed_position'],False)

    def test_daily_pullback_causality_and_engine(self):
        df=self.prices()
        df.index=pd.date_range('2023-01-01',periods=len(df),freq='1D',tz='UTC')
        df.iloc[-1,df.columns.get_loc('close')]-=2
        f=pullback_features(df,1)
        self.assertTrue(f.iloc[-1]['entries'])
        pd.testing.assert_frame_equal(f.iloc[:900],pullback_features(df.iloc[:900],1))
        with patch.object(gpt,'_position_state',return_value={'available':True,'in_position':True}), \
             patch.object(gpt,'_gpt_prompt',return_value='test'), \
             patch.object(gpt,'_query_openai',return_value={'action':'BUY','confidence':90}), \
             patch.dict(os.environ,{'OPENAI_API_KEY':'test-only'}):
            runner=load_strategy('config.recommended.yaml',symbol_override='BNBUSDT',timeframe_override='1d',engine_override='pullback_swing')
            result=runner(df,managed_position=False)
            self.assertEqual(result['last_signal'],'BUY')
            self.assertEqual(result['stats']['blended_scores']['BUY'],84)

    def test_shadow_does_not_become_live_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            j=DecisionState(Path(tmp)/'state.json')
            j.record_shadow('BNBUSDT:pullback100_v1','BUY','2026-01-01')
            self.assertTrue(managed_position(j,'BNBUSDT:pullback100_v1','shadow')[0])
            self.assertFalse(managed_position(j,'BNBUSDT:pullback100_v1','live')[0])

    def test_wallet_holdings_do_not_block_managed_entry(self):
        with patch.object(gpt,'_position_state',return_value={'available':True,'in_position':True,'balances':{'BTC':1.}}), \
             patch.object(gpt,'_gpt_prompt',return_value='test'), \
             patch.object(gpt,'_query_openai',return_value={'action':'BUY','confidence':100}) as query, \
             patch.dict(os.environ,{'OPENAI_API_KEY':'test-only'}):
            p=gpt.GPTParams(engine='band_trend',gpt_authority=.4)
            result=gpt.run_gpt_strategy('BTCUSDT',self.prices(),{},'4h',p,managed_position=False)
            self.assertEqual(result['last_signal'],'BUY')
            self.assertEqual(result['confidence'],88)
            query.assert_called_once()
            result=gpt.run_gpt_strategy('BTCUSDT',self.prices(),{},'4h',p)
            self.assertEqual(result['last_signal'],'HOLD')

    def test_terra_veto_and_api_failure_hold(self):
        p=gpt.GPTParams(engine='band_trend',gpt_authority=.4)
        row=band_features(self.prices()).iloc[-1]
        view=band_decision(row,False)
        self.assertEqual(gpt._blend_with_gpt(view,{'action':'HOLD','confidence':100},p)['action'],'HOLD')
        with patch.object(gpt,'_position_state',return_value={'available':True,'in_position':False}), \
             patch.object(gpt,'_gpt_prompt',return_value='test'), \
             patch.object(gpt,'_query_openai',side_effect=RuntimeError('unavailable')), \
             patch.dict(os.environ,{'OPENAI_API_KEY':'test-only'}):
            r=gpt.run_gpt_strategy('BTCUSDT',self.prices(),{},'4h',p,managed_position=False)
            self.assertEqual(r['last_signal'],'HOLD')

    def test_confirmed_lot_fees_dust_restart_and_ambiguous_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json';j=DecisionState(path);key='BTCUSDT:band50_v1'
            self.assertEqual(managed_position(j,key),(False,Decimal(0)))
            j.begin_order(key,'BUY','2026-01-01')
            with self.assertRaises(ValueError):
                managed_position(j,key)
            confirm_managed_order(j,key,fill('BUY','.0005','.0000005'),'BTC','.0005','BUY')
            j=DecisionState(path)
            self.assertEqual(managed_position(j,key),(True,Decimal('.0004995')))
            qty=market_quantity(INFO,'.0004995',100000)
            self.assertEqual(Decimal(qty),Decimal('.00049'))
            j.begin_order(key,'SELL','2026-01-02')
            confirm_managed_order(j,key,fill('SELL',qty,'0.049','USDT'),'BTC',qty,'SELL')
            self.assertEqual(managed_position(j,key),(False,Decimal(0)))
            self.assertEqual(Decimal(j._read()['orders'][key]['dust_qty']),Decimal('.0000095'))
            j.begin_order(key,'BUY','2026-01-03')
            broken=fill('BUY','.0005');broken['status']='PARTIALLY_FILLED'
            with self.assertRaises(ValueError):
                confirm_managed_order(j,key,broken,'BTC','.0005','BUY')
            self.assertIn('pending',j._read()['orders'][key])

    def test_market_rules_reject_tiny_lot(self):
        with self.assertRaises(ValueError):
            market_quantity(INFO,'.000009',100000)
        with self.assertRaises(ValueError):
            market_quantity(INFO,'.00001',100)

    def test_runner_buy_sell_with_existing_wallet_and_no_network(self):
        with patch('binance.client.Client'),patch('trading.notifier.send_message'), \
             patch('trading.performance_tracker.record_trade'),patch.dict(os.environ,{'BOT_CONFIG':'config.recommended.yaml'}):
            spec=importlib.util.spec_from_file_location('recommended_runner_test',Path(__file__).resolve().parents[1]/'run_bot.py')
            m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
            self.assertEqual([(r['symbol'],r['timeframe'],r['engine']) for r in m.SYMBOL_RUNNERS],
                             [('BTCUSDT','4h','band_trend'),('BNBUSDT','1d','pullback_swing')])
            with tempfile.TemporaryDirectory() as tmp:
                m.DECISION_STATE=DecisionState(Path(tmp)/'state.json')
                df=self.prices();df[['open','high','low','close']]=100000.
                portfolio={'total_usdt':101000.,'breakdown':{'BTC':{'balance':1.,'value_usdt':100000.},'USDT':{'balance':1000.,'value_usdt':1000.}}}
                runner=Mock(side_effect=[{'last_signal':'BUY'},{'last_signal':'SELL'}])
                cfg={'symbol':'BTCUSDT','order_quantity':.0005,'strategy_func':runner}
                m.client.get_symbol_info.return_value=INFO
                with patch.object(m,'fetch_live_ohlcv',side_effect=[df.iloc[:-1],df]), \
                     patch.object(m,'closed_bars',side_effect=lambda d,t:d), \
                     patch.object(m,'get_portfolio_value',return_value=portfolio), \
                     patch.object(m,'execute_order',side_effect=[fill('BUY','.00050','.0000005'),fill('SELL','.00049','0.049','USDT')]) as execute:
                    m.run_one_symbol(cfg);m.run_one_symbol(cfg)
                    self.assertIs(runner.call_args_list[0].kwargs['managed_position'],False)
                    self.assertIs(runner.call_args_list[1].kwargs['managed_position'],True)
                    self.assertEqual(Decimal(execute.call_args_list[1].kwargs['quantity']),Decimal('.00049'))
                    self.assertEqual(managed_position(m.DECISION_STATE,'BTCUSDT:band50_v1'),(False,Decimal(0)))


if __name__=='__main__':
    unittest.main()
