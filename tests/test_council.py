import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from strategies.strategy_council import council_features,council_cards,council_decision,council_blend,GROUPS,WEIGHTS
from strategies import gpt_strategy as gpt
from strategy_loader import load_strategy
from trading.decision_state import DecisionState
from trading.managed_position import managed_position


class CouncilTests(unittest.TestCase):
    def prices(self,n=1500):
        c=100+np.arange(n)*.07+np.sin(np.arange(n)/9)*2
        return pd.DataFrame({'open':c,'high':c+1,'low':c-1,'close':c,'volume':100.},
                            index=pd.date_range('2024-01-01',periods=n,freq='4h',tz='UTC'))

    def test_prefix_causality_daily_availability_and_window(self):
        df=self.prices();f=council_features(df)
        for end in [1100,1101,1102,1103,1104,1105]:
            short=council_features(df.iloc[:end])
            pd.testing.assert_frame_equal(short,f.iloc[:end])
            row=council_features(df.iloc[end-1000:end]).iloc[-1]
            for col in ['ready','regime']+[n+'_signal' for n in GROUPS]:
                self.assertEqual(row[col],f.iloc[end-1][col])
            self.assertLessEqual(row.daily_as_of, row.name+pd.Timedelta(hours=4))
        changed=df.copy();changed.iloc[1200:,0:4]*=3
        pd.testing.assert_frame_equal(council_features(changed).iloc[:1200],f.iloc[:1200])

    def test_partial_day_and_gaps_never_silently_filled(self):
        df=self.prices(1102)
        row=council_features(df).iloc[-1]
        previous=df.index[-1].floor('D')
        self.assertEqual(row.daily_as_of,previous)
        gap=df.drop(df.index[-10])
        row=council_features(gap).iloc[-1]
        cards,market=council_cards(row)
        self.assertFalse(row.ready)
        self.assertEqual(council_decision(cards,market,False)['allowed_actions'],['HOLD'])

    def test_groups_capped_and_neutral_not_sell(self):
        cards,market=council_cards(council_features(self.prices()).iloc[-1])
        for group,budget in WEIGHTS[market['regime_label']].items():
            self.assertAlmostEqual(sum(c['weight'] for c in cards.values() if c['category']==group),budget)
        for card in cards.values(): card['signal']='HOLD'
        cards['pullback100']['signal']='BUY'
        local=council_decision(cards,market,False)
        self.assertEqual(local['scores']['BUY'],100)
        self.assertEqual(local['scores']['SELL'],0)
        self.assertIn('BUY',local['allowed_actions'])
        held=council_decision(cards,market,True)
        self.assertNotIn('BUY',held['allowed_actions'])

    def test_final_gpt_choice_and_40_percent_gate(self):
        # Local threshold prefers HOLD; GPT can choose technically supported BUY.
        local={'action':'HOLD','scores':{'BUY':40.,'SELL':60.,'HOLD':0.},
               'allowed_actions':['HOLD','BUY'],'entry_threshold':60.,'exit_threshold':55.,'reason':'conflict'}
        self.assertEqual(council_blend(local,{'action':'BUY','confidence':90},.4)['action'],'BUY')
        self.assertEqual(council_blend(local,{'action':'BUY','confidence':89},.4)['action'],'HOLD')
        for review in [None,{'action':'HOLD','confidence':100},{'action':'SELL','confidence':100},
                       {'action':'BUY','confidence':float('nan')},{'action':'BUY','confidence':True}]:
            self.assertEqual(council_blend(local,review,.4)['action'],'HOLD')
        local['scores']['BUY']=100
        self.assertEqual(council_blend(local,{'action':'HOLD','confidence':1},.4)['action'],'HOLD')

    def test_prompt_preserves_all_experts_and_final_authority(self):
        cards,market=council_cards(council_features(self.prices()).iloc[-1])
        local=council_decision(cards,market,False)
        with patch.object(gpt,'get_news_summary',return_value='none'),patch.object(gpt,'get_fgi',return_value={}),patch.object(gpt,'_funding_summary',return_value='none'):
            payload=json.loads(gpt._gpt_prompt('BTCUSDT','4h',cards,market,local,{'in_position':False},gpt.GPTParams(engine='council_v1',gpt_authority=.4)))
        self.assertEqual(len(payload['strategies']),6)
        self.assertIn('evidence',payload['strategies']['pullback100'])
        self.assertEqual(payload['decision_gate']['gpt_weight'],.4)
        self.assertIn('final BUY/SELL/HOLD',payload['review_mode'])

    def test_runtime_routes_both_coins_and_api_failure_holds(self):
        with patch('strategy_loader.run_gpt_strategy') as run:
            for symbol in ['BTCUSDT','BNBUSDT']:
                load_strategy('config.recommended.yaml',symbol_override=symbol,timeframe_override='4h',engine_override='council_v1')(self.prices(),managed_position=False)
                self.assertEqual(run.call_args.kwargs['params'].engine,'council_v1')
                self.assertEqual(run.call_args.kwargs['params'].gpt_authority,.4)
        with patch.object(gpt,'_position_state',return_value={'available':True,'in_position':False}),patch.object(gpt,'_gpt_prompt',return_value='synthetic'),patch.object(gpt,'_query_openai',side_effect=RuntimeError('offline')),patch.dict(os.environ,{'OPENAI_API_KEY':'test'}):
            result=gpt.run_gpt_strategy('BTCUSDT',self.prices(),{},'4h',gpt.GPTParams(engine='council_v1',gpt_authority=.4),managed_position=False)
            self.assertEqual(result['last_signal'],'HOLD')
            self.assertEqual(len(result['strategy_summaries']),6)

    def test_switch_does_not_orphan_existing_managed_lot(self):
        with tempfile.TemporaryDirectory() as temp:
            journal=DecisionState(Path(temp)/'state.json')
            state=journal._read();state['orders']['BTCUSDT:band50_v1']={'managed_qty':'.001'}
            journal._write(state)
            with self.assertRaisesRegex(ValueError,'Another strategy'):
                managed_position(journal,'BTCUSDT:council_v1','live')

    def test_research_context_is_dated_and_not_fitted_into_weights(self):
        df=self.prices()
        params=gpt.GPTParams(engine='council_v1',gpt_authority=.4)
        _,past=gpt._summaries_from_outputs('BNBUSDT','4h',df,{},params,{})
        self.assertNotIn('research_available_after',past)
        df.index=pd.date_range('2027-01-01',periods=len(df),freq='4h',tz='UTC')
        cards,market=gpt._summaries_from_outputs('BNBUSDT','4h',df,{},params,{})
        self.assertIn('research_available_after',market)
        self.assertNotIn('donchian20',cards)
        self.assertIn('donchian20',market['excluded_strategies'])
        self.assertAlmostEqual(sum(c['weight'] for c in cards.values()),1.)

    def test_asset_specific_removal_preserves_relative_weights(self):
        row=council_features(self.prices()).iloc[-1]
        old,_=council_cards(row)
        btc,_=council_cards(row,'BTCUSDT')
        bnb,market=council_cards(row,'BNBUSDT')
        self.assertEqual(len(btc),6)
        self.assertEqual(len(bnb),5)
        self.assertNotIn('donchian20',bnb)
        self.assertAlmostEqual(sum(x['weight'] for x in bnb.values()),1.)
        self.assertAlmostEqual(bnb['keltner20']['weight']/bnb['band50']['weight'],
                               old['keltner20']['weight']/old['band50']['weight'])
        self.assertAlmostEqual(sum(market['group_budgets'].values()),1.)


if __name__=='__main__': unittest.main()
