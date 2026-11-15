import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import pandas as pd
import yaml
from trading.decision_state import DecisionState
ROOT=Path(__file__).resolve().parents[1]


class BtcCoreTests(unittest.TestCase):
    def test_profile_and_exit_only_execution(self):
        cfg=yaml.safe_load((ROOT/'config.btc_core.yaml').read_text())
        self.assertEqual(cfg['strategy']['params']['gpt_authority'],.4)
        self.assertTrue(cfg['symbols'][0]['entry_enabled'])
        self.assertFalse(cfg['symbols'][1]['entry_enabled'])
        with patch('binance.client.Client'),patch('trading.notifier.send_message'),patch('trading.order_executor.execute_order') as execute:
            spec=importlib.util.spec_from_file_location('core_profile_test',ROOT/'run_bot.py')
            runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
            with patch.object(runner,'load_strategy',return_value=Mock()):
                rt=runner._build_runtime(cfg)
                self.assertFalse(rt['symbol_runners'][1]['entry_enabled'])
                cfg['symbols'][1]['entry_enabled']='false'
                with self.assertRaises(ValueError):runner._build_runtime(cfg)
            now=pd.Timestamp.now(tz='UTC').floor('4h')
            df=pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.,'volume':1.},
                            index=pd.date_range(end=now,periods=1000,freq='4h'))
            portfolio={'total_usdt':1000.,'breakdown':{'BNB':{'balance':1.,'value_usdt':100.},'USDT':{'balance':900.,'value_usdt':900.}}}
            with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'fetch_live_ohlcv',return_value=df),patch.object(runner,'get_portfolio_value',return_value=portfolio):
                runner.RUNTIME['trading_cfg']['execution_mode']='shadow'
                strategy=Mock(return_value={'last_signal':'BUY','size_multiplier':1})
                symbol={'symbol':'BNBUSDT','engine':'legacy','order_quantity':.05,'entry_enabled':False,'strategy_func':strategy}
                runner.DECISION_STATE=DecisionState(Path(tmp)/'buy.json')
                runner.run_one_symbol(symbol)
                self.assertEqual(runner.DECISION_STATE._read().get('shadow',{}),{})
                strategy.return_value={'last_signal':'SELL'}
                runner.DECISION_STATE=DecisionState(Path(tmp)/'sell.json')
                runner.run_one_symbol(symbol)
                self.assertEqual(runner.DECISION_STATE._read()['shadow']['BNBUSDT']['side'],'SELL')
                execute.assert_not_called()


if __name__=='__main__':unittest.main()
