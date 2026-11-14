"""Exercise the real runner with every external side effect replaced."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from trading.decision_state import DecisionState


class RunnerShadowTest(unittest.TestCase):
    def test_shadow_and_repeated_bar_never_submit_order(self):
        with patch('binance.client.Client'), patch('trading.notifier.send_message'), \
             patch('trading.order_executor.execute_order') as execute:
            spec = importlib.util.spec_from_file_location('runner_under_test', Path(__file__).resolve().parents[1]/'run_bot.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as tmp:
                module.DECISION_STATE=DecisionState(Path(tmp)/'state.json')
                module.RUNTIME['trading_cfg']['execution_mode']='shadow'
                now=pd.Timestamp.now(tz='UTC').floor('4h')
                df=pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.,'volume':1.},
                                index=pd.date_range(end=now,periods=1000,freq='4h'))
                portfolio={'total_usdt':1000.,'breakdown':{
                    'BTC':{'balance':1.,'value_usdt':100.},
                    'USDT':{'balance':900.,'value_usdt':900.}}}
                with patch.object(module,'fetch_live_ohlcv',return_value=df), \
                     patch.object(module,'get_portfolio_value',return_value=portfolio):
                    from unittest.mock import Mock
                    strategy=Mock(return_value={'last_signal':'SELL'})
                    cfg={'symbol':'BTCUSDT','order_quantity':.0005,'strategy_func':strategy}
                    module.run_one_symbol(cfg)
                    module.run_one_symbol(cfg)
                    strategy.assert_called_once()
                    self.assertEqual(strategy.call_args.args[0].index[-1],now-pd.Timedelta(hours=4))
                    execute.assert_not_called()
                    self.assertEqual(module.DECISION_STATE._read()['shadow']['BTCUSDT']['side'],'SELL')


if __name__ == '__main__':
    unittest.main()
