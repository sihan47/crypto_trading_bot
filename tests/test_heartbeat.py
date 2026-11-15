import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
from trading.heartbeat import Heartbeat
from trading import notifier


class HeartbeatTests(unittest.TestCase):
    def test_four_hour_slots_survive_restart_and_hold_needs_no_gpt(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'heartbeat.json';send=Mock(return_value=True)
            states={'BTCUSDT':{'health':'評估完成','decision':'HOLD','gpt':'未呼叫','reason':'證據不足'}}
            self.assertTrue(Heartbeat(path).send_if_due(send,states,'live','2026-11-15T08:01Z'))
            self.assertFalse(Heartbeat(path).send_if_due(send,states,'live','2026-11-15T11:59Z'))
            self.assertTrue(Heartbeat(path).send_if_due(send,states,'live','2026-11-15T12:01Z'))
            self.assertEqual(send.call_count,2)
            self.assertIn('HOLD',send.call_args.args[0]);self.assertIn('未呼叫',send.call_args.args[0])

    def test_failed_delivery_retries_without_marking_delivered(self):
        with tempfile.TemporaryDirectory() as temp:
            h=Heartbeat(Path(temp)/'heartbeat.json');send=Mock(side_effect=[False,True])
            self.assertFalse(h.send_if_due(send,{},'shadow','2026-11-15T08:00Z'))
            self.assertFalse(h.send_if_due(send,{},'shadow','2026-11-15T08:04Z'))
            self.assertTrue(h.send_if_due(send,{},'shadow','2026-11-15T08:05Z'))
            self.assertEqual(send.call_count,2)

    def test_notifier_requires_telegram_acknowledgment(self):
        with patch.object(notifier,'TELEGRAM_TOKEN','test'),patch.object(notifier,'CHAT_ID','test'),patch.object(notifier.requests,'post') as post:
            post.return_value.status_code=200;post.return_value.json.return_value={'ok':False}
            self.assertFalse(notifier.send_message('test'))
            post.return_value.json.return_value={'ok':True}
            self.assertTrue(notifier.send_message('test'))
            post.side_effect=RuntimeError('test')
            self.assertFalse(notifier.send_message('test'))

    def test_cycle_errors_still_send_heartbeat_without_orders(self):
        with patch('binance.client.Client'),patch('trading.notifier.send_message'),patch('trading.performance_tracker.record_trade'):
            spec=importlib.util.spec_from_file_location('heartbeat_runner_test',Path(__file__).resolve().parents[1]/'run_bot.py')
            m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as temp,patch.object(m,'_maybe_reload_config'),patch.object(m,'run_one_symbol',side_effect=ValueError('synthetic data error')),patch.object(m,'get_portfolio_value',side_effect=ValueError('synthetic account error')),patch.object(m,'send_message',return_value=True) as send,patch.object(m,'execute_order') as order:
            m.HEARTBEAT=Heartbeat(Path(temp)/'heartbeat.json')
            m.run_all_symbols()
            self.assertEqual(send.call_count,1)
            self.assertIn('執行錯誤',send.call_args.args[0])
            order.assert_not_called()

    def test_config_failure_reports_and_does_not_run_strategy(self):
        with patch('binance.client.Client'),patch('trading.notifier.send_message'):
            spec=importlib.util.spec_from_file_location('heartbeat_config_test',Path(__file__).resolve().parents[1]/'run_bot.py')
            m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as temp,patch.object(m,'_maybe_reload_config',side_effect=ValueError('bad config')),patch.object(m,'run_one_symbol') as run,patch.object(m,'send_message',return_value=True) as send:
            m.HEARTBEAT=Heartbeat(Path(temp)/'heartbeat.json')
            m.run_all_symbols()
            run.assert_not_called()
            self.assertIn('設定錯誤',send.call_args.args[0])


if __name__=='__main__': unittest.main()
