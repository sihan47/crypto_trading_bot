import json
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from trading.live_microstructure import summarize,FIELDS
from trading.macro_calendar import parse_ics,parse_fed,parse_weekly,macro_veto,SOURCES,utc
from trading.live_context import get_live_context
from strategies.risk_overlay import adapt_response

class LiveContextTests(unittest.TestCase):
    def test_ics_timezone_folding_cancel(self):
        body='BEGIN:VEVENT\nSUMMARY:Consumer Price\n  Index\nDTSTART;TZID=America/New_York:20260714T083000\nEND:VEVENT\nBEGIN:VEVENT\nSUMMARY:Employment Situation\nDTSTART:20260703T123000Z\nSTATUS:CANCELLED\nEND:VEVENT'
        events=parse_ics(body,SOURCES['bls'])
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['time_utc'],'2026-07-14T12:30:00+00:00')

    def test_fed_year_and_dst(self):
        body='2026 FOMC Meetings<div class="row fomc-meeting"><div class="fomc-meeting__month"><strong>January</strong></div><div class="fomc-meeting__date">27-28</div>2027 FOMC Meetings<div class="row fomc-meeting"><div class="fomc-meeting__month"><strong>June</strong></div><div class="fomc-meeting__date">8-9</div>'
        events=parse_fed(body,SOURCES['fed'])
        self.assertEqual(events[0]['time_utc'],'2026-01-28T19:00:00+00:00')
        self.assertEqual(events[1]['time_utc'],'2027-06-09T18:00:00+00:00')

    def test_weekly_coverage_not_inferred_from_empty(self):
        body=json.dumps([{'title':'CPI m/m','country':'USD','date':'2026-11-18T08:30:00-05:00'}])
        events,covers=parse_weekly(body,SOURCES['ff_week'],utc('2026-11-17T00:00Z'))
        self.assertTrue(covers)
        self.assertEqual(events[0]['provenance'],'third_party_calendar')
        self.assertFalse(parse_weekly(body,SOURCES['ff_week'],utc('2026-11-22T00:00Z'))[1])
        with self.assertRaises(ValueError):parse_weekly('[]','x',utc())

    def test_veto_fresh_boundary_and_stale(self):
        now=utc('2026-11-17T00:00Z')
        cal={'status':'partial','as_of':now.isoformat(),'events':[{'kind':'CPI','source':SOURCES['bls'],'time_utc':(now+pd.Timedelta(hours=4)).isoformat()}]}
        self.assertEqual(len(macro_veto(cal,now)),1)
        self.assertEqual(macro_veto(cal,now-pd.Timedelta(seconds=1)),[])
        self.assertEqual(macro_veto(cal,now+pd.Timedelta(hours=2)),[])

    def test_hard_veto_buy_not_exit(self):
        now=utc()
        cal={'status':'partial','as_of':now.isoformat(),'events':[{'kind':'CPI','source':SOURCES['bls'],'time_utc':(now+pd.Timedelta(hours=2)).isoformat()}]}
        review={'red_team_counterargument':'risk','catalyst_and_flow_alignment':'flow','veto_triggered':False,'size_multiplier':1.5,'execution_note':'approve'}
        payload={'proposed_action':'BUY','allowed_actions':['BUY','SELL','HOLD'],'risk_overlay_context':{'macro_calendar':cal,'max_size_multiplier':1}}
        result=adapt_response(review,payload)
        self.assertEqual(result['action'],'HOLD')
        self.assertEqual(result['size_multiplier'],0)
        self.assertTrue(result['macro_veto_enforced'])
        payload['proposed_action']='SELL'
        self.assertEqual(adapt_response(review,payload)['action'],'SELL')
        self.assertEqual(adapt_response(review,payload)['size_multiplier'],1)

    def test_context_failure_explicit(self):
        with patch('trading.live_context.fetch_telemetry',side_effect=RuntimeError('offline')),patch('trading.live_context.fetch_calendar',return_value={'status':'partial'}):
            result=get_live_context('BTCUSDT')
        self.assertEqual(result['telemetry']['status'],'unavailable')
        self.assertEqual(result['macro_calendar']['status'],'partial')
        self.assertEqual(result['max_size_multiplier'],1)

    def test_percentile_excludes_current_and_stale_rejected(self):
        now=utc('2026-11-17T00:00Z')
        index=pd.date_range(end=now,periods=2200,freq='h')
        panel=pd.DataFrame({'close':np.arange(2200)+100.,'funding_rate':.0001,'create_time':index-pd.Timedelta(hours=1),'funding_time':index-pd.Timedelta(hours=2)},index=index)
        for field in FIELDS:panel[field]=1.
        engineered=pd.DataFrame({name:np.arange(2200,dtype=float) for name in ['squeeze_anomaly','smart_money_floor','compression_x_flow_acceleration','signed_flow_low_impact']},index=index)
        with patch('trading.live_microstructure.features',return_value=engineered):
            result=summarize(panel,'BTCUSDT',now,[])
        self.assertEqual(result['squeeze_anomaly']['reference_n'],2160)
        self.assertEqual(result['squeeze_anomaly']['reference_end'],str(index[-2]))
        self.assertEqual(result['squeeze_anomaly']['percentile'],100)
        with self.assertRaises(ValueError):summarize(panel,'BTCUSDT',now+pd.Timedelta(hours=4),[])

if __name__=='__main__':unittest.main()

