"""Offline checks only: explicit allowlist excludes network/notifier scripts."""
import os
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
suite=unittest.TestSuite()
for pattern in ['test_ensemble.py','test_decision_controls.py','test_runner_shadow.py',
                'test_news_summary.py','test_intraday.py','test_research_candidates.py',
                'test_forum_research.py','test_recommended.py','test_council.py','test_heartbeat.py','test_risk_overlay.py','test_btc_core.py','test_live_context.py']:
    suite.addTests(unittest.TestLoader().discover(str(ROOT/'tests'),pattern=pattern))
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
