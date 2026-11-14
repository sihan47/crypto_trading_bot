"""Paid smoke test for configured models. No orders or Telegram notifications."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from dotenv import load_dotenv
from strategies.btc_news_summary import fetch_news, summarize_with_openai
from strategies.gpt_strategy import _query_openai


def main():
    load_dotenv(ROOT / '.env')
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))['ai']
    news_cfg = cfg['news']
    sample = "--sample" in sys.argv
    headlines = (["SYNTHETIC TEST: A Bitcoin conference was announced; no price data supplied."]
                 if sample else fetch_news())
    if not headlines:
        raise RuntimeError('No headlines available for end-to-end smoke test')
    summary = summarize_with_openai(
        headlines, news_cfg['model'], news_cfg['reasoning_effort'], news_cfg['max_completion_tokens']
    )
    print(json.dumps({'news_model': news_cfg['model'], 'synthetic': sample, 'headlines': len(headlines),
                      'summary_chars': len(summary)}))
    decision_cfg = cfg['openai']
    result = _query_openai(
        'This is an offline integration test. There is no market or position data. '
        'The only allowed action is HOLD. Return JSON with action, confidence, reason. '
        'News is context only: ' + summary,
        decision_cfg['model'], False, decision_cfg['reasoning_effort'],
        decision_cfg['max_completion_tokens'], notify=False,
    )
    parsed = json.loads(result['raw_response'])
    assert parsed['action'] == 'HOLD'
    assert isinstance(parsed['confidence'], (int, float))
    assert 0 <= parsed['confidence'] <= 100
    assert isinstance(parsed['reason'], str) and parsed['reason']
    print('PASS: news and decision models returned complete output; no orders sent.')


if __name__ == '__main__':
    main()
