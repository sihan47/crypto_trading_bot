"""Read public closed bars and optionally ask GPT; NEVER submit any orders."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from strategies.regime_ensemble import EnsembleParams, run_regime_ensemble, decision_from_row
from strategies.gpt_strategy import GPTParams, _query_openai, _blend_with_gpt
from trading.decision_state import closed_bars


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--review',action='store_true',help='Paid API review; synthetic position, no orders')
    args=parser.parse_args()
    load_dotenv(ROOT/'.env')
    cfg=yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
    ep=EnsembleParams(**cfg['strategy']['params']['ensemble'])
    ai=cfg['ai']['openai']
    gp=GPTParams(engine='regime_v2',ensemble=cfg['strategy']['params']['ensemble'],
                 gpt_authority=cfg['strategy']['params']['gpt_authority'])
    for entry in cfg['symbols']:
        sym=entry['symbol']
        r=requests.get('https://api.binance.com/api/v3/klines',params={
            'symbol':sym,'interval':cfg['data']['timeframe'],'limit':1000},timeout=30)
        r.raise_for_status()
        df=pd.DataFrame([row[:6] for row in r.json()],columns=['time','open','high','low','close','volume'])
        df.index=pd.to_datetime(df.pop('time'),unit='ms',utc=True)
        df=closed_bars(df.astype(float),cfg['data']['timeframe'])
        row=run_regime_ensemble(df,ep).iloc[-1]
        view=decision_from_row(row,True,ep)
        print(json.dumps({'symbol':sym,'last_closed_bar':str(df.index[-1]),'assumed_in_position':True,
                          'local':view},ensure_ascii=False))
        if args.review and view['action'] != 'HOLD':
            prompt=json.dumps({'mode':'read-only diagnostic; no order execution',
                               'symbol':sym,'local_engine':view,'allowed_actions':view['allowed_actions'],
                               'instructions':'Return JSON action, confidence (integer 0-100), reason. '
                               'Assess only this technical evidence. No news or real account state is provided.'})
            review=_query_openai(prompt,ai['model'],False,ai['reasoning_effort'],ai['max_completion_tokens'],notify=False)
            print(json.dumps({'review':review,'final':_blend_with_gpt(view,review,gp)},ensure_ascii=False))


if __name__ == '__main__':
    main()
