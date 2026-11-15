"""Configuration/provider contract check. Synthetic data, no exchange/account calls.

Default is offline. --review sends two small synthetic prompts to configured
OpenAI; it never sends Telegram messages or submits an exchange order.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
from strategies.gpt_strategy import GPTParams,_summaries_from_outputs,_local_decision,_blend_with_gpt,_query_openai


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--review',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/'config.recommended.yaml').read_text(encoding='utf-8'))
    for item in cfg['symbols']:
        engine=item['engine'];tf=item['timeframe']
        params=GPTParams(engine=engine,band_trend=cfg['strategy']['params']['band_trend'],
                         gpt_authority=cfg['strategy']['params']['gpt_authority'],**cfg['ai']['openai'])
        c=100+np.arange(1000)*.1
        if engine=='pullback_swing': c[-1]-=2
        df=pd.DataFrame({'open':c,'high':c+1,'low':c-1,'close':c,'volume':1.},
                        index=pd.date_range('2020-01-01',periods=len(c),freq=tf,tz='UTC'))
        cards,market=_summaries_from_outputs(item['symbol'],tf,df,{},params,{})
        local=_local_decision(cards,market,params,False)
        print(item['symbol'],engine,params.model,params.gpt_authority,'synthetic local:',local['action'])
        if args.review:
            prompt=json.dumps({'purpose':'Synthetic integration test only; no real portfolio or orders.',
                'instruction':'Review local technical action; return JSON action BUY/SELL/HOLD, confidence integer 0..100, reason.',
                'allowed_actions':local['allowed_actions'],'local':local,'market':market,'cards':cards})
            review=_query_openai(prompt,params.model,False,params.reasoning_effort,params.max_completion_tokens,notify=False)
            print('Provider schema OK; synthetic blend:',_blend_with_gpt(local,review,params)['action'])


if __name__=='__main__':
    main()
