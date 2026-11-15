"""Public-data-only smoke check: no GPT, account access, orders or notifications."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from trading.live_context import get_live_context

if __name__=='__main__':
    result=get_live_context(sys.argv[1] if len(sys.argv)>1 else 'BTCUSDT')
    path=ROOT/'output/live_context_probe.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))
