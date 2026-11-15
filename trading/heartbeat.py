"""Four-hour UTC status notifications independent of GPT/trade eligibility."""
import json
from pathlib import Path
import pandas as pd


class Heartbeat:
    def __init__(self,path):
        self.path=Path(path)

    def send_if_due(self,sender,statuses,mode,now=None):
        now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
        if now.tzinfo is None: raise ValueError('Heartbeat time must be timezone aware')
        now=now.tz_convert('UTC');bucket=now.floor('4h').isoformat()
        state=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        if state.get('sent_bucket')==bucket: return False
        if now.timestamp()-state.get('last_attempt',0)<300: return False
        state['last_attempt']=now.timestamp()
        self._write(state)
        lines=[f"💓 Bot heartbeat | {now.strftime('%Y-%m-%d %H:%M UTC')}",f"模式：{mode} | 每 4 小時回報（不額外呼叫 GPT）"]
        for symbol,s in statuses.items():
            lines.append(f"{symbol} | {s.get('health','尚未檢查')} | 最近決策：{s.get('decision','未知')} | GPT：{s.get('gpt','未評估')}")
            if s.get('bar'): lines.append(f"K線開盤時間：{s['bar']}")
            lines.append(f"說明：{str(s.get('reason','尚無摘要'))[:240]}")
            if s.get('order'): lines.append(f"訂單：{s['order']}")
        lines.append('此訊息確認本次主迴圈有執行；不代表已成交。程序停止或 Telegram 故障時，無法自行發出通知。')
        if sender('\n'.join(lines)[:3900]) is True:
            state['sent_bucket']=bucket
            self._write(state)
            return True
        return False

    def _write(self,state):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(state),encoding='utf-8');temp.replace(self.path)
