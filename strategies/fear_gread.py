# research/fear_greed.py
import json
import time
from pathlib import Path
import urllib.request

_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
_CACHE.mkdir(parents=True, exist_ok=True)
_FGI_FILE = _CACHE / "fear_greed.json"
_TTL = 60 * 60  # 1 小時

def _fetch_latest():
    # 使用 alternative.me 的免費 API（無需金鑰）
    url = "https://api.alternative.me/fng/?limit=1&format=json"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))
    v = int(data["data"][0]["value"])
    c = data["data"][0]["value_classification"]
    ts = int(data["data"][0]["timestamp"])
    return {"value": v, "label": c, "timestamp": ts}

def get_fgi():
    now = int(time.time())
    if _FGI_FILE.exists():
        try:
            cached = json.loads(_FGI_FILE.read_text())
            if now - cached.get("fetched_at", 0) < _TTL:
                return cached["payload"]
        except Exception:
            pass
    try:
        payload = _fetch_latest()
        _FGI_FILE.write_text(json.dumps({"fetched_at": now, "payload": payload}, ensure_ascii=False))
        return payload
    except Exception:
        # 失敗就讀舊檔；再失敗就給中性
        if _FGI_FILE.exists():
            return json.loads(_FGI_FILE.read_text())["payload"]
        return {"value": 50, "label": "Neutral", "timestamp": now}
