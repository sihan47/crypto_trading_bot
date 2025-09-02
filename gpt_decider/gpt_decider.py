import json
from pathlib import Path
from datetime import datetime


def mock_gpt_decision(signals: dict) -> str:
    """Mock GPT decision logic (rule-based)."""
    votes = {"buy": 0, "sell": 0, "hold": 0}

    for strat, data in signals.items():
        stats = data.get("stats", {})
        ret = stats.get("total_return", 0)

        if ret > 0:
            votes["buy"] += 1
        elif ret < 0:
            votes["sell"] += 1
        else:
            votes["hold"] += 1

    decision = max(votes, key=votes.get)
    return decision


def decide_from_json(json_path: Path):
    with open(json_path, "r") as f:
        signals = json.load(f)

    decision = mock_gpt_decision(signals)

    out = {
        "timestamp": datetime.utcnow().isoformat(),
        "decision": decision,
        "strategies": list(signals.keys())
    }

    print(f"✅ GPT Decision from {json_path.name}: {out}")
    return out


if __name__ == "__main__":
    signals_dir = Path(__file__).resolve().parent.parent / "research" / "signals"
    json_files = sorted(signals_dir.glob("*.json"))

    if not json_files:
        print("⚠️ No signals JSON found in research/signals/")
    else:
        latest = json_files[-1]
        decide_from_json(latest)
