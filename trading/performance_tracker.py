"""
Persistent performance tracking.

Two append-only journals:
  - logs/trades.jsonl     : one record per executed BUY/SELL (delta = fees+slippage at execution)
  - logs/snapshots.jsonl  : one record per main-loop cycle (portfolio value over time)

Plus a regularly-rewritten summary file:
  - logs/performance_summary.json  : aggregate stats for quick inspection
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


pathlib.Path("logs").mkdir(exist_ok=True)

perf_logger = logger.bind(tag="PERF")
logger.add(
    "logs/performance.log",
    rotation="1 week",
    retention="90 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | [{extra[tag]}] {message}",
    filter=lambda record: record["extra"].get("tag") == "PERF",
)

LOGS_DIR = pathlib.Path("logs")
TRADES_FILE = LOGS_DIR / "trades.jsonl"
SNAPSHOTS_FILE = LOGS_DIR / "snapshots.jsonl"
SUMMARY_FILE = LOGS_DIR / "performance_summary.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_trade(
    symbol: str,
    side: str,
    quantity: float,
    portfolio_before: dict,
    portfolio_after: dict,
    meta: Optional[dict] = None,
) -> None:
    """Append a single executed trade to logs/trades.jsonl."""
    delta = round(
        portfolio_after.get("total_usdt", 0.0) - portfolio_before.get("total_usdt", 0.0), 4
    )
    record = {
        "ts": _now_iso(),
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "total_before_usdt": portfolio_before.get("total_usdt", 0.0),
        "total_after_usdt": portfolio_after.get("total_usdt", 0.0),
        "delta_usdt": delta,
        "breakdown_after": portfolio_after.get("breakdown", {}),
        "meta": meta or {},
    }
    with open(TRADES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    perf_logger.info(f"TRADE {side} {symbol} qty={quantity} Δ${delta:+.4f}")


def record_snapshot(portfolio: dict, label: str = "cycle") -> None:
    """Append a portfolio snapshot to logs/snapshots.jsonl."""
    record = {
        "ts": _now_iso(),
        "label": label,
        "total_usdt": portfolio.get("total_usdt", 0.0),
        "breakdown": portfolio.get("breakdown", {}),
    }
    with open(SNAPSHOTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def summary() -> dict[str, Any]:
    """Compute aggregate stats from journals and persist to performance_summary.json."""
    trades = _read_jsonl(TRADES_FILE)
    snapshots = _read_jsonl(SNAPSHOTS_FILE)

    s: dict[str, Any] = {
        "generated_at": _now_iso(),
        "total_trades": len(trades),
        "total_snapshots": len(snapshots),
    }

    if trades:
        # Per-trade delta is mostly fees + slippage at execution moment.
        # True P&L over holding periods is captured by snapshots, not individual trade deltas.
        by_symbol: dict[str, dict[str, Any]] = {}
        for t in trades:
            sym = t["symbol"]
            d = t["delta_usdt"]
            row = by_symbol.setdefault(
                sym, {"trades": 0, "buys": 0, "sells": 0, "sum_delta": 0.0}
            )
            row["trades"] += 1
            row["sum_delta"] += d
            if t["side"] == "BUY":
                row["buys"] += 1
            elif t["side"] == "SELL":
                row["sells"] += 1
        for sym, row in by_symbol.items():
            row["sum_delta"] = round(row["sum_delta"], 2)
        s["by_symbol"] = by_symbol
        s["total_execution_cost_usdt"] = round(sum(t["delta_usdt"] for t in trades), 2)

    if snapshots:
        first = snapshots[0]
        last = snapshots[-1]
        s["first_snapshot_ts"] = first["ts"]
        s["latest_snapshot_ts"] = last["ts"]
        s["start_total_usdt"] = first["total_usdt"]
        s["latest_total_usdt"] = last["total_usdt"]
        s["pnl_usdt"] = round(last["total_usdt"] - first["total_usdt"], 2)
        if first["total_usdt"] > 0:
            s["pnl_pct"] = round((last["total_usdt"] / first["total_usdt"] - 1) * 100, 4)

        # Max drawdown across the snapshot series
        peak = 0.0
        max_dd_pct = 0.0
        for snap in snapshots:
            v = snap["total_usdt"]
            if v > peak:
                peak = v
            if peak > 0:
                dd = (v / peak - 1) * 100
                if dd < max_dd_pct:
                    max_dd_pct = dd
        s["max_drawdown_pct"] = round(max_dd_pct, 4)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    return s


if __name__ == "__main__":
    # `python trading/performance_tracker.py` prints current summary.
    import pprint
    pprint.pprint(summary())
