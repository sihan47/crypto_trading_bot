from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import random
import os
import re
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from binance.client import Client

# Load environment variables from .env
load_dotenv()


@dataclass
class GPTParams:
    provider: str = "mock"       # mock | openai
    vote_threshold: int = 2
    exit_vote_threshold: int = 1
    hour_momentum_threshold: float = 0.002
    mode: str = "backtest"       # backtest | live
    weight_sma: float = 1.0
    weight_rsi: float = 1.0
    weight_macd: float = 1.0
    weight_bollinger: float = 1.0
    context_hours: int = 4       # default: 4 hours context


def _load_best_params():
    """Load best_params.json, which includes both params and __backtest section."""
    path = Path(__file__).resolve().parents[1] / "research" / "best_params.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _decorate_strategy_name(raw_name: str, symbol: str, timeframe: str, best_map: dict) -> str:
    """Return strategy name with best params + backtest info if available."""
    name = raw_name.upper()
    key = f"{symbol}_{timeframe}_{raw_name.lower()}"

    params_entry = best_map.get(key)
    backtest_entry = best_map.get("__backtest", {}).get(key)

    if not params_entry:
        return f"{name} (no backtest record)"

    if isinstance(params_entry, dict):
        params_str = ", ".join(f"{k}={v}" for k, v in params_entry.items())
    else:
        params_str = str(params_entry)

    if backtest_entry:
        perf = backtest_entry.get("performance", "NA")
        period = backtest_entry.get("period", "NA")
        return f"{name} (best: {params_str}, perf={perf}, period={period})"
    else:
        return f"{name} (best: {params_str}) (incomplete record)"


def _fetch_last_context_ohlcv(symbol="BTCUSDT", interval="15m", context_hours=4):
    """Fetch last N hours OHLCV from Binance (15m bars)."""
    api_key = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY")
    client = Client(api_key, secret_key)

    limit = max(1, (context_hours * 60) // 15)  # number of 15m bars
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(
        klines,
        columns=["timestamp", "open", "high", "low", "close", "volume",
                 "_1", "_2", "_3", "_4", "_5", "_6"]
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)  # tz-aware UTC
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df


def _make_prompt(
    symbol: str,
    strat_signals: Dict[str, str],
    ohlcv_last_context: pd.DataFrame,
    in_pos: bool,
    context_hours: int,
    timeframe: str
) -> str:
    """Construct a prompt for GPT using strategy signals, position status, and last N hours OHLCV (15m bars)."""
    best_map = _load_best_params()
    lines = []
    lines.append(f"You are a trading assistant for {symbol}. Decide BUY, SELL, or HOLD.")
    lines.append(f"\n--- Current Position ---\n{'IN POSITION' if in_pos else 'NO POSITION'}")
    lines.append("\n--- Strategy signals ---")
    for name, sig in strat_signals.items():
        decorated_name = _decorate_strategy_name(name, symbol, timeframe, best_map)
        lines.append(f"{decorated_name}: {sig}")
    lines.append(f"\n--- Last {context_hours} hours OHLCV (15m bars) ---")
    for _, row in ohlcv_last_context.iterrows():
        lines.append(
            f"{row['timestamp']} O:{row['open']:.2f} H:{row['high']:.2f} "
            f"L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:.2f}"
        )
    return "\n".join(lines)


def _query_openai(prompt: str) -> str:
    """Send prompt to OpenAI GPT model and return its decision (BUY/SELL/HOLD)."""
    print("\n=== GPT Prompt Preview ===")
    print(prompt[:2000])  # preview first 2000 chars
    print("==========================\n")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "Answer ONLY with one word: BUY, SELL, or HOLD. No explanation."},
            {"role": "user", "content": prompt},
        ],
    )

    full_response = (resp.choices[0].message.content or "").strip().upper()
    match = re.search(r"\b(BUY|SELL|HOLD)\b", full_response)
    if not match:
        print(f"Unexpected GPT response: {full_response}")
        decision = "HOLD"  # fallback
    else:
        decision = match.group(1)

    print(f"\n=== GPT Decision ===\n{decision}\n====================\n")
    return decision


def run_gpt_strategy(
    symbol: str,
    ohlcv: pd.DataFrame,
    other_results: Dict[str, Dict[str, pd.Series]],
    timeframe: str,
    params: GPTParams,
    force: bool = False,  # ✅ 新增
) -> Dict[str, Any]:
    n = len(ohlcv)
    entries = pd.Series(False, index=ohlcv.index, name="entries")
    exits = pd.Series(False, index=ohlcv.index, name="exits")

    in_pos = False
    last_signal = None
    decisions_count = 0

    for i in range(n):
        strat_signals: Dict[str, str] = {}
        changed = False

        for name, out in other_results.items():
            sig = "HOLD"
            if out["entries"].iloc[i]:
                sig = "BUY"
            elif out["exits"].iloc[i]:
                sig = "SELL"
            strat_signals[name] = sig

            if i > 0:
                prev_sig = "HOLD"
                if out["entries"].iloc[i - 1]:
                    prev_sig = "BUY"
                elif out["exits"].iloc[i - 1]:
                    prev_sig = "SELL"
                if sig != prev_sig:
                    changed = True

        if params.mode == "live":
            try:
                from trading.order_executor import get_balances
                balances = get_balances()
                base = symbol.replace("USDT", "")
                in_pos = balances.get(base, 0) > 0
            except Exception:
                in_pos = False

        # ✅ 如果 force=True，或是訊號有變，就一定跑 GPT
        if force or changed:
            if params.provider == "openai":
                if params.mode == "backtest":
                    bars_for_context = min(max(1, (params.context_hours * 60) // 15), i + 1)
                    ohlcv_last_context = ohlcv.iloc[i - bars_for_context + 1: i + 1].copy()
                    if "timestamp" not in ohlcv_last_context.columns:
                        ohlcv_last_context["timestamp"] = ohlcv_last_context.index
                else:
                    ohlcv_last_context = _fetch_last_context_ohlcv(
                        symbol=symbol, interval=timeframe, context_hours=params.context_hours
                    )
                prompt = _make_prompt(symbol, strat_signals, ohlcv_last_context, in_pos, params.context_hours, timeframe)
                signal = _query_openai(prompt)
            else:
                signal = random.choice(["BUY", "SELL", "HOLD"])

            last_signal = signal
            decisions_count += 1
            print(f"GPT decided {signal} at {ohlcv.index[i]}")

        if params.mode == "backtest":
            if last_signal == "BUY" and not in_pos:
                entries.iloc[i] = True
                in_pos = True
            elif last_signal == "SELL" and in_pos:
                exits.iloc[i] = True
                in_pos = False

    return {
        "entries": entries,
        "exits": exits,
        "params": vars(params),
        "stats": {"decisions_count": decisions_count},
        "last_signal": last_signal if last_signal else "HOLD",
    }
