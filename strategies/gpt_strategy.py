from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
import random
import os
import re

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


def _fetch_last_context_ohlcv(symbol="BTCUSDT", interval="15m", context_hours=4):
    """Fetch last N hours OHLCV from Binance (15m bars)."""
    api_key = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY")
    client = Client(api_key, secret_key)

    limit = (context_hours * 60) // 15  # bars count
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(
        klines,
        columns=["timestamp", "open", "high", "low", "close", "volume",
                 "_1", "_2", "_3", "_4", "_5", "_6"]
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df


def _make_prompt(symbol: str, strat_signals: Dict[str, str], ohlcv_last_context: pd.DataFrame, in_pos: bool, context_hours: int) -> str:
    """Construct a prompt for GPT using strategy signals, position status, and last N hours OHLCV (15m bars)."""
    lines = []
    lines.append(f"You are a trading assistant for {symbol}. Decide BUY, SELL, or HOLD.")
    lines.append(f"\n--- Current Position ---\n{'IN POSITION' if in_pos else 'NO POSITION'}")
    lines.append("\n--- Strategy signals ---")
    for name, sig in strat_signals.items():
        lines.append(f"{name}: {sig}")
    lines.append(f"\n--- Last {context_hours} hours OHLCV (15m bars) ---")
    for _, row in ohlcv_last_context.iterrows():
        lines.append(f"{row['timestamp']} O:{row['open']:.2f} H:{row['high']:.2f} "
                     f"L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:.2f}")
    return "\n".join(lines)


def _query_openai(prompt: str) -> str:
    """Send prompt to OpenAI GPT model and return its decision (text only, validated)."""
    print("\n=== GPT Prompt Preview ===")
    print(prompt[:800])  # limit preview
    print("==========================\n")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "Answer ONLY with one word: BUY, SELL, or HOLD. No explanation."},
            {"role": "user", "content": prompt},
        ],
    )

    full_response = resp.choices[0].message.content.strip().upper()

    # ✅ Extract only BUY / SELL / HOLD
    match = re.search(r"\b(BUY|SELL|HOLD)\b", full_response)
    if not match:
        print(f"⚠️ Unexpected GPT response: {full_response}")
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
) -> Dict[str, Any]:
    """
    GPT meta-strategy:
    - Inputs: outputs of other strategies at bar i (entries/exits), current position, and last N hours OHLCV (15m bars).
    - Logic:
      * Only query GPT when at least one strategy signal has changed.
      * Backtest mode uses given OHLCV (take last N hours).
      * Live mode fetches OHLCV from Binance API.
    """
    n = len(ohlcv)
    entries = pd.Series(False, index=ohlcv.index, name="entries")
    exits = pd.Series(False, index=ohlcv.index, name="exits")

    in_pos = False
    last_signal = None
    decisions_count = 0

    for i in range(n):
        strat_signals = {}
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

        if params.provider == "openai" and changed:
            if params.mode == "backtest":
                bars_for_context = min((params.context_hours * 60) // 15, i + 1)
                ohlcv_last_context = ohlcv.iloc[i - bars_for_context + 1 : i + 1].copy()
                if "timestamp" not in ohlcv_last_context.columns:
                    ohlcv_last_context = ohlcv_last_context.copy()
                    ohlcv_last_context["timestamp"] = ohlcv_last_context.index
            else:
                ohlcv_last_context = _fetch_last_context_ohlcv(symbol=symbol, context_hours=params.context_hours)

            prompt = _make_prompt(symbol, strat_signals, ohlcv_last_context, in_pos, params.context_hours)
            signal = _query_openai(prompt)
            last_signal = signal
            decisions_count += 1

            print(f"GPT decided {signal} at {ohlcv.index[i]}")

        elif params.provider == "mock" and changed:
            last_signal = random.choice(["BUY", "SELL", "HOLD"])
            decisions_count += 1
            print(f"MOCK GPT decided {last_signal} at {ohlcv.index[i]}")

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
    }
