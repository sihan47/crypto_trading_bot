# strategies/gpt_strategy.py

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import os
import re
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from binance.client import Client
from trading.order_executor import get_balances

# Load environment variables
load_dotenv()


@dataclass
class GPTParams:
    provider: str = "openai"     # openai | mock
    mode: str = "live"           # live only (we do not backtest with GPT)
    context_hours: int = 4       # context window in hours (15m bars)
    weight_sma: float = 1.0
    weight_rsi: float = 1.0
    weight_macd: float = 1.0
    weight_bollinger: float = 1.0
    # legacy fields kept for compatibility; not used in live-only mode
    vote_threshold: int = 2
    exit_vote_threshold: int = 1
    hour_momentum_threshold: float = 0.002
    # new: toggle prompt preview
    show_prompt: bool = False


def _load_best_params() -> dict:
    """Load best_params.json, which includes both params and __backtest section."""
    path = Path(__file__).resolve().parents[1] / "research" / "best_params.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
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


def _fetch_last_context_ohlcv(symbol: str = "BTCUSDT", interval: str = "15m", context_hours: int = 4) -> pd.DataFrame:
    """
    Fetch last N hours of OHLCV from Binance (15m bars).
    Use mainnet for market data (testnet has no historical klines).
    """
    api_key = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY")
    client = Client(api_key, secret_key)  # do not set testnet=True for klines

    limit = max(1, (context_hours * 60) // 15)
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "_1", "_2", "_3", "_4", "_5", "_6"
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"])
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df


def _make_prompt(
    symbol: str,
    strat_signals: Dict[str, str],
    ohlcv_last_context: pd.DataFrame,
    in_pos: bool,
    context_hours: int,
    timeframe: str,
) -> str:
    """Construct a prompt for GPT using strategy signals, position status, and last N hours OHLCV (15m bars)."""
    best_map = _load_best_params()

    lines = []
    lines.append(f"You are a trading assistant for {symbol}. Decide BUY, SELL, or HOLD BTC.")
    lines.append(f"\n--- Current Position ---\n {get_balances()}")
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


def _query_openai(prompt: str, show: bool) -> str:
    """Send prompt to OpenAI GPT model and return its decision (BUY/SELL/HOLD)."""
    if show:
        print("\n=== GPT Prompt Preview ===")
        print(prompt[:2000])
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
    decision = match.group(1) if match else "HOLD"

    print(f"\n=== GPT Decision ===\n{decision}\n====================\n")
    return decision


def _signals_at_last_bar(other_results: Dict[str, Dict[str, pd.Series]]) -> Dict[str, str]:
    """Convert base strategies entries/exits to BUY/SELL/HOLD at the last closed bar."""
    out: Dict[str, str] = {}
    for name, res in other_results.items():
        sig = "HOLD"
        try:
            if bool(res["entries"].iloc[-1]):
                sig = "BUY"
            elif bool(res["exits"].iloc[-1]):
                sig = "SELL"
        except Exception:
            sig = "HOLD"
        out[name] = sig
    return out


def run_gpt_strategy(
    symbol: str,
    ohlcv: pd.DataFrame,
    other_results: Dict[str, Dict[str, pd.Series]],
    timeframe: str,
    params: GPTParams,
    initial_run: bool = False,  # kept for compatibility; not used in live-only mode
) -> Dict[str, Any]:
    """
    Live-only GPT meta-strategy:
      - Looks at the LAST closed bar only.
      - Builds a prompt from base-strategy signals and the last N hours of 15m OHLCV.
      - Queries OpenAI (or mock) exactly once per call.
      - Does NOT place orders; order execution is handled by the caller.
    Returns a dict with "last_signal" plus empty entries/exits (for compatibility).
    """
    # Determine position from balances (live)
    in_pos = False
    try:
        from trading.order_executor import get_balances
        balances = get_balances()
        base = symbol.replace("USDT", "")
        in_pos = balances.get(base, 0) > 0
    except Exception:
        in_pos = False

    # Convert base strategy outputs to simple signals at last bar
    strat_signals = _signals_at_last_bar(other_results)

    # Build context OHLCV (15m bars for last N hours)
    ohlcv_last_context = _fetch_last_context_ohlcv(
        symbol=symbol, interval=timeframe, context_hours=params.context_hours
    )

    # Create prompt
    prompt = _make_prompt(
        symbol, strat_signals, ohlcv_last_context, in_pos, params.context_hours, timeframe
    )

    # Decide using provider
    provider = (params.provider or "openai").lower()
    if provider == "openai":
        decision = _query_openai(prompt, show=params.show_prompt)
    else:
        if params.show_prompt:
            print("\n=== GPT Prompt Preview (mock) ===")
            print(prompt[:2000])
            print("==========================\n")
        decision = random.choice(["BUY", "SELL", "HOLD"])
        print(f"\n=== GPT Decision (mock) ===\n{decision}\n====================\n")

    # Return a compatible structure
    idx = ohlcv.index if isinstance(ohlcv.index, pd.Index) else pd.RangeIndex(len(ohlcv))
    entries = pd.Series(False, index=idx, name="entries")
    exits = pd.Series(False, index=idx, name="exits")

    return {
        "entries": entries,
        "exits": exits,
        "params": vars(params),
        "stats": {"decisions_count": 1},
        "last_signal": decision,
    }
