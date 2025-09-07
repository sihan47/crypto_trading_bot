import os
import sys
import time
import pathlib
from typing import Optional, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv
from binance.client import Client
from loguru import logger

from strategy_loader import load_strategy
from trading.order_executor import execute_order, get_balances

# --- Environment ---
load_dotenv()
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY") is not None)

# --- Logging (console + file) ---
pathlib.Path("logs").mkdir(parents=True, exist_ok=True)
logger.remove()  # remove default sink
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | [BOT] {message}",
)
logger.add(
    "logs/bot.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | [BOT] {message}",
)

# --- Config ---
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f) or {}

strategy_name = (CONFIG.get("strategy", {}).get("name") or CONFIG.get("strategy") or "gpt")
data_cfg = CONFIG.get("data", {}) or {}
symbol = data_cfg.get("symbol", "BTCUSDT")
timeframe = data_cfg.get("timeframe", "15m")
lookback = int(data_cfg.get("lookback", 200))

# --- Binance client (testnet for safety) ---
api_key = os.getenv("BINANCE_API_KEY")
secret_key = os.getenv("BINANCE_SECRET_KEY")
client = Client(api_key, secret_key, testnet=True)

# --- Strategy function from loader (no args per your repo) ---
strategy_func = load_strategy()

_last_processed_bar: Optional[pd.Timestamp] = None


def _tf_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe: {tf}")


def fetch_live_ohlcv(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    klines = client.get_klines(symbol=symbol, interval=interval, limit=max(2, limit))
    df = pd.DataFrame(
        klines,
        columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"
        ],
    )
    df = df[["open_time","open","high","low","close","volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.astype({"open": float,"high": float,"low": float,"close": float,"volume": float})
    df = df.set_index("open_time").sort_index()
    return df


def split_closed_open(df: pd.DataFrame, tf: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    if df.empty:
        return df, None
    minutes = _tf_to_minutes(tf)
    now = pd.Timestamp.now(tz="UTC")
    is_closed = (df.index + pd.Timedelta(minutes=minutes)) <= now
    closed_df = df[is_closed].copy()
    open_bar = df[~is_closed].iloc[-1] if (~is_closed).any() else None
    return closed_df, open_bar


def should_run_for_new_bar(closed_df: pd.DataFrame) -> bool:
    global _last_processed_bar
    if closed_df.empty:
        return False
    last_bar = closed_df.index[-1]
    if _last_processed_bar is None or last_bar > _last_processed_bar:
        _last_processed_bar = last_bar
        return True
    return False


def run_strategy(force: bool = False):
    df_all = fetch_live_ohlcv(symbol, timeframe, limit=max(lookback + 2, 300))
    closed_df, _ = split_closed_open(df_all, timeframe)

    if closed_df.empty:
        logger.warning("No closed bars available.")
        return

    logger.info(f"Data mode: live | source: Binance | bars(closed)={len(closed_df)}")

    # Only run on new closed bar unless forced
    if not force and not should_run_for_new_bar(closed_df):
        return

    # Safety: ensure enough bars for indicators
    if len(closed_df) < 60:
        logger.warning(f"Not enough bars for indicators (got {len(closed_df)}). Skipping.")
        return

    balances_before = get_balances()
    logger.info(f"=== Balances BEFORE decision === {balances_before}")

    # Call strategy with compatibility for optional kwargs
    try:
        result = strategy_func(closed_df, force=force)
    except TypeError:
        try:
            result = strategy_func(closed_df, initial_run=force)
        except TypeError:
            result = strategy_func(closed_df)

    decision = None
    if isinstance(result, dict):
        decision = (result.get("last_signal") or result.get("signal") or "").upper()
    elif isinstance(result, str):
        decision = result.strip().upper()

    logger.info(f"Strategy={strategy_name} | Decision={decision or 'HOLD'} | LastBar={closed_df.index[-1]}")

    if decision in ("BUY", "SELL"):
        order = execute_order(decision, symbol=symbol, quantity=0.001)
        if order:
            logger.success(f"Order executed: {decision}")
        else:
            logger.error(f"Order failed or was not created: {decision}")
        balances_after = get_balances()
        logger.info(f"=== Balances AFTER decision === {balances_after}")


if __name__ == "__main__":
    logger.info(f"🚀 Bot started | strategy={strategy_name} | symbol={symbol} | timeframe={timeframe} | lookback={lookback}")
    logger.info(f"Initial balances: {get_balances()}")
    logger.info("First forced run")
    run_strategy(force=True)

    while True:
        try:
            run_strategy(force=False)
        except Exception as e:
            logger.exception(f"Main loop error: {e}")
        time.sleep(10)
