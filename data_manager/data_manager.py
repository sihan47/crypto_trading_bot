import os
import sqlite3
from typing import Optional

import pandas as pd
import yaml
from binance.client import Client
from dotenv import load_dotenv

from backtesting.utils import resample_ohlcv

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "local_data.db")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

load_dotenv()


# === Config Helpers ===

def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: failed to parse {CONFIG_PATH}: {exc}")
        return {}


def _load_testnet_flag(default: bool = False) -> bool:
    value = _load_config().get("trading", {}).get("testnet")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


# === DB Helpers ===
def get_connection():
    return sqlite3.connect(DB_PATH)





def _create_client(testnet: Optional[bool] = None) -> Client:
    if testnet is None:
        testnet = _load_testnet_flag(default=False)
    api_key = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("BINANCE_API_KEY and BINANCE_SECRET_KEY must be set before updating data")
    return Client(api_key, secret_key, testnet=testnet)


def _latest_timestamp(symbol: str, interval: str) -> Optional[int]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(timestamp) FROM klines WHERE symbol=? AND interval=?",
            (symbol.upper(), interval)
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else None
    finally:
        conn.close()


def _to_milliseconds(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    ts = pd.Timestamp(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def init_db():
    """Initialize the database with klines table if not exists."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS klines (
            timestamp INTEGER,
            symbol TEXT,
            interval TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (timestamp, symbol, interval)
        )
        """
    )
    conn.commit()
    conn.close()


# === Core Functions ===
def update_data(
    symbol: str,
    interval: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: Optional[int] = None,
    testnet: Optional[bool] = None,
) -> int:
    """Download or update historical klines from Binance into the local SQLite DB."""
    symbol = symbol.upper()
    init_db()

    client = _create_client(testnet=testnet)

    start_ms = _to_milliseconds(start)
    end_ms = _to_milliseconds(end)

    if start_ms is None:
        latest = _latest_timestamp(symbol, interval)
        if latest:
            start_ms = latest + 1
        else:
            start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=365)).timestamp() * 1000)

    if end_ms is not None and end_ms <= start_ms:
        return 0

    rows = []
    inserted = 0
    batch_limit = 1000
    current_start = start_ms

    conn = get_connection()
    cursor = conn.cursor()
    try:
        while True:
            if limit is not None:
                remaining = limit - inserted
                if remaining <= 0:
                    break
                batch_size = min(batch_limit, remaining)
            else:
                batch_size = batch_limit

            params = dict(symbol=symbol, interval=interval, limit=batch_size, startTime=current_start)
            if end_ms is not None:
                params['endTime'] = end_ms

            klines = client.get_klines(**params)
            if not klines:
                break

            for kline in klines:
                rows.append(
                    (
                        int(kline[0]),
                        symbol,
                        interval,
                        float(kline[1]),
                        float(kline[2]),
                        float(kline[3]),
                        float(kline[4]),
                        float(kline[5]),
                    )
                )

            inserted += len(klines)
            current_start = klines[-1][0] + 1

            if len(klines) < batch_size:
                break

        if not rows:
            return 0

        cursor.executemany(
            """
            INSERT OR REPLACE INTO klines
            (timestamp, symbol, interval, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return inserted
    finally:
        conn.close()



def load_data(symbol: str, interval: str) -> pd.DataFrame:
    """Load historical data for a given symbol and interval."""
    init_db()
    conn = get_connection()
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM klines
        WHERE symbol=? AND interval=?
        ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, params=(symbol.upper(), interval))
    conn.close()

    if df.empty:
        raise ValueError(f"No data found for {symbol} {interval}. Did you run update first?")

    # convert ms to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# === Unified Access for Research/Backtest ===
def load_1m_ohlcv(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw 1m OHLCV from local SQLite using existing load_data().
    start/end are optional ISO date strings; trimming is done after load.
    """
    df = load_data(symbol, "1m")
    df = df.rename(columns={"timestamp": "date"}).set_index("date")
    df.index = pd.to_datetime(df.index, utc=True)

    if start is not None:
        df = df[df.index >= pd.to_datetime(start, utc=True)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end, utc=True)]
    return df


def get_ohlcv(symbol: str, start: Optional[str], end: Optional[str], timeframe: str) -> pd.DataFrame:
    """
    Unified access: always load 1m OHLCV and resample to target timeframe.
    Returns DataFrame with columns: ['open','high','low','close','volume'].
    """
    df_1m = load_1m_ohlcv(symbol, start=start, end=end)
    df_tf = resample_ohlcv(df_1m, timeframe)
    return df_tf


# === CLI ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Manager for local SQLite DB")
    parser.add_argument("action", choices=["update", "load"], help="update or load")
    parser.add_argument("symbol", type=str, help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("interval", type=str, help="Interval, e.g. 1m, 5m, 1h")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD or ISO timestamp)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD or ISO timestamp)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of klines to download")
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet keys")

    args = parser.parse_args()

    if args.action == "update":
        rows = update_data(
            args.symbol,
            args.interval,
            start=args.start,
            end=args.end,
            limit=args.limit,
            testnet=True if args.testnet else None,
        )
        print(f"Inserted/updated {rows} rows for {args.symbol.upper()} {args.interval}")
    else:
        df = load_data(args.symbol, args.interval)
        print(df.head())
        print(f"Loaded {len(df)} rows for {args.symbol} {args.interval}")
        print(f"Date range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
