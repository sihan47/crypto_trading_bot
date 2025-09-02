import os
import sqlite3
import pandas as pd
from typing import Optional
from backtesting.utils import resample_ohlcv

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "local_data.db")


# === DB Helpers ===
def get_connection():
    return sqlite3.connect(DB_PATH)


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
def update_data(symbol: str, interval: str):
    """Download or update historical data from Binance (placeholder)."""
    print(f"⚠️ update_data not implemented: would fetch {symbol} {interval} data from Binance")


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

    args = parser.parse_args()

    if args.action == "update":
        update_data(args.symbol, args.interval)

    elif args.action == "load":
        df = load_data(args.symbol, args.interval)
        print(df.head())
        print(f"✅ Loaded {len(df)} rows for {args.symbol} {args.interval}")
        print(f"📅 Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
