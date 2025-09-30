from pathlib import Path
import textwrap

path = Path('data_manager/data_manager.py')
text = path.read_text(encoding='utf-8')

# inject new imports after existing ones
old_imports = "import os\nimport sqlite3\nimport pandas as pd\nfrom typing import Optional\nfrom backtesting.utils import resample_ohlcv\n"
if old_imports not in text:
    raise SystemExit('expected import block not found')
new_imports = "import os\nimport sqlite3\nfrom typing import Optional, Tuple\n\nimport pandas as pd\nfrom binance.client import Client\nfrom dotenv import load_dotenv\n\nfrom backtesting.utils import resample_ohlcv\n"
text = text.replace(old_imports, new_imports, 1)

# add load_dotenv call after constants maybe
marker = "os.makedirs(DATA_DIR, exist_ok=True)\n\nDB_PATH = os.path.join(DATA_DIR, \"local_data.db\")\n\n\n"
if marker not in text:
    raise SystemExit('expected DATA_DIR marker missing')
replacement = marker + "load_dotenv()\n\n\n"
text = text.replace(marker, replacement, 1)

# helper functions to insert before init_db maybe
insert_point = "def init_db():\n    \"\"\"Initialize the database with klines table if not exists.\"\"\"\n"
helpers = textwrap.dedent('''

def _create_client(testnet: bool = False) -> Client:
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


''')
text = text.replace(insert_point, helpers + insert_point, 1)

# replace update_data definition
old_func = "def update_data(symbol: str, interval: str):\n    \"\"\"Download or update historical data from Binance (placeholder).\"\"\"\n    print(f\"?? update_data not implemented: would fetch {symbol} {interval} data from Binance\")\n\n\n"
if old_func not in text:
    raise SystemExit('old update_data function not found')
new_func = textwrap.dedent('''
def update_data(
    symbol: str,
    interval: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: Optional[int] = None,
    testnet: bool = False,
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
            # default to roughly one year of history
            start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=365)).timestamp() * 1000)

    if end_ms is not None and end_ms <= start_ms:
        return 0

    batch_limit = 1000
    inserted = 0
    rows = []
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
                params["endTime"] = end_ms

            klines = client.get_klines(**params)
            if not klines:
                break

            for k in klines:
                rows.append(
                    (
                        int(k[0]),
                        symbol,
                        interval,
                        float(k[1]),
                        float(k[2]),
                        float(k[3]),
                        float(k[4]),
                        float(k[5]),
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


''')
text = text.replace(old_func, new_func, 1)

# update load_data to uppercase interval? maybe existing; leave.

# modify CLI portion to add argparse options
old_cli = "    parser = argparse.ArgumentParser(description=\"Data Manager for local SQLite DB\")\n    parser.add_argument(\"action\", choices=[\"update\", \"load\"], help=\"update or load\")\n    parser.add_argument(\"symbol\", type=str, help=\"Trading pair, e.g. BTCUSDT\")\n    parser.add_argument(\"interval\", type=str, help=\"Interval, e.g. 1m, 5m, 1h\")\n\n    args = parser.parse_args()\n\n    if args.action == \"update\":\n        update_data(args.symbol, args.interval)\n\n    elif args.action == \"load\":\n        df = load_data(args.symbol, args.interval)\n        print(df.head())\n        print(f\"??Loaded {len(df)} rows for {args.symbol} {args.interval}\")\n        print(f\"?? Date range: {df['timestamp'].iloc[0]} ??{df['timestamp'].iloc[-1]}\")\n"

new_cli = textwrap.dedent('''
    parser = argparse.ArgumentParser(description="Data Manager for local SQLite DB")
    parser.add_argument("action", choices=["update", "load"], help="update or load")
    parser.add_argument("symbol", type=str, help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("interval", type=str, help="Interval, e.g. 1m, 5m, 1h")
    parser.add_argument("--start", type=str, default=None, help="Start date/time (YYYY-MM-DD or timestamp)")
    parser.add_argument("--end", type=str, default=None, help="End date/time (YYYY-MM-DD or timestamp)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of klines to download")
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet (default: mainnet)")

    args = parser.parse_args()

    if args.action == "update":
        rows = update_data(
            args.symbol,
            args.interval,
            start=args.start,
            end=args.end,
            limit=args.limit,
            testnet=args.testnet,
        )
        print(f"Inserted/updated {rows} rows for {args.symbol.upper()} {args.interval}")

    elif args.action == "load":
        df = load_data(args.symbol, args.interval)
        print(df.head())
        print(f"Loaded {len(df)} rows for {args.symbol} {args.interval}")
        print(f"Date range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
''')
text = text.replace(old_cli, new_cli, 1)

path.write_text(text, encoding='utf-8')
