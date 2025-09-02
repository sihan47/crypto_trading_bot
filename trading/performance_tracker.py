import os, json, pathlib
from dotenv import load_dotenv
from binance.client import Client
from loguru import logger
from datetime import datetime

pathlib.Path("logs").mkdir(exist_ok=True)

# Dedicated logger for PERFORMANCE
perf_logger = logger.bind(tag="PERF")
logger.add("logs/performance.log",
           rotation="1 week",
           retention="90 days",
           level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | [{extra[tag]}] {message}",
           filter=lambda record: record["extra"].get("tag") == "PERF")

# Load API keys
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
secret_key = os.getenv("BINANCE_SECRET_KEY")
client = Client(api_key, secret_key, testnet=True)

STATE_FILE = "logs/performance_state.json"

def get_balances():
    usdt = client.get_asset_balance(asset="USDT")
    btc = client.get_asset_balance(asset="BTC")
    return {"USDT": float(usdt["free"]), "BTC": float(btc["free"])}

def get_total_value():
    balances = get_balances()
    price = float(client.get_symbol_ticker(symbol="BTCUSDT")["price"])
    return balances["USDT"] + balances["BTC"] * price, balances, price

def load_state():
    return json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}

def save_state(state):
    json.dump(state, open(STATE_FILE, "w"))

def log_daily_performance(trades_today, wins_today):
    total_value, balances, price = get_total_value()
    state = load_state()
    yesterday_value = state.get("total_value", total_value)
    pnl = total_value - yesterday_value
    win_rate = (wins_today / trades_today * 100) if trades_today > 0 else 0

    perf_logger.info(
        f"📊 Daily Performance | {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Total equity: {total_value:.2f} USDT (BTC={balances['BTC']}, USDT={balances['USDT']}, Price={price:.2f})\n"
        f"PnL vs yesterday: {pnl:.2f} USDT\n"
        f"Trades: {trades_today}, Win rate: {win_rate:.2f}%"
    )
    save_state({"total_value": total_value})
