import os, pathlib
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
from loguru import logger

pathlib.Path("logs").mkdir(exist_ok=True)

# Dedicated logger for TRADING
trade_logger = logger.bind(tag="TRADE")
logger.add("logs/trading.log",
           rotation="1 day",
           retention="30 days",
           level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | [{extra[tag]}] {message}",
           filter=lambda record: record["extra"].get("tag") == "TRADE")

# Load API keys
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
secret_key = os.getenv("BINANCE_SECRET_KEY")
client = Client(api_key, secret_key, testnet=True)

def get_balances():
    usdt = client.get_asset_balance(asset="USDT")
    btc = client.get_asset_balance(asset="BTC")
    return {"USDT": float(usdt["free"]), "BTC": float(btc["free"])}

def execute_order(signal, symbol="BTCUSDT", quantity=0.001):
    balances_before = get_balances()
    try:
        if signal == "BUY":
            order = client.create_order(
                symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=quantity
            )
            balances_after = get_balances()
            trade_logger.success(f"BUY executed | Before: {balances_before} | After: {balances_after}")
            return order
        elif signal == "SELL":
            order = client.create_order(
                symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=quantity
            )
            balances_after = get_balances()
            trade_logger.success(f"SELL executed | Before: {balances_before} | After: {balances_after}")
            return order
        else:
            trade_logger.info(f"HOLD | Current balances: {balances_before}")
            return None
    except Exception as e:
        trade_logger.error(f"Order failed: {e}")
        return None
