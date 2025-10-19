import os
import pathlib
from typing import Any

import yaml
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
from loguru import logger

pathlib.Path("logs").mkdir(exist_ok=True)

# Dedicated logger for TRADING
trade_logger = logger.bind(tag="TRADE")
logger.add(
    "logs/trading.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | [{extra[tag]}] {message}",
    filter=lambda record: record["extra"].get("tag") == "TRADE",
)

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "config.yaml"


def _load_config() -> dict[str, Any]:
    try:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        trade_logger.warning(f"Failed to parse {CONFIG_PATH}: {exc}")
        return {}


def _load_testnet_flag(default: bool = True) -> bool:
    value = _load_config().get("trading", {}).get("testnet")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


# Load API keys
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
secret_key = os.getenv("BINANCE_SECRET_KEY")
if not api_key or not secret_key:
    raise RuntimeError("BINANCE_API_KEY and BINANCE_SECRET_KEY must be set before trading")
client = Client(api_key, secret_key, testnet=_load_testnet_flag())


def get_balances():
    usdt = client.get_asset_balance(asset="USDT")
    btc = client.get_asset_balance(asset="BTC")
    return {"USDT": float(usdt["free"]), "BTC": float(btc["free"])}


def execute_order(signal, symbol="BTCUSDT", quantity=None):
    if quantity is None:
        raise ValueError('order quantity must be provided')
    if quantity <= 0:
        trade_logger.error(f'Invalid quantity requested: {quantity}')
        return None

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
