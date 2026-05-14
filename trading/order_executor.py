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


def _parse_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _load_testnet_flag(default: bool = True) -> bool:
    env_value = os.getenv("BINANCE_TESTNET")
    if env_value is not None:
        return _parse_bool(env_value, default)

    value = _load_config().get("trading", {}).get("testnet")
    return _parse_bool(value, default)


load_dotenv()
_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("BINANCE_API_KEY and BINANCE_SECRET_KEY must be set before trading")

    _client = Client(api_key, secret_key, testnet=_load_testnet_flag())
    return _client


def assets_from_symbol(symbol: str) -> list[str]:
    """Derive base and quote asset names from a Binance symbol string."""
    for quote in ("USDT", "BUSD", "BTC", "ETH", "BNB"):
        if symbol.upper().endswith(quote):
            base = symbol.upper()[: -len(quote)]
            return [base, quote]
    return ["BTC", "USDT"]


def get_balances(assets: list[str] | None = None):
    """Return free balances for the requested assets (default: BTC + USDT)."""
    if assets is None:
        assets = ["BTC", "USDT"]
    client = _get_client()
    result = {}
    for asset in assets:
        raw = client.get_asset_balance(asset=asset)
        result[asset] = float(raw["free"]) if raw else 0.0
    return result


def get_portfolio_value(assets: list[str]) -> dict:
    """
    Compute total account value in USDT-equivalent across the given assets.
    Returns: {
        "total_usdt": float,
        "breakdown": {asset: {"balance": float, "price_usdt": float, "value_usdt": float}}
    }
    """
    client = _get_client()
    balances = get_balances(assets=assets)

    total = 0.0
    breakdown = {}
    for asset, amount in balances.items():
        if asset == "USDT":
            price = 1.0
            value = amount
        else:
            try:
                ticker = client.get_symbol_ticker(symbol=f"{asset}USDT")
                price = float(ticker["price"])
                value = amount * price
            except Exception:
                price = 0.0
                value = 0.0
        breakdown[asset] = {
            "balance": round(amount, 8),
            "price_usdt": round(price, 4),
            "value_usdt": round(value, 2),
        }
        total += value

    return {"total_usdt": round(total, 2), "breakdown": breakdown}


def execute_order(signal, symbol=None, quantity=None):
    if not symbol:
        raise ValueError('symbol must be provided (e.g. "BTCUSDT")')
    if quantity is None:
        raise ValueError('order quantity must be provided')
    if quantity <= 0:
        trade_logger.error(f'Invalid quantity requested: {quantity}')
        return None
    client = _get_client()

    pair_assets = assets_from_symbol(symbol)
    pf_before = get_portfolio_value(pair_assets)
    try:
        if signal == "BUY":
            order = client.create_order(
                symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=quantity
            )
            pf_after = get_portfolio_value(pair_assets)
            trade_logger.success(
                f"BUY {symbol} | Before total≈${pf_before['total_usdt']:.2f} {pf_before['breakdown']} "
                f"| After total≈${pf_after['total_usdt']:.2f} {pf_after['breakdown']}"
            )
            return order
        elif signal == "SELL":
            order = client.create_order(
                symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=quantity
            )
            pf_after = get_portfolio_value(pair_assets)
            trade_logger.success(
                f"SELL {symbol} | Before total≈${pf_before['total_usdt']:.2f} {pf_before['breakdown']} "
                f"| After total≈${pf_after['total_usdt']:.2f} {pf_after['breakdown']}"
            )
            return order
        else:
            trade_logger.info(f"HOLD | Total≈${pf_before['total_usdt']:.2f} {pf_before['breakdown']}")
            return None
    except Exception as e:
        trade_logger.error(f"Order failed: {e}")
        return None
