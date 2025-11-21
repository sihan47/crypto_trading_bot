"""Quick helper to inspect Binance balances manually."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import yaml
from binance.client import Client
from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
DEFAULT_ASSETS = ("USDT", "BTC")
VALID_ACCOUNT_TYPES = {"spot", "futures"}
ALIASES = {
    "future": "futures",
    "perp": "futures",
    "perps": "futures",
    "perpetual": "futures",
}

load_dotenv()


def _load_config() -> Dict[str, object]:
    try:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: failed to parse {CONFIG_PATH}: {exc}")
        return {}


def _parse_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _load_account_type(default: str = "spot") -> str:
    data = _load_config()
    raw_value = (data.get("trading", {}).get("account_type") or default).strip().lower()
    account_type = ALIASES.get(raw_value, raw_value)
    if account_type not in VALID_ACCOUNT_TYPES:
        print(f"Warning: unknown account_type '{raw_value}', defaulting to {default}")
        return default
    return account_type


def _load_testnet_flag(default: bool = True) -> bool:
    env_value = os.getenv("BINANCE_TESTNET")
    if env_value is not None:
        return _parse_bool(env_value, default)

    data = _load_config()
    value = data.get("trading", {}).get("testnet")
    return _parse_bool(value, default)


def _create_client(testnet: bool = True) -> Client:
    api_key = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("BINANCE_API_KEY and BINANCE_SECRET_KEY must be set in the environment")
    return Client(api_key, secret_key, testnet=testnet)


def _fetch_spot_balances(client: Client, assets: Iterable[str]) -> Dict[str, float]:
    balances: Dict[str, float] = {}
    for asset in assets:
        info = client.get_asset_balance(asset=asset)
        balances[asset.upper()] = float(info["free"])
    return balances


def _fetch_futures_balances(client: Client, assets: Iterable[str]) -> Dict[str, float]:
    balances: Dict[str, float] = {a.upper(): 0.0 for a in assets}
    info = client.futures_account_balance()
    for entry in info:
        asset = entry.get("asset", "").upper()
        if asset in balances:
            balances[asset] = float(entry.get("availableBalance", 0.0))
    return balances


def fetch_balances(
    assets: Iterable[str] = DEFAULT_ASSETS,
    testnet: Optional[bool] = None,
) -> Mapping[str, float]:
    """Return free balances for the requested assets using spot or futures endpoints."""
    if testnet is None:
        testnet = _load_testnet_flag()
    client = _create_client(testnet=testnet)
    account_type = _load_account_type()
    if account_type == "futures":
        return _fetch_futures_balances(client, assets)
    return _fetch_spot_balances(client, assets)


if __name__ == "__main__":
    balances = fetch_balances()
    print("Binance balances:")
    for asset, amount in balances.items():
        print(f"  {asset}: {amount}")
