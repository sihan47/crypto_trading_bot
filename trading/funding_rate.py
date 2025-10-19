"""Utility for fetching Binance perpetual funding rates and annualized costs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

API_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
FU_NGS_PER_DAY = 3  # Binance futures funds every 8 hours
DAYS_PER_YEAR = 365


class FundingRateError(RuntimeError):
    """Raised when the funding rate endpoint cannot be retrieved."""


def fetch_funding_rate(symbol: str = "BTCUSDT", timeout: int = 10) -> Dict[str, float]:
    """Fetch the latest funding data for a perpetual futures symbol.

    Returns a dictionary containing:
      - symbol: uppercase symbol (e.g. BTCUSDT)
      - rate: last funding rate as a decimal (e.g. 0.0001 => 0.01%)
      - mark_price: latest mark price
      - timestamp: aware UTC datetime of the API timestamp
      - next_funding_time: aware UTC datetime when the next cycle applies
    """
    params = {"symbol": symbol.upper()}
    try:
        resp = requests.get(API_URL, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network error path
        raise FundingRateError(f"Failed to fetch funding rate for {symbol}: {exc}") from exc

    payload = resp.json()
    try:
        rate = float(payload["lastFundingRate"])
        mark_price = float(payload["markPrice"])
        current_ts = datetime.fromtimestamp(payload["time"] / 1000, tz=timezone.utc)
        next_ts = datetime.fromtimestamp(payload["nextFundingTime"] / 1000, tz=timezone.utc)
    except (KeyError, ValueError, TypeError) as exc:
        raise FundingRateError(f"Unexpected response structure for {symbol}: {payload}") from exc

    return {
        "symbol": symbol.upper(),
        "rate": rate,
        "mark_price": mark_price,
        "timestamp": current_ts,
        "next_funding_time": next_ts,
    }


def annualized_from_rate(rate: float) -> float:
    """Convert a single-period funding rate (8h) into an annualized decimal."""
    return rate * FU_NGS_PER_DAY * DAYS_PER_YEAR


def format_summary(data: Dict[str, float], precision: int = 4) -> str:
    annual_decimal = annualized_from_rate(data["rate"])
    annual_percent = annual_decimal * 100
    period_percent = data["rate"] * 100
    return (
        f"Symbol        : {data['symbol']}\n"
        f"Mark Price    : {data['mark_price']:.2f}\n"
        f"Last Funding  : {period_percent:.{precision}f}% per 8h\n"
        f"Annualized    : {annual_percent:.{precision}f}%\n"
        f"Timestamp UTC : {data['timestamp'].isoformat()}\n"
        f"Next Funding  : {data['next_funding_time'].isoformat()}"
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Binance perpetual funding rate and annualized cost")
    parser.add_argument("symbol", nargs="?", default="BTCUSDT", help="Perpetual futures symbol (default: BTCUSDT)")
    parser.add_argument("--precision", type=int, default=4, help="Decimal precision for percentage output")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout in seconds")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    data = fetch_funding_rate(args.symbol, timeout=args.timeout)
    print(format_summary(data, precision=args.precision))


if __name__ == "__main__":
    main()
