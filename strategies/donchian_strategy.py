from dataclasses import dataclass

import pandas as pd


@dataclass
class DonchianParams:
    window: int = 20


def run_donchian_strategy(df: pd.DataFrame, params: DonchianParams):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    upper = high.rolling(params.window).max().shift(1)
    lower = low.rolling(params.window).min().shift(1)
    mid = (upper + lower) / 2

    entries = (close > upper).fillna(False)
    exits = (close < lower).fillna(False)

    return {
        "strategy": "donchian",
        "params": {"window": params.window},
        "entries": entries,
        "exits": exits,
        "indicators": {
            "upper": upper,
            "lower": lower,
            "mid": mid,
        },
    }
