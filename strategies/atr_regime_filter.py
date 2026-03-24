from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ATRFilterParams:
    window: int = 14
    percentile_window: int = 96
    max_atr_pct: float = 0.018
    max_percentile: float = 0.85


def run_atr_regime_filter(df: pd.DataFrame, params: ATRFilterParams):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(params.window).mean()
    atr_pct = atr / close.replace(0, np.nan)
    atr_pct_ma = atr_pct.rolling(params.percentile_window).mean()

    atr_rank = atr_pct.rolling(params.percentile_window).apply(
        lambda values: pd.Series(values).rank(pct=True).iloc[-1],
        raw=False,
    )

    allow_trade = (
        (atr_pct <= params.max_atr_pct)
        & (atr_rank <= params.max_percentile)
    ).fillna(False)

    return {
        "strategy": "atr_filter",
        "params": {
            "window": params.window,
            "percentile_window": params.percentile_window,
            "max_atr_pct": params.max_atr_pct,
            "max_percentile": params.max_percentile,
        },
        "entries": pd.Series(False, index=df.index, name="entries"),
        "exits": pd.Series(False, index=df.index, name="exits"),
        "indicators": {
            "atr": atr,
            "atr_pct": atr_pct,
            "atr_pct_ma": atr_pct_ma,
            "atr_rank": atr_rank,
            "allow_trade": allow_trade,
        },
    }
