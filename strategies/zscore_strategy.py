from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ZScoreParams:
    window: int = 100      # rolling window for mean/std
    entry_z: float = 1.5   # enter when |zscore| exceeds this
    exit_z: float = 0.5    # exit when |zscore| falls below this (mean reversion complete)


def run_zscore_strategy(close: pd.Series, params: ZScoreParams) -> dict:
    """
    Z-score mean reversion strategy.
    Suited for ratio pairs (e.g. BNBBTC) where price oscillates around a rolling mean.

    BUY  when zscore < -entry_z  (price unusually low, expect reversion up)
    SELL when zscore >  entry_z  (price unusually high, expect reversion down)
    """
    mean = close.rolling(params.window).mean()
    std = close.rolling(params.window).std(ddof=0)
    zscore = (close - mean) / std.replace(0, np.nan)

    entries = zscore < -params.entry_z
    exits = zscore > params.exit_z

    return {
        "strategy": "zscore",
        "params": {
            "window": params.window,
            "entry_z": params.entry_z,
            "exit_z": params.exit_z,
        },
        "entries": entries.fillna(False),
        "exits": exits.fillna(False),
        "indicators": {
            "zscore": zscore,
            "mean": mean,
            "std": std,
        },
    }
