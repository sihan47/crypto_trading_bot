from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MeanReversionParams:
    rsi_window: int = 14
    rsi_lower: float = 32.0
    rsi_upper: float = 68.0
    bb_window: int = 20
    bb_std: float = 2.2


def _compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def run_mean_reversion_strategy(close: pd.Series, params: MeanReversionParams):
    rsi = _compute_rsi(close, params.rsi_window)

    basis = close.rolling(params.bb_window).mean()
    std = close.rolling(params.bb_window).std(ddof=0)
    upper = basis + std * params.bb_std
    lower = basis - std * params.bb_std

    entries = (close < lower) & (rsi < params.rsi_lower)
    exits = ((close > basis) & (rsi > 50)) | (close > upper) | (rsi > params.rsi_upper)

    return {
        "strategy": "mean_reversion",
        "params": {
            "rsi_window": params.rsi_window,
            "rsi_lower": params.rsi_lower,
            "rsi_upper": params.rsi_upper,
            "bb_window": params.bb_window,
            "bb_std": params.bb_std,
        },
        "entries": entries.fillna(False),
        "exits": exits.fillna(False),
        "indicators": {
            "rsi": rsi,
            "basis": basis,
            "upper": upper,
            "lower": lower,
            "zscore": (close - basis) / std.replace(0, np.nan),
        },
    }
