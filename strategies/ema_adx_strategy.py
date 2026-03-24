from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EMAADXParams:
    fast: int = 21
    slow: int = 55
    adx_window: int = 14
    adx_threshold: float = 18.0


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> dict[str, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean()

    plus_di = 100 * plus_dm.rolling(window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window).mean()

    return {
        "atr": atr,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "adx": adx,
    }


def run_ema_adx_strategy(df: pd.DataFrame, params: EMAADXParams):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    fast_ema = close.ewm(span=params.fast, adjust=False).mean()
    slow_ema = close.ewm(span=params.slow, adjust=False).mean()
    adx_pack = _compute_adx(high, low, close, params.adx_window)
    adx = adx_pack["adx"]

    entries = (
        (fast_ema > slow_ema)
        & (fast_ema.shift(1) <= slow_ema.shift(1))
        & (adx >= params.adx_threshold)
    )
    exits = (
        (fast_ema < slow_ema)
        & (fast_ema.shift(1) >= slow_ema.shift(1))
        & (adx >= params.adx_threshold)
    )

    return {
        "strategy": "ema_adx",
        "params": {
            "fast": params.fast,
            "slow": params.slow,
            "adx_window": params.adx_window,
            "adx_threshold": params.adx_threshold,
        },
        "entries": entries.fillna(False),
        "exits": exits.fillna(False),
        "indicators": {
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "adx": adx_pack["adx"],
            "plus_di": adx_pack["plus_di"],
            "minus_di": adx_pack["minus_di"],
            "atr": adx_pack["atr"],
        },
    }
