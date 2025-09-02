# backtesting/utils.py
import pandas as pd
import re

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input OHLCV missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("OHLCV index must be a pandas.DatetimeIndex (UTC preferred).")
    return df.sort_index()

def _to_pandas_freq(tf: str) -> str:
    """
    Convert timeframe like '1m','5m','1h','1d' into valid pandas freq string.
    """
    if not isinstance(tf, str):
        return tf
    m = re.fullmatch(r"(\d+)([mhdw])", tf.strip().lower())
    if not m:
        return tf
    n, u = m.groups()
    unit_map = {"m": "min", "h": "H", "d": "D", "w": "W"}
    return f"{n}{unit_map[u]}"

def resample_ohlcv(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample 1m OHLCV to target timeframe. Right-closed/right-labeled bars.
    """
    df_1m = ensure_ohlcv(df_1m)

    if timeframe in ("1m", "1T", "1min"):
        return df_1m.copy()

    freq = _to_pandas_freq(timeframe)
    ohlc = df_1m.resample(freq, label="right", closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return ohlc.dropna(how="any")
