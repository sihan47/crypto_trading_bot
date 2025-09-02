from dataclasses import dataclass
import pandas as pd
import numpy as np
import vectorbt as vbt


@dataclass
class RSIParams:
    window: int = 14
    lower: int = 30
    upper: int = 70


def run_rsi_strategy(close: pd.Series, params: RSIParams):
    """
    RSI overbought/oversold strategy.
    Buy when RSI < lower, sell when RSI > upper.
    """
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series.")

    # Compute RSI
    rsi = vbt.RSI.run(close, window=params.window).rsi

    # Signals
    entries = (rsi < params.lower).astype(bool)
    exits = (rsi > params.upper).astype(bool)

    # Fill NaN & align
    entries = entries.fillna(False).reindex(close.index, fill_value=False)
    exits = exits.fillna(False).reindex(close.index, fill_value=False)

    # Debug
    print(f"[DEBUG] RSI({params.window}, {params.lower}/{params.upper}) "
          f"signals: entries={int(entries.sum())}, exits={int(exits.sum())}")

    # Ensure alignment
    assert entries.index.equals(close.index)
    assert exits.index.equals(close.index)

    return {
        "signal_long": entries.rename("entries"),
        "signal_exit": exits.rename("exits"),
        "indicators": {"rsi": rsi},
        "params": {"window": params.window, "lower": params.lower, "upper": params.upper},
    }
