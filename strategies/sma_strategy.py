from dataclasses import dataclass
import pandas as pd
import numpy as np
import vectorbt as vbt

@dataclass
class SMAParams:
    fast: int = 25
    slow: int = 50

def run_sma_strategy(close: pd.Series, params: SMAParams):
    """
    Simple SMA crossover strategy.
    Returns entries/exits signals aligned with close index.
    """
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series.")
    if params.fast >= params.slow:
        raise ValueError("fast window must be strictly less than slow window.")

    # Compute moving averages
    fast_ma = vbt.MA.run(close, window=params.fast).ma
    slow_ma = vbt.MA.run(close, window=params.slow).ma

    # Generate signals
    entries = (fast_ma > slow_ma).astype(bool)
    exits = (fast_ma < slow_ma).astype(bool)

    # Fill NaN with False
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Force alignment with close index
    entries = entries.reindex(close.index, fill_value=False)
    exits = exits.reindex(close.index, fill_value=False)

    # Debug print: number of signals
    print(f"[DEBUG] SMA({params.fast},{params.slow}) signals: "
          f"entries={int(entries.sum())}, exits={int(exits.sum())}")

    # Ensure index alignment
    assert entries.index.equals(close.index), "Entries index mismatch with close"
    assert exits.index.equals(close.index), "Exits index mismatch with close"

    return {
        "signal_long": entries.rename("entries"),
        "signal_exit": exits.rename("exits"),
        "indicators": {"sma_fast": fast_ma, "sma_slow": slow_ma},
        "params": {"fast": params.fast, "slow": params.slow},
    }
