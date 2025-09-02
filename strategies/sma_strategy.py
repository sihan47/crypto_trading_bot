from dataclasses import dataclass
import pandas as pd

@dataclass
class SMAParams:
    fast: int = 10
    slow: int = 50

def run_sma_strategy(close: pd.Series, params: SMAParams):
    fast_ma = close.rolling(params.fast).mean()
    slow_ma = close.rolling(params.slow).mean()

    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits   = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    return {
        "strategy": "sma",
        "params": {"fast": params.fast, "slow": params.slow},
        "entries": entries.fillna(False),
        "exits": exits.fillna(False),
        "indicators": {"fast_ma": fast_ma, "slow_ma": slow_ma},
    }
