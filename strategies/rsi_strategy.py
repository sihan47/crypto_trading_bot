from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class RSIParams:
    window: int = 14
    lower: int = 30
    upper: int = 70

def run_rsi_strategy(close: pd.Series, params: RSIParams):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(params.window).mean()
    loss = -delta.clip(upper=0).rolling(params.window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    entries = rsi < params.lower
    exits   = rsi > params.upper

    return {
        "strategy": "rsi",
        "params": {"window": params.window, "lower": params.lower, "upper": params.upper},
        "entries": entries.fillna(False),
        "exits": exits.fillna(False),
        "indicators": {"rsi": rsi},
    }
