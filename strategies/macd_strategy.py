from dataclasses import dataclass
import pandas as pd
import pandas_ta as ta


@dataclass
class MACDParams:
    fast: int = 12
    slow: int = 26
    signal: int = 9


def run_macd_strategy(close: pd.Series, params: MACDParams):
    macd_df = ta.macd(close, fast=params.fast, slow=params.slow, signal=params.signal)
    macd_line = macd_df[f"MACD_{params.fast}_{params.slow}_{params.signal}"]
    signal_line = macd_df[f"MACDs_{params.fast}_{params.slow}_{params.signal}"]

    
    entries = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    exits = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    return {
        "strategy": "macd",
        "params": {"fast": params.fast, "slow": params.slow, "signal": params.signal},
        "entries": entries.fillna(False),
        "exits": exits.fillna(False),
        "indicators": {"macd": macd_line, "signal": signal_line},
    }
