from dataclasses import dataclass
import pandas as pd
import pandas_ta as ta


@dataclass
class BollingerParams:
    window: int = 20
    std: int = 2


def run_bollinger_strategy(close: pd.Series, params: BollingerParams):
    bbands = ta.bbands(close, length=params.window, std=params.std)
    lower = bbands[f"BBL_{params.window}_{params.std}.0"]
    upper = bbands[f"BBU_{params.window}_{params.std}.0"]

    entries = close < lower
    exits = close > upper

    return {
        "strategy": "bollinger",
        "params": {"window": params.window, "std": params.std},
        "entries": entries.fillna(False),
        "exits": exits.fillna(False),
        "indicators": {"lower": lower, "upper": upper},
    }
