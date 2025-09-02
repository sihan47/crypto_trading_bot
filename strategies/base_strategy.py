from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a pd.Series of signals (1=BUY, -1=SELL, 0=HOLD)."""
        pass
