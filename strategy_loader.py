import yaml
import json
from pathlib import Path

from strategies.sma_strategy import run_sma_strategy, SMAParams
from strategies.rsi_strategy import run_rsi_strategy, RSIParams
from strategies.macd_strategy import run_macd_strategy, MACDParams
from strategies.bollinger_strategy import run_bollinger_strategy, BollingerParams
from strategies.gpt_strategy import run_gpt_strategy, GPTParams


# === Load best_params.json if exists ===
def load_best_params():
    path = Path(__file__).resolve().parent / "research" / "best_params.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


BEST_PARAMS = load_best_params()


def get_strategy_params(symbol: str, timeframe: str, strat_name: str, default: dict):
    """Return params from best_params.json if available, else from config.yaml, else fallback default."""
    key = f"{symbol}_{timeframe}_{strat_name}"
    if key in BEST_PARAMS:
        return BEST_PARAMS[key]
    return default


def load_strategy(config_file: str):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    strat_cfg = config.get("strategy", {})
    name = strat_cfg.get("name", "sma").lower()

    # === Data defaults ===
    data_cfg = config.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    timeframe = data_cfg.get("timeframe", "15m")

    if name == "sma":
        defaults = {"fast": 10, "slow": 50}
        params = strat_cfg.get("params", get_strategy_params(symbol, timeframe, "sma", defaults))
        sma_params = SMAParams(**params)

        def sma_func(df, **kwargs):
            return run_sma_strategy(df["close"], sma_params)

        return sma_func

    elif name == "rsi":
        defaults = {"window": 14, "lower": 30, "upper": 70}
        params = strat_cfg.get("params", get_strategy_params(symbol, timeframe, "rsi", defaults))
        rsi_params = RSIParams(**params)

        def rsi_func(df, **kwargs):
            return run_rsi_strategy(df["close"], rsi_params)

        return rsi_func

    elif name == "macd":
        defaults = {"fast": 12, "slow": 26, "signal": 9}
        params = strat_cfg.get("params", get_strategy_params(symbol, timeframe, "macd", defaults))
        macd_params = MACDParams(**params)

        def macd_func(df, **kwargs):
            return run_macd_strategy(df["close"], macd_params)

        return macd_func

    elif name == "bollinger":
        defaults = {"window": 20, "std": 2}
        params = strat_cfg.get("params", get_strategy_params(symbol, timeframe, "bollinger", defaults))
        boll_params = BollingerParams(**params)

        def boll_func(df, **kwargs):
            return run_bollinger_strategy(df["close"], boll_params)

        return boll_func

    elif name == "gpt":
        params = strat_cfg.get("params", {})
        gpt_params = GPTParams(**params)

        def gpt_func(df, force: bool = False, **kwargs):
            # === 子策略用 best_params.json（或 fallback）===
            sma_params = SMAParams(**get_strategy_params(symbol, timeframe, "sma", {"fast": 10, "slow": 50}))
            rsi_params = RSIParams(**get_strategy_params(symbol, timeframe, "rsi", {"window": 14, "lower": 30, "upper": 70}))
            macd_params = MACDParams(**get_strategy_params(symbol, timeframe, "macd", {"fast": 12, "slow": 26, "signal": 9}))
            boll_params = BollingerParams(**get_strategy_params(symbol, timeframe, "bollinger", {"window": 20, "std": 2}))

            sma_out = run_sma_strategy(df["close"], sma_params)
            rsi_out = run_rsi_strategy(df["close"], rsi_params)
            macd_out = run_macd_strategy(df["close"], macd_params)
            boll_out = run_bollinger_strategy(df["close"], boll_params)

            other_results = {
                "sma": sma_out,
                "rsi": rsi_out,
                "macd": macd_out,
                "bollinger": boll_out,
            }

            return run_gpt_strategy(symbol, df, other_results, timeframe, gpt_params, force=force)

        return gpt_func

    else:
        raise ValueError(f"Unknown strategy name: {name}")
