import yaml
from strategies.sma_strategy import generate_sma_signal
from strategies.gpt_strategy import generate_gpt_signal

def load_strategy(config_file="config.yaml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    name = config["strategy"]["name"]
    params = config["strategy"].get("params", {})

    if name == "sma":
        return lambda df: generate_sma_signal(df,
                                              short_window=params.get("short_window", 5),
                                              long_window=params.get("long_window", 20))
    elif name == "gpt":
        return lambda df: generate_gpt_signal(df,
                                              prompt=params.get("prompt", "Decide BUY, SELL or HOLD."))
    else:
        raise ValueError(f"Unknown strategy: {name}")
