import os
import time
import schedule
import pathlib
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client
from loguru import logger

from strategy_loader import load_strategy
from trading.order_executor import execute_order, get_balances
from trading.performance_tracker import log_daily_performance
from data_manager.data_manager import get_ohlcv

# Load chosen strategy from config.yaml
strategy_func = load_strategy("config.yaml")

# Create logs folder
pathlib.Path("logs").mkdir(exist_ok=True)

# Logger setup
logger.add(
    "logs/bot.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}"
)

# Load API keys
load_dotenv()
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY") is not None)

api_key = os.getenv("BINANCE_API_KEY")
secret_key = os.getenv("BINANCE_SECRET_KEY")
client = Client(api_key, secret_key, testnet=True)

# Bot state
trades_today, wins_today = 0, 0
last_run_bar = None  # track last bar time to avoid duplicate runs


def is_new_bar_closed(df, timeframe: str) -> bool:
    """Check if a new bar is closed based on timeframe and last_run_bar."""
    global last_run_bar
    last_bar_time = df.index[-1]

    if last_run_bar is None:
        return True  # first run always
    return last_bar_time > last_run_bar


def run_strategy(force: bool = False):
    """Fetch data, run strategy, execute orders if needed."""
    global trades_today, wins_today, last_run_bar

    # Load config
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    timeframe = data_cfg.get("timeframe", "15m")
    start = data_cfg.get("start")
    end = data_cfg.get("end")
    lookback = data_cfg.get("lookback", 200)

    # Fetch OHLCV
    try:
        if start or end:
            df = get_ohlcv(symbol, timeframe=timeframe, start=start, end=end)
        else:
            klines = client.get_klines(symbol=symbol, interval=timeframe, limit=lookback)
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "_1", "_2", "_3", "_4", "_5", "_6"
                ]
            )
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV: {e}")
        return

    if df is None or df.empty:
        logger.warning("No data fetched.")
        return

    if not force and not is_new_bar_closed(df, timeframe):
        return

    last_run_bar = df.index[-1]

    balances = get_balances()
    logger.info(f"Current balances: {balances}")

    # ✅ pass force flag to strategy
    decision = strategy_func(df, force=force)

    if decision in ["BUY", "SELL"]:
        order = execute_order(decision, symbol=symbol, quantity=0.001)
        if order:
            trades_today += 1
            logger.info(f"Order placed: {decision} | OrderID: {order.get('orderId', 'N/A')}")
        else:
            logger.warning(f"Decision was {decision}, but no order was executed.")
    else:
        logger.info(f"Decision was HOLD, no trade executed.")


schedule.every().day.at("23:59").do(lambda: log_daily_performance(trades_today, wins_today))


if __name__ == "__main__":
    logger.info("🚀 Bot started | strategy=gpt | symbol=BTCUSDT | timeframe=15m | lookback=200")

    balances = get_balances()
    logger.info(f"Initial balances: {balances}")

    logger.info("🔄 First forced run")
    run_strategy(force=True)

    while True:
        try:
            run_strategy()
            schedule.run_pending()
            time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Bot stopped manually.")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(5)
