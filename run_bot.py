import os, time, schedule, threading, pathlib
from dotenv import load_dotenv
from binance.client import Client
from strategies.sma_strategy import generate_sma_signal
from trading.order_executor import execute_order, get_balances
from trading.performance_tracker import log_daily_performance
from trading.ws_manager import WSManager
from loguru import logger
from strategy_loader import load_strategy

# Load chosen strategy from config.yaml
strategy_func = load_strategy("config.yaml")

# Create logs folder
pathlib.Path("logs").mkdir(exist_ok=True)

# Dedicated logger for BOT
bot_logger = logger.bind(tag="BOT")
logger.add("logs/bot.log",
           rotation="1 day",
           retention="30 days",
           level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | [{extra[tag]}] {message}",
           filter=lambda record: record["extra"].get("tag") == "BOT")

# Load API keys
load_dotenv()
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY") is not None)

api_key = os.getenv("BINANCE_API_KEY")
secret_key = os.getenv("BINANCE_SECRET_KEY")
client = Client(api_key, secret_key, testnet=True)

trades_today, wins_today = 0, 0

def run_strategy(ws_manager):
    global trades_today, wins_today
    df = ws_manager.get_latest_df()
    if len(df) < 10:
        return
    signal = strategy_func(df)
    price = df.iloc[-1]["close"]
    source = df.iloc[-1]["source"]
    balances = get_balances()

    bot_logger.info(f"Signal: {signal} | Price: {price:.2f} | Balances: {balances} | Source: [{source}]")

    order = execute_order(signal, symbol="BTCUSDT", quantity=0.001)
    if order:
        trades_today += 1
        if signal == "SELL":
            try:
                buy_price = float(order["fills"][0]["price"])
                if price > buy_price:
                    wins_today += 1
            except Exception:
                pass

# Daily performance summary at 23:59
schedule.every().day.at("23:59").do(lambda: log_daily_performance(trades_today, wins_today))

if __name__ == "__main__":
    bot_logger.info("🚀 SMA Bot started (trade+1m hybrid mode with auto-restart WS)")
    ws = WSManager(api_key, secret_key, testnet=True)

    # Start WebSocket
    t = threading.Thread(target=ws.start, args=("BTCUSDT",), daemon=True)
    t.start()

    # Start WebSocket monitor
    monitor_thread = threading.Thread(target=ws.monitor_connection, args=("BTCUSDT",), daemon=True)
    monitor_thread.start()

    while True:
        ws.fetch_1m_klines()
        run_strategy(ws)
        schedule.run_pending()
        time.sleep(5)
