from binance import ThreadedWebsocketManager
from binance.client import Client
import pandas as pd
from loguru import logger
import time, threading

class WSManager:
    def __init__(self, api_key, secret_key, testnet=True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.twm = None
        self.client = Client(api_key, secret_key, testnet=testnet)

        # DataFrames
        self.df_trade = pd.DataFrame(columns=["timestamp", "close", "source"])
        self.df_1m = pd.DataFrame(columns=["timestamp", "close", "source"])

        # Last trade timestamp for monitoring
        self.last_trade_time = time.time()

    def start(self, symbol="BTCUSDT"):
        """Start WebSocket"""
        logger.info("[WS] Starting WebSocket...")
        self.twm = ThreadedWebsocketManager(api_key=self.api_key, api_secret=self.secret_key, testnet=self.testnet)
        self.twm.start()
        self.twm.start_trade_socket(callback=self.handle_trade, symbol=symbol)
        self.twm.join()

    def handle_trade(self, msg):
        """Handle each trade event from WebSocket"""
        price = float(msg["p"])
        ts = msg["T"] // 1000
        self.last_trade_time = time.time()

        new_row = pd.DataFrame([{"timestamp": ts, "close": price, "source": "TRADE"}])

        if self.df_trade.empty:
            self.df_trade = new_row
        else:
            self.df_trade = pd.concat([self.df_trade, new_row]).tail(500)

        logger.info(f"[TRADE] Price update: {price:.2f}")

    def fetch_1m_klines(self, symbol="BTCUSDT", limit=50):
        """Fetch 1-minute klines from REST API"""
        klines = self.client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "close"]]
        df["source"] = "1M"
        self.df_1m = df

    def get_latest_df(self):
        """Return trade data if available, otherwise fallback to 1m data"""
        if len(self.df_trade) > 0:
            return self.df_trade
        else:
            return self.df_1m

    def monitor_connection(self, symbol="BTCUSDT", timeout=600):
        """Restart WebSocket if no trade data received for timeout seconds"""
        while True:
            elapsed = time.time() - self.last_trade_time
            if elapsed > timeout:
                logger.warning(f"[WS] No trade for {elapsed:.0f}s, restarting WebSocket...")
                try:
                    self.twm.stop()
                except Exception as e:
                    logger.error(f"[WS] Failed to stop old WebSocket: {e}")
                t = threading.Thread(target=self.start, args=(symbol,), daemon=True)
                t.start()
                self.last_trade_time = time.time()
            time.sleep(60)
