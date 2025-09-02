import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from strategies.base_strategy import BaseStrategy

# 載入 .env
load_dotenv()

class GPTStrategy(BaseStrategy):
    def __init__(self, lookback=20, model="gpt-4o-mini"):
        self.lookback = lookback
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """逐根 candle 呼叫 GPT 產生訊號（非常慢，主要用於小區間測試）"""
        signals = pd.Series(0, index=df.index)

        for i in range(self.lookback, len(df)):
            closes = df["close"].iloc[i - self.lookback:i].tolist()
            prompt = f"收盤價序列 {closes}，請輸出 BUY / SELL / HOLD"

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                decision = response.choices[0].message.content.strip().upper()
                if "BUY" in decision:
                    signals.iloc[i] = 1
                elif "SELL" in decision:
                    signals.iloc[i] = -1
                else:
                    signals.iloc[i] = 0
            except Exception as e:
                print(f"⚠️ GPT API 出錯：{e}")
                signals.iloc[i] = 0

        return signals
