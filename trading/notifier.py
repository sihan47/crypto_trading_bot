# trading/notifier.py
import os
import requests
from dotenv import load_dotenv

# load environment variables from .env
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_message(text: str):
    """Send message to Telegram chat"""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        if resp.status_code != 200:
            print(f"Telegram HTTP failure: {resp.status_code}")
            return False
        return resp.json().get('ok') is True
    except Exception as e:
        print(f"Telegram send failed: {type(e).__name__}")
        return False
