# btc_news_summary.py
import os
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai

# Load .env
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def fetch_news():
    """Fetch BTC news from NewsAPI within last 72 hours"""
    end = datetime.utcnow()
    start = end - timedelta(hours=72)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "bitcoin",
        "from": start.isoformat(),
        "to": end.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }
    r = requests.get(url, params=params).json()
    return [a["title"] + " - " + (a["description"] or "") for a in r.get("articles", [])]


def summarize_with_gemini(news_list, model="gemini-1.5-flash", retries=3):
    """Summarize with Gemini model, retry on failure"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    text_input = "\n".join(news_list[:15])
    prompt = f"""
    Summarize the following Bitcoin news headlines and descriptions 
    from the last 72 hours. Focus on key themes, sentiment, and possible 
    market impact. Provide a concise digest:

    {text_input}
    """

    for i in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return resp.text.strip()
        except Exception as e:
            wait = 2 ** i
            print(f"⚠️ Error with {model}: {e} | retrying in {wait}s...")
            time.sleep(wait)

    return None


def get_gemini_news():
    """Return summarized BTC news string, fallback to pro model if flash fails"""
    news = fetch_news()
    if not news:
        return "[NEWS] No recent BTC news available."

    # Try flash first
    summary = summarize_with_gemini(news, model="gemini-1.5-flash", retries=3)
    if summary:
        return "[NEWS] " + summary

    # Fallback to pro if flash failed
    print("⚠️ Falling back to gemini-1.5-pro...")
    summary = summarize_with_gemini(news, model="gemini-1.5-pro", retries=2)
    if summary:
        return "[NEWS] " + summary

    return "[NEWS] Gemini API unavailable after retries."


if __name__ == "__main__":
    gemini_news = get_gemini_news()
    print("=== Gemini BTC News Summary ===")
    print(gemini_news)
    with open("btc_summary.log", "a", encoding="utf-8") as f:
        f.write(f"\n==== {datetime.utcnow()} ====\n")
        f.write(gemini_news + "\n")
