# btc_news_summary.py
import os
import requests
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
    start = end - timedelta(hours=720)
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

def get_gemini_news():
    """Return summarized BTC news string for strategy integration"""
    news = fetch_news()
    if not news:
        return "No recent BTC news available."

    client = genai.Client(api_key=GEMINI_API_KEY)

    # 取前 15 條避免 token 爆掉
    text_input = "\n".join(news[:100])

    prompt = f"""
    Summarize the following Bitcoin news headlines and descriptions 
    from the last 72 hours. Focus on key themes, sentiment, and possible 
    market impact. Provide a concise digest:

    {text_input}
    """

    resp = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    gemini_news = resp.text.strip()
    return gemini_news

if __name__ == "__main__":
    gemini_news = get_gemini_news()
    print("=== Gemini BTC News Summary ===")
    print(gemini_news)
