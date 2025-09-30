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
    """Fetch BTC news from NewsAPI within last 72 hours."""
    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY is not set; cannot fetch news.")

    end = datetime.utcnow()
    start = end - timedelta(hours=72)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "bitcoin",
        "from": start.isoformat(),
        "to": end.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"NewsAPI request failed: {exc}") from exc

    if data.get("status") != "ok":
        message = data.get("message") or "Unknown error"
        raise RuntimeError(f"NewsAPI error: {message}")

    articles = data.get("articles") or []
    items = []
    for article in articles:
        title = (article.get("title") or "").strip()
        description = (article.get("description") or "").strip()
        if title:
            snippet = f"{title} - {description}" if description else title
            items.append(snippet)
    return items


def summarize_with_gemini(news_list, model="gemini-1.5-flash", retries=3):
    """Summarize with Gemini model, retry on failure."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set; cannot summarize news.")
    if not news_list:
        return None

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
            print(f"[WARN] Error with {model}: {e} | retrying in {wait}s...")
            time.sleep(wait)

    return None


def get_gemini_news():
    """Return summarized BTC news string, fallback to pro model if flash fails."""
    try:
        news = fetch_news()
    except Exception as exc:
        return f"[NEWS] Failed to fetch news: {exc}"

    if not news:
        return "[NEWS] No recent BTC news available."

    # Try flash first
    try:
        summary = summarize_with_gemini(news, model="gemini-2.5-flash-lite", retries=3)
    except Exception as exc:
        return f"[NEWS] Failed to summarize news: {exc}"

    if summary:
        return "[NEWS] " + summary

    # Fallback to pro if flash failed
    print("[WARN] Falling back to gemini-2.0-flash-lite...")
    summary = summarize_with_gemini(news, model="gemini-2.0-flash-lite", retries=2)
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
