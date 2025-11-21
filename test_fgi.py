"""
Quick check for the current Crypto Fear & Greed Index.

Run:
    python test_fgi.py

It prints both the cached value (what the bot would use) and a live value
fetched directly from the API so you can confirm the cache isn’t stuck.
"""

from strategies.fear_gread import get_fgi, _fetch_latest, _FGI_FILE


def main() -> None:
    cached = get_fgi()
    print("[Cached]  get_fgi() ->", cached)
    print("Cache file:", _FGI_FILE)

    live = _fetch_latest()
    print("[Live]    _fetch_latest() ->", live)

    if cached != live:
        print("Cache differs from live. Delete the cache file to refresh:")
        print(f"  del \"{_FGI_FILE}\"")


if __name__ == "__main__":
    main()
