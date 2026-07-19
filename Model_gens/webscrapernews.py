import feedparser
import requests
import time

try:
    from Model_gens.text_cleaning import clean_text_for_model, fix_mojibake
except ImportError:  # when run as a standalone script (cwd on sys.path)
    from text_cleaning import clean_text_for_model, fix_mojibake


def _clean_for_display(text: str, max_len: int = 220) -> str:
    """
    Lightweight cleanup for human-readable output. Repairs mojibake and collapses
    real newlines so the UI does not show garbled or multi-line snippets.
    """
    text = fix_mojibake(str(text)).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def scrape_google_news_rss(stock_name: str, max_articles: int = 10) -> list[dict]:
    """
    Scrape Google News RSS and return structured items.
    Returns:
      [
        {"title": ..., "summary": ..., "link": ..., "text_for_model": ...},
        ...
      ]
    """
    query = stock_name.replace(" ", "+") + "+stock"
    rss_url = f"https://news.google.com/rss/search?q={query}"

    retries = 2
    timeout_seconds = 8
    backoff_seconds = 0.3
    feed = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(rss_url, timeout=timeout_seconds)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            break
        except Exception:
            if attempt < retries:
                time.sleep(backoff_seconds * (attempt + 1))
            else:
                return []
    if feed is None:
        return []
    items: list[dict] = []

    for entry in feed.entries[:max_articles]:
        title = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or ""
        link = getattr(entry, "link", "") or ""

        combined_text = f"{title} {summary}"
        items.append(
            {
                "title": _clean_for_display(title),
                "summary": _clean_for_display(summary),
                "link": link,
                "text_for_model": clean_text_for_model(combined_text),
            }
        )

    return items

if __name__ == "__main__":
    stock = input("Enter stock name: ")
    news_data = scrape_google_news_rss(stock)

    print("\nNews Items:\n")
    for i, item in enumerate(news_data, 1):
        print(f"{i}. {item['title']}\n   {item['summary']}\n   {item['link']}\n")
