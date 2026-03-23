import feedparser
import re
import requests
import time


def _clean_for_model(text: str) -> str:
    """
    Clean text similar to the tweet-cleaning used by the sentiment model.
    Keeps only lowercase letters/spaces (removes punctuation/digits), removes URLs.
    """
    text = str(text).lower()
    text = re.sub(r"http\\S+", "", text)
    # Remove any HTML tags that may appear in RSS summaries.
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def _clean_for_display(text: str, max_len: int = 220) -> str:
    """
    Lightweight cleanup for human-readable output.
    """
    text = str(text).replace("\\n", " ").strip()
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
                "text_for_model": _clean_for_model(combined_text),
            }
        )

    return items

if __name__ == "__main__":
    stock = input("Enter stock name: ")
    news_data = scrape_google_news_rss(stock)

    print("\nNews Items:\n")
    for i, item in enumerate(news_data, 1):
        print(f"{i}. {item['title']}\n   {item['summary']}\n   {item['link']}\n")
