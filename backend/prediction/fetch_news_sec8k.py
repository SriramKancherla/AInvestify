"""
Fetches a "news tone" signal per ticker from SEC 8-K filings, replacing the
GDELT-based approach (fetch_news_gdelt.py), which turned out to be
unreliable in practice - inconsistent API behavior, vague/unverifiable
historical coverage claims, and blank error messages that were hard to
diagnose without a live connection to debug against.

WHAT THIS ACTUALLY CAPTURES, HONESTLY: 8-K filings are "material event"
disclosures companies are legally required to file within ~4 business days
of specific events (earnings, M&A, executive changes, bankruptcy,
material agreements, etc.). This is NARROWER than general news coverage -
it won't capture analyst commentary, product buzz, or general media
sentiment the way GDELT was supposed to. What you get in exchange:
- 100% free, no API key beyond the same User-Agent SEC already requires
- Genuinely historical (SEC's EDGAR system covers back to 1993-1994,
  comfortably covering 2015-2025)
- Point-in-time correct (filing dates are real, verified dates)
- Built on the SAME infrastructure already tested and working in
  fetch_fundamentals_sec.py - no new unproven API to debug

HOW "TONE" IS DERIVED: this is a coarse heuristic, not text-based
sentiment analysis. Each 8-K's item code(s) (e.g. "1.03" = Bankruptcy,
"1.01" = new material agreement) are mapped to a rough tone score based on
whether that TYPE of event is generally negative, positive, or genuinely
ambiguous (many item codes, like 5.02 "officer departure/appointment,"
cover both good and bad outcomes and are scored neutral rather than guessed
at). This is deliberately conservative - see ITEM_TONE_MAP below. A
stronger (but much heavier) upgrade path would be to fetch each 8-K's
actual text/press-release exhibit and run real sentiment analysis (VADER)
on it - noted in README as a next step, not implemented here.

Requires live internet access - will NOT run inside a network-sandboxed
environment.

CHECKPOINTED per ticker. Safe to interrupt and re-run.

Usage:
    python fetch_news_sec8k.py            # fetch, then compile
    python fetch_news_sec8k.py --compile  # only recompile from cache
"""

import sys
import time
import json
import pandas as pd
from tqdm import tqdm
import urllib.request
import urllib.error

import config
from fetch_fundamentals_sec import fetch_cik_map, HEADERS

SEC_8K_CACHE_DIR = config.CACHE_DIR / "sec_8k"

# Coarse item-code -> tone mapping. 0 = genuinely ambiguous/procedural from
# the item code alone, not "no news happened." Deliberately conservative:
# only codes with a clear, unambiguous direction get a non-zero score.
ITEM_TONE_MAP = {
    "1.01": 3,    # Entry into a material definitive agreement - constructive
    "1.02": -3,   # Termination of a material agreement
    "1.03": -8,   # Bankruptcy or receivership
    "1.04": -2,   # Mine safety reporting
    "1.05": -5,   # Material cybersecurity incident
    "2.01": 1,    # Completion of acquisition/disposition - mild positive lean
    "2.02": 0,    # Results of operations (earnings) - direction unknown from item code alone
    "2.03": 0,    # Creation of a financial obligation - neutral (financing activity)
    "2.04": -5,   # Triggering event accelerating a financial obligation
    "2.05": -3,   # Costs associated with exit/disposal activities
    "2.06": -6,   # Material impairments
    "3.01": -7,   # Notice of delisting
    "3.02": 0,    # Unregistered sales of equity securities
    "3.03": 0,    # Material modification to security holder rights
    "4.01": -2,   # Change in certifying accountant
    "4.02": -6,   # Non-reliance on previously issued financials (restatement)
    "5.02": 0,    # Officer/director departure OR appointment - genuinely ambiguous from item code
    "5.03": 0,    # Bylaws/fiscal year amendments
    "5.04": 0,    # Trading suspension under benefit plans
    "5.05": 0,    # Code of ethics amendments
    "5.07": 0,    # Shareholder vote results
    "7.01": 0,    # Regulation FD disclosure
    "8.01": 0,    # Catch-all "other events"
    "9.01": 0,    # Financial statements/exhibits - procedural
}
DEFAULT_ITEM_TONE = 0.0


def score_items(items_str: str) -> float:
    """'1.01,9.01' -> average tone across the listed item codes."""
    if not items_str or not isinstance(items_str, str):
        return DEFAULT_ITEM_TONE
    codes = [c.strip() for c in items_str.split(",") if c.strip()]
    if not codes:
        return DEFAULT_ITEM_TONE
    scores = [ITEM_TONE_MAP.get(c, DEFAULT_ITEM_TONE) for c in codes]
    return sum(scores) / len(scores)


def fetch_json(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            wait = (attempt + 1) * 5
            tqdm.write(f"  HTTP {e.code} for {url}, retrying after {wait}s")
            time.sleep(wait)
        except Exception as e:
            wait = (attempt + 1) * 5
            tqdm.write(f"  Error fetching {url}: {type(e).__name__}: {e!r}. Retrying after {wait}s")
            time.sleep(wait)
    return None


def extract_8k_rows(ticker: str, filings_block: dict) -> list[dict]:
    """One dict per 8-K filing found in a submissions JSON block (either
    the top-level 'recent' block or one of the older paginated files -
    both have the same array shape)."""
    rows = []
    forms = filings_block.get("form", [])
    dates = filings_block.get("filingDate", [])
    items = filings_block.get("items", [])
    for i, form in enumerate(forms):
        if not form.startswith("8-K"):
            continue
        item_str = items[i] if i < len(items) else ""
        date = dates[i] if i < len(dates) else None
        if not date:
            continue
        rows.append({"Ticker": ticker, "Date": date, "Tone": score_items(item_str)})
    return rows


def fetch_8k_history_for_ticker(ticker: str, cik: str) -> pd.DataFrame:
    """
    Fetches ALL 8-K filings for one company, following pagination into
    older submission files as needed to cover the full configured date
    range (the main submissions JSON only holds the most recent ~1000
    filings / 1 year - older history lives in separate linked files).
    """
    main = fetch_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if main is None:
        return pd.DataFrame(columns=["Ticker", "Date", "Tone"])

    all_rows = extract_8k_rows(ticker, main.get("filings", {}).get("recent", {}))

    for older_file in main.get("filings", {}).get("files", []):
        name = older_file.get("name")
        if not name:
            continue
        older_json = fetch_json(f"https://data.sec.gov/submissions/{name}")
        if older_json is not None:
            all_rows.extend(extract_8k_rows(ticker, older_json))
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame(columns=["Ticker", "Date", "Tone"])

    df = pd.DataFrame(all_rows)
    # Multiple items already averaged per filing; if a company somehow
    # files >1 8-K on the same calendar day, average across those too so
    # there's exactly one tone value per ticker-day, matching what
    # render_news_channel expects.
    df = df.groupby(["Ticker", "Date"], as_index=False)["Tone"].mean()
    return df[(df["Date"] >= config.START_DATE) & (df["Date"] <= config.END_DATE)]


def ticker_is_cached(ticker: str) -> bool:
    return (SEC_8K_CACHE_DIR / f"{ticker}.csv").exists()


def fetch_all():
    SEC_8K_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching ticker -> CIK mapping ...")
    cik_map = fetch_cik_map()
    print(f"  {len(cik_map)} companies in SEC's mapping\n")

    todo = [t for t in config.TICKERS if not ticker_is_cached(t)]
    already_done = len(config.TICKERS) - len(todo)
    if already_done:
        print(f"Resuming: {already_done}/{len(config.TICKERS)} tickers already cached, skipping them.")
    if not todo:
        print("All tickers already cached. Nothing to fetch.")
        return

    failed_tickers = []
    pbar = tqdm(todo, desc="Fetching SEC 8-K news tone", unit="ticker")
    for ticker in pbar:
        pbar.set_postfix_str(ticker)
        cik = cik_map.get(ticker)
        if not cik:
            tqdm.write(f"  [{ticker}] not found in SEC's ticker->CIK map, skipping")
            pd.DataFrame(columns=["Ticker", "Date", "Tone"]).to_csv(SEC_8K_CACHE_DIR / f"{ticker}.csv", index=False)
            continue
        try:
            df = fetch_8k_history_for_ticker(ticker, cik)
        except Exception as e:
            tqdm.write(f"  [{ticker}] UNEXPECTED ERROR, skipping this ticker entirely: {type(e).__name__}: {e!r}")
            failed_tickers.append(ticker)
            continue
        df.to_csv(SEC_8K_CACHE_DIR / f"{ticker}.csv", index=False)
        if df.empty:
            tqdm.write(f"  [{ticker}] WARNING: zero 8-K filings found in the configured date range")
        time.sleep(0.3)  # stay well under SEC's 10 req/sec fair-use guidance

    if failed_tickers:
        print(f"\n{len(failed_tickers)} tickers hit an unexpected error and were skipped: {failed_tickers}")
        print("Re-run this script to retry them.")


def compile_from_cache():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SEC_8K_CACHE_DIR.glob("*.csv"))
    if not files:
        print("No cached SEC 8-K files found - nothing to compile.")
        return

    dfs, unreadable = [], []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            unreadable.append(f.stem)

    if unreadable:
        print(f"WARNING: {len(unreadable)} cached files were empty/corrupt and skipped: {unreadable[:10]}")

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["Ticker", "Date", "Tone"])
    combined.to_csv(config.NEWS_TONE_CSV, index=False)
    print(f"Compiled {len(combined)} daily tone rows from {len(dfs)}/{len(files)} tickers -> {config.NEWS_TONE_CSV}")

    if not combined.empty:
        coverage_by_year = pd.to_datetime(combined["Date"], errors="coerce").dt.year.value_counts().sort_index()
        print("\nRow counts by year:")
        print(coverage_by_year.to_string())

    missing = [t for t in config.TICKERS if not ticker_is_cached(t)]
    if missing:
        print(f"\n{len(missing)} tickers not yet fetched: {missing[:10]}{' ...' if len(missing) > 10 else ''}")


def main():
    import training_config as tc
    tc.refuse_rebuild_or_retrain()
    if "--compile" in sys.argv:
        compile_from_cache()
        return
    fetch_all()
    compile_from_cache()


if __name__ == "__main__":
    main()