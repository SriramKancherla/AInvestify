"""
Fetch REAL point-in-time fundamentals from SEC EDGAR's free XBRL API
(data.sec.gov) - no key required, but SEC requires a descriptive
User-Agent header on every request (set config.SEC_USER_AGENT to a real
contact email before running).

This replaces the old "reuse today's snapshot for all history" approach.
Every fact SEC returns comes with the date it was actually FILED, so we can
correctly answer "what did the market know about this company's financials
as of date t" without leaking future information into the past.

Requires live internet access - will NOT run inside a network-sandboxed
environment.

CHECKPOINTED: each ticker's full company-facts JSON is cached individually.
Safe to interrupt and re-run.

Usage:
    python fetch_fundamentals_sec.py            # fetch, then compile
    python fetch_fundamentals_sec.py --compile  # only recompile from cache
"""

import sys
import json
import time
import pandas as pd
from tqdm import tqdm
import urllib.request
import urllib.error

import config

SEC_FACTS_CACHE_DIR = config.CACHE_DIR / "sec_facts"
CIK_MAP_CACHE = config.CACHE_DIR / "cik_map.json"

HEADERS = {"User-Agent": config.SEC_USER_AGENT}


def fetch_cik_map() -> dict:
    """
    Ticker -> zero-padded 10-digit CIK, from SEC's free bulk mapping file.
    Cached locally since it rarely changes and is ~1000 requests worth of
    lookups otherwise.
    """
    if CIK_MAP_CACHE.exists():
        return json.load(open(CIK_MAP_CACHE))

    url = "https://www.sec.gov/files/company_tickers.json"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.load(resp)

    # raw is {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    mapping = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in raw.values()}
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(mapping, open(CIK_MAP_CACHE, "w"))
    return mapping


def fetch_company_facts(cik: str, retries: int = 3) -> dict | None:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # company genuinely has no XBRL facts (rare, e.g. very new listing)
            wait = (attempt + 1) * 5
            tqdm.write(f"  HTTP {e.code} for CIK {cik}, retrying after {wait}s")
            time.sleep(wait)
        except Exception as e:
            wait = (attempt + 1) * 5
            tqdm.write(f"  Error for CIK {cik}: {e}. Retrying after {wait}s")
            time.sleep(wait)
    return None


def extract_facts_table(ticker: str, facts_json: dict) -> pd.DataFrame:
    """
    Flatten a companyfacts JSON into rows of:
      Ticker, Concept, End, Filed, Val, Form, FiscalPeriod
    keeping only the concepts we care about (config.SEC_XBRL_CONCEPTS) and
    only 10-K/10-Q filings (skip amendments' noisy duplicates for a first
    pass - see README for the note on restatements).
    """
    rows = []
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
    for concept in config.SEC_XBRL_CONCEPTS:
        concept_data = us_gaap.get(concept)
        if not concept_data:
            continue
        for unit, entries in concept_data.get("units", {}).items():
            for e in entries:
                if e.get("form") not in ("10-K", "10-Q"):
                    continue
                rows.append({
                    "Ticker": ticker,
                    "Concept": concept,
                    "End": e.get("end"),
                    "Filed": e.get("filed"),
                    "Val": e.get("val"),
                    "Form": e.get("form"),
                    "FiscalPeriod": e.get("fp"),
                })
    if not rows:
        # pd.DataFrame([]) has ZERO COLUMNS (not just zero rows), which
        # writes to CSV as a single blank line - no header at all. Reading
        # that back later crashes with "EmptyDataError: No columns to
        # parse from file". Explicit columns here guarantee a valid
        # (if empty) CSV either way.
        return pd.DataFrame(columns=["Ticker", "Concept", "End", "Filed", "Val", "Form", "FiscalPeriod"])
    return pd.DataFrame(rows)


def ticker_is_cached(ticker: str) -> bool:
    return (SEC_FACTS_CACHE_DIR / f"{ticker}.csv").exists()


def fetch_all():
    SEC_FACTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

    pbar = tqdm(todo, desc="Fetching SEC fundamentals", unit="ticker")
    for ticker in pbar:
        pbar.set_postfix_str(ticker)

        cik = cik_map.get(ticker)
        if not cik:
            tqdm.write(f"  [{ticker}] not found in SEC's ticker->CIK map, skipping")
            pd.DataFrame(columns=["Ticker", "Concept", "End", "Filed", "Val", "Form", "FiscalPeriod"]).to_csv(
                SEC_FACTS_CACHE_DIR / f"{ticker}.csv", index=False
            )
            continue

        facts_json = fetch_company_facts(cik)
        if facts_json is None:
            table = pd.DataFrame(columns=["Ticker", "Concept", "End", "Filed", "Val", "Form", "FiscalPeriod"])
        else:
            table = extract_facts_table(ticker, facts_json)
        table.to_csv(SEC_FACTS_CACHE_DIR / f"{ticker}.csv", index=False)

        # SEC's fair-use guidance: max 10 requests/second. We're nowhere
        # near that, but stay polite - large bulk pulls have gotten IPs
        # temporarily blocked in the past.
        time.sleep(0.3)


def compile_from_cache():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    fact_files = sorted(SEC_FACTS_CACHE_DIR.glob("*.csv"))
    if not fact_files:
        print("No cached SEC facts found - nothing to compile.")
        return

    dfs, unreadable = [], []
    for f in fact_files:
        try:
            dfs.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            unreadable.append(f.stem)  # genuinely empty/corrupt - skip, don't crash the whole compile

    if unreadable:
        print(f"WARNING: {len(unreadable)} cached files were empty/corrupt and skipped: {unreadable[:10]}"
              f"{' ...' if len(unreadable) > 10 else ''}")
        print("These tickers likely have no data for the tracked concepts, or the fetch was interrupted "
              "mid-write. Delete their cache files and re-run this script to retry them individually:")
        for t in unreadable[:5]:
            print(f"  rm data/cache/sec_facts/{t}.csv")

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    combined.to_csv(config.SEC_FACTS_CSV, index=False)
    print(f"Compiled {len(combined)} SEC fact rows from {len(dfs)}/{len(fact_files)} tickers -> {config.SEC_FACTS_CSV}")

    missing = [t for t in config.TICKERS if not ticker_is_cached(t)]
    if missing:
        print(f"\n{len(missing)} tickers not yet fetched: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        print("Re-run this script again to fetch just the missing ones.")


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