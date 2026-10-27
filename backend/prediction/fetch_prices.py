"""
Fetch daily OHLCV price history AND real dividend payment history for every
ticker in config.TICKERS.

Both of these are genuinely time-varying series (unlike the old fundamentals
snapshot this pipeline used to take) — dividends are actual paid amounts on
actual dates, used later to compute a real point-in-time dividend yield
instead of a static "current yield" reused across a decade.

Requires live internet access to Yahoo Finance — will NOT run inside a
network-sandboxed environment. Run it locally or in a notebook (Colab, etc.)
with normal internet access.

CHECKPOINTED: each ticker's result is cached individually. Safe to interrupt
and re-run — already-cached tickers are skipped.

Usage:
    python fetch_prices.py            # fetch (skips cached tickers), then compile
    python fetch_prices.py --compile  # only recompile from existing cache
"""

import sys
import time
import random
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

import config

PRICE_CACHE_DIR = config.CACHE_DIR / "prices"
DIVIDENDS_CACHE_DIR = config.CACHE_DIR / "dividends"


def fetch_price_history(ticker: str, retries: int = 5) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(
                start=config.START_DATE, end=config.END_DATE, interval="1d",
                auto_adjust=True,  # explicit, not relying on the library default: adjusts
                                   # OHLC for splits/dividends so a split doesn't render as
                                   # a fake 50%+ "crash" in the chart. Confirmed this is
                                   # yfinance's own default for Ticker.history() as of this
                                   # writing, but stated explicitly here rather than assumed.
            )
            if df is None or df.empty:
                raise ValueError(f"Empty history returned for {ticker}")
            df = df.reset_index()
            # yfinance returns the exchange's local timezone (e.g. America/New_York).
            # Strip it here rather than let it flow downstream - across DST
            # transitions, rows end up with different UTC offsets (-05:00 vs
            # -04:00), which breaks re-parsing later (confirmed: causes
            # "mixed timezones" / "tz-aware datetime cannot be converted"
            # errors in pandas). Daily OHLCV only needs the calendar date
            # anyway, so drop the tz info entirely rather than carry it.
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df["Ticker"] = ticker
            return df[["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]]
        except YFRateLimitError:
            wait = (attempt + 1) * 8 + random.randint(1, 3)
            tqdm.write(f"  [{ticker}] rate limited, waiting {wait}s (retry {attempt + 1}/{retries})")
            time.sleep(wait)
        except Exception as e:
            wait = (attempt + 1) * 5
            tqdm.write(f"  [{ticker}] error: {e}. Retrying after {wait}s")
            time.sleep(wait)
    tqdm.write(f"  [{ticker}] FAILED after {retries} retries")
    return pd.DataFrame()


def fetch_dividend_history(ticker: str, retries: int = 3) -> pd.DataFrame:
    """Real dividend payments with real dates - used for point-in-time yield."""
    for attempt in range(retries):
        try:
            div = yf.Ticker(ticker).dividends  # pd.Series, index=date, value=$/share
            df = div.reset_index()
            df.columns = ["Date", "DividendPerShare"]
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df["Ticker"] = ticker
            return df[["Ticker", "Date", "DividendPerShare"]]
        except Exception as e:
            wait = (attempt + 1) * 4
            tqdm.write(f"  [{ticker}] dividends error: {e}. Retrying after {wait}s")
            time.sleep(wait)
    # No dividends is a legitimate outcome (many companies pay none) -
    # return empty rather than treat as failure.
    return pd.DataFrame(columns=["Ticker", "Date", "DividendPerShare"])


def ticker_is_cached(ticker: str) -> bool:
    return (PRICE_CACHE_DIR / f"{ticker}.csv").exists() and (DIVIDENDS_CACHE_DIR / f"{ticker}.csv").exists()


def fetch_all():
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DIVIDENDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    todo = [t for t in config.TICKERS if not ticker_is_cached(t)]
    already_done = len(config.TICKERS) - len(todo)
    if already_done:
        print(f"Resuming: {already_done}/{len(config.TICKERS)} tickers already cached, skipping them.")
    if not todo:
        print("All tickers already cached. Nothing to fetch.")
        return

    pbar = tqdm(todo, desc="Fetching prices+dividends", unit="ticker")
    for ticker in pbar:
        pbar.set_postfix_str(ticker)

        prices = fetch_price_history(ticker)
        if not prices.empty:
            prices.to_csv(PRICE_CACHE_DIR / f"{ticker}.csv", index=False)

        dividends = fetch_dividend_history(ticker)
        dividends.to_csv(DIVIDENDS_CACHE_DIR / f"{ticker}.csv", index=False)

        time.sleep(1.5)


def compile_from_cache():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    price_files = sorted(PRICE_CACHE_DIR.glob("*.csv"))
    if price_files:
        price_df = pd.concat([pd.read_csv(f) for f in price_files], ignore_index=True)
        price_df.to_csv(config.PRICES_CSV, index=False)
        print(f"Compiled {len(price_df)} price rows from {len(price_files)} tickers -> {config.PRICES_CSV}")
    else:
        print("No cached price files found - nothing to compile.")

    div_files = sorted(DIVIDENDS_CACHE_DIR.glob("*.csv"))
    if div_files:
        div_df = pd.concat([pd.read_csv(f) for f in div_files], ignore_index=True)
        div_df.to_csv(config.DIVIDENDS_CSV, index=False)
        print(f"Compiled {len(div_df)} dividend rows from {len(div_files)} tickers -> {config.DIVIDENDS_CSV}")
    else:
        print("No cached dividend files found - nothing to compile.")

    missing = [t for t in config.TICKERS if not ticker_is_cached(t)]
    if missing:
        print(f"\n{len(missing)} tickers still missing: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        print("Re-run this script again to retry just the missing ones.")


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