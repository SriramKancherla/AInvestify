"""
Central configuration for the multimodal stock-chart forecasting dataset.

Edit TICKERS / date range / window sizes here — every other script reads
from this file so there's a single source of truth.
"""

from pathlib import Path
import os
import sys
import time

def _clog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [config.py] {msg}", file=sys.stderr, flush=True)

_clog("starting: import pandas (can be slow on a cold Windows start, same as torch)")
_t0 = time.time()
import pandas as pd
_clog(f"finished: import pandas ({time.time() - _t0:.1f}s)")

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "cache"
SP500_LIST_CSV = CACHE_DIR / "sp500_constituents.csv"

# ---------------------------------------------------------------------
# Universe & date range
# ---------------------------------------------------------------------
# SUBSET_SIZE controls how many S&P 500 tickers to use, stratified by
# GICS sector so smaller subsets still span the whole market rather than
# clustering in tech/finance. Set to None to use the full ~500.
#
# Storage is not a real constraint here (compressed uint8 samples run
# ~1-2 KB each), so the limiting factors are really fetch time and
# yfinance rate limits, not disk space:
#
#   15   -> ~5K samples    (pipeline smoke-testing only, NOT enough to train on)
#   100  -> ~35K samples   (~0.06 GB - reasonable first real training run)
#   250  -> ~87K samples   (~0.14 GB - solid diversity)
#   None -> ~174K samples  (~0.29 GB - full S&P 500, best diversity, longest fetch)
SUBSET_SIZE = None


def _load_tickers(subset_size):
    if not SP500_LIST_CSV.exists():
        raise FileNotFoundError(
            f"{SP500_LIST_CSV} not found. Fetch the constituent list first:\n"
            f'  curl -s "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv" '
            f"-o {SP500_LIST_CSV}"
        )
    df = pd.read_csv(SP500_LIST_CSV)
    if subset_size is None or subset_size >= len(df):
        return df
    # Stratified sample by sector so a small subset still spans the market
    frac = subset_size / len(df)
    sampled = (
        df.groupby("GICS Sector", group_keys=False)
        .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=42))
    )
    return sampled.head(subset_size).reset_index(drop=True)


_tickers_df = None
TICKERS: list[str] = []
TICKER_NAMES: dict = {}


def ensure_tickers_loaded():
    """Lazy-load S&P ticker list. Live API inference does not need this file."""
    global _tickers_df, TICKERS, TICKER_NAMES
    if _tickers_df is not None:
        return _tickers_df
    _clog(f"starting: load tickers from {SP500_LIST_CSV} (SUBSET_SIZE={SUBSET_SIZE})")
    _t0 = time.time()
    _tickers_df = _load_tickers(SUBSET_SIZE)
    _clog(f"finished: load tickers ({time.time() - _t0:.1f}s) - {len(_tickers_df)} rows")
    TICKERS = _tickers_df["Symbol"].str.replace(".", "-", regex=False).tolist()
    TICKER_NAMES = dict(zip(TICKERS, _tickers_df["Security"]))
    _clog(f"TICKERS list built: {len(TICKERS)} tickers")
    return _tickers_df


# Keep attribute access working for scripts that still read config.TICKERS
# without calling ensure_tickers_loaded() — attempt load, else leave empty.
try:
    if SP500_LIST_CSV.exists():
        ensure_tickers_loaded()
    else:
        _clog(f"S&P list not found at {SP500_LIST_CSV}; skipping ticker load (live forecast still works)")
except Exception as exc:
    _clog(f"ticker load skipped: {exc}")

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# ---------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------
# Input window: a FULL TRADING YEAR of history (t-252 ... t), not a short
# slice - so every sample genuinely has a year of price+fundamentals+news
# behind it, matching how a full year of context should inform a forecast.
INPUT_WINDOW_DAYS = 252  # ~1 trading year (252 trading days ≈ 1 calendar year)

# Forecast horizons: instead of one fixed number of days ahead, generate a
# sample at EACH of these horizons per input window, so the model learns to
# predict at any distance up to a year out - not just one fixed point.
#   5   ≈ 1 week ahead
#   21  ≈ 1 month ahead
#   63  ≈ 1 quarter ahead
#   126 ≈ 6 months ahead
#   189 ≈ 9 months ahead
#   252 ≈ 1 year ahead
# Each (ticker, window_start) produces len(HORIZON_DAYS_OPTIONS) samples,
# all sharing the same 1-year input X but a different target Y and horizon.
HORIZON_DAYS_OPTIONS = [5, 21, 63, 126, 189, 252]

STRIDE_DAYS = 21          # step between consecutive window starts for the
                           # same ticker (21 ≈ monthly - with a 252-day
                           # window, a small stride creates extreme overlap
                           # between consecutive samples)

# ---------------------------------------------------------------------
# Image / channel dimensions
# ---------------------------------------------------------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
# Channels stacked per timestep: [price_chart, fundamentals, news]
NUM_CHANNELS = 3

# ---------------------------------------------------------------------
# Point-in-time fundamentals fields
# ---------------------------------------------------------------------
# Every field here is either a raw SEC-filed value (as-of its real filing
# date, no look-ahead) or derived from SEC data + our own point-in-time
# price/dividend history. See fetch_fundamentals_sec.py and
# build_dataset.py's `compute_pit_fundamentals` for exactly how each is
# computed. This intentionally replaces the old "reuse today's snapshot"
# approach — see README.md "Point-in-time correctness" section.
FUNDAMENTALS_FIELDS = [
    "market_cap",           # price(t) * shares_outstanding(t)
    "pe_ratio",              # price(t) / trailing-12mo EPS(t)
    "dividend_yield",        # trailing-12mo dividends/share(t) / price(t)
    "week52_high",           # rolling max of OUR OWN price history, 252d window ending t
    "week52_low",            # rolling min of OUR OWN price history, 252d window ending t
    "book_value_per_share",  # StockholdersEquity(t) / shares_outstanding(t)
    "ebitda",                # OperatingIncomeLoss(t) + D&A(t), most recent filed quarter as-of t
    "price_to_sales",        # price(t) / trailing-12mo revenue-per-share(t)
    "price_to_book",         # price(t) / book_value_per_share(t)
    "current_price",         # price(t) itself, included so the channel also encodes price level
]

# SEC XBRL concepts pulled per company (us-gaap taxonomy). SEC requires a
# descriptive User-Agent identifying you on every request - a real contact
# email is required, not optional: placeholder agents get HTTP 403 and repeat
# offenders get the IP banned.
#
# Read from the environment first, so this contact can be changed per deploy —
# or kept out of the repo entirely — without editing code.
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "AInvestify kancherlasriram2006@gmail.com"
)
SEC_XBRL_CONCEPTS = [
    "Revenues", "SalesRevenueNet",  # revenue (try both; companies vary in which they use)
    "NetIncomeLoss",
    "StockholdersEquity",
    "OperatingIncomeLoss",
    "DepreciationDepletionAndAmortization",
    "CommonStockSharesOutstanding",
]

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PRICES_CSV = RAW_DIR / "prices.csv"
SEC_FACTS_CSV = RAW_DIR / "sec_facts.csv"          # point-in-time filed financial facts, all tickers
DIVIDENDS_CSV = RAW_DIR / "dividends.csv"           # point-in-time dividend payment history, all tickers
NEWS_TONE_CSV = RAW_DIR / "news_tone.csv"           # daily 8-K-derived tone timeline, all tickers (fetch_news_sec8k.py)

DATASET_MANIFEST = PROCESSED_DIR / "manifest.csv"
SAMPLES_DIR = PROCESSED_DIR / "samples"


# Storage: samples are saved as uint8 (0-255) instead of float32 (0.0-1.0).
# All channel values are normalized to [0,1] by render_channels.py, so this
# is a lossless-enough quantization for image data and cuts sample storage
# 4x (a real concern once you're at 100+ tickers - see SUBSET_SIZE comment
# above for size estimates).
STORAGE_DTYPE = "uint8"

