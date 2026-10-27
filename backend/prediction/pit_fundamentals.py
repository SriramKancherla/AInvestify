"""
Computes REAL point-in-time fundamentals for any (ticker, as_of_date) pair,
using only information that was actually public as of that date.

This is the module that fixes the core bug: instead of reusing a single
"current" snapshot across 11 years of history, every value here is looked
up as-of a specific date, respecting SEC filing dates (no look-ahead) and
using our own price/dividend history for anything price-derived.

Known simplification (documented honestly, not hidden): revenue and net
income are taken from the most recently FILED ANNUAL (10-K) report rather
than a true rolling trailing-twelve-month figure computed by subtracting
consecutive quarterly filings. SEC's 10-Q figures are year-to-date
cumulative, not discrete quarters, so a correct TTM calculation requires
subtracting consecutive quarters - that's a real upgrade worth doing before
treating this as final, but "most recent annual figure" is still genuinely
point-in-time correct (no look-ahead), just lower time-resolution than TTM
between annual filings. See README.md.
"""

import numpy as np
import pandas as pd

import config


def normalize_date_column(series: pd.Series) -> pd.Series:
    """
    Robustly parses a Date column into plain, timezone-NAIVE datetimes.

    Why this exists: yfinance returns timestamps in the exchange's local
    timezone (e.g. America/New_York for US stocks). Once written to CSV and
    read back, rows from different times of year can carry different UTC
    offsets (-05:00 in winter vs -04:00 in summer, across DST transitions).
    A plain pd.to_datetime(...) call chokes on this ("mixed timezones" or
    "tz-aware datetime cannot be converted... unless utc=True") - confirmed
    by reproducing it directly. Passing utc=True first forces every value
    onto a common footing regardless of its original offset, and then we
    drop the tz info entirely, since daily OHLCV data only needs the
    calendar date, not a timezone-precise instant.
    """
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.dt.tz_localize(None)


def build_concept_series(sec_facts: pd.DataFrame, concepts: list[str], forms: tuple = ("10-K",)) -> pd.DataFrame:
    """
    From one ticker's raw SEC facts table, build a clean as-filed series for
    the first matching concept in `concepts` (tries each in order - e.g.
    ["Revenues", "SalesRevenueNet"] since companies use different tags),
    restricted to the given filing forms, sorted by Filed date.

    Returns columns: Filed (datetime), Val (float). Empty DataFrame if the
    ticker has no matching facts at all.
    """
    for concept in concepts:
        sub = sec_facts[(sec_facts["Concept"] == concept) & (sec_facts["Form"].isin(forms))]
        if sub.empty:
            continue
        sub = sub.dropna(subset=["Filed", "Val"]).copy()
        sub["Filed"] = pd.to_datetime(sub["Filed"])
        sub["End"] = pd.to_datetime(sub["End"])
        # If multiple facts share a Filed date (e.g. amendments), keep the
        # one with the latest fiscal period End - "most current info as of
        # that filing date."
        sub = sub.sort_values(["Filed", "End"]).drop_duplicates("Filed", keep="last")
        return sub[["Filed", "Val"]].sort_values("Filed").reset_index(drop=True)
    return pd.DataFrame(columns=["Filed", "Val"])


def lookup_as_of(concept_series: pd.DataFrame, as_of_dates: pd.Series) -> pd.Series:
    """
    For each date in as_of_dates, find the most recent Filed <= that date
    in concept_series and return its Val. NaN where no filing existed yet
    (e.g. a query date before the company's first 10-K).

    Vectorized via merge_asof rather than looping per-window - matters once
    you're computing this across tens of thousands of samples.
    """
    if concept_series.empty:
        return pd.Series(np.nan, index=as_of_dates.index)

    query = pd.DataFrame({"as_of": pd.to_datetime(as_of_dates).values}).sort_values("as_of")
    merged = pd.merge_asof(
        query, concept_series.sort_values("Filed"),
        left_on="as_of", right_on="Filed", direction="backward",
    )
    # merge_asof requires sorted input; restore original order
    merged.index = query.index
    return merged["Val"].reindex(as_of_dates.index)


class TickerFundamentalsIndex:
    """
    Pre-builds all the point-in-time concept series for one ticker once,
    then answers many as-of lookups against it cheaply. Build one of these
    per ticker in build_dataset.py rather than re-parsing raw SEC facts for
    every single window.
    """

    def __init__(self, ticker: str, sec_facts: pd.DataFrame, prices: pd.DataFrame, dividends: pd.DataFrame):
        self.ticker = ticker
        self.prices = prices.copy()
        self.prices["Date"] = normalize_date_column(self.prices["Date"])
        self.prices = self.prices.sort_values("Date").reset_index(drop=True)
        self.dividends = dividends.copy()
        if not self.dividends.empty:
            self.dividends["Date"] = normalize_date_column(self.dividends["Date"])

        self.revenue = build_concept_series(sec_facts, ["Revenues", "SalesRevenueNet"])
        self.net_income = build_concept_series(sec_facts, ["NetIncomeLoss"])
        self.stockholders_equity = build_concept_series(sec_facts, ["StockholdersEquity"])
        self.operating_income = build_concept_series(sec_facts, ["OperatingIncomeLoss"])
        self.depreciation = build_concept_series(sec_facts, ["DepreciationDepletionAndAmortization"])
        self.shares_outstanding = build_concept_series(
            sec_facts, ["CommonStockSharesOutstanding"], forms=("10-K", "10-Q")
        )

    def price_as_of(self, as_of_date: pd.Timestamp) -> float:
        past = self.prices[self.prices["Date"] <= as_of_date]
        if past.empty:
            return np.nan
        return float(past.iloc[-1]["Close"])

    def rolling_52w_high_low(self, as_of_date: pd.Timestamp) -> tuple[float, float]:
        window = self.prices[
            (self.prices["Date"] <= as_of_date) & (self.prices["Date"] > as_of_date - pd.Timedelta(days=365))
        ]
        if window.empty:
            return np.nan, np.nan
        return float(window["High"].max()), float(window["Low"].min())

    def trailing_dividend_per_share(self, as_of_date: pd.Timestamp) -> float:
        if self.dividends.empty:
            return 0.0
        window = self.dividends[
            (self.dividends["Date"] <= as_of_date) & (self.dividends["Date"] > as_of_date - pd.Timedelta(days=365))
        ]
        return float(window["DividendPerShare"].sum()) if not window.empty else 0.0

    def compute_for_dates(self, as_of_dates: list[str]) -> pd.DataFrame:
        """Compute the full config.FUNDAMENTALS_FIELDS vector for each date."""
        dates = pd.to_datetime(pd.Series(as_of_dates))

        shares = lookup_as_of(self.shares_outstanding, dates)
        equity = lookup_as_of(self.stockholders_equity, dates)
        revenue = lookup_as_of(self.revenue, dates)
        net_income = lookup_as_of(self.net_income, dates)
        op_income = lookup_as_of(self.operating_income, dates)
        depreciation = lookup_as_of(self.depreciation, dates).fillna(0.0)

        prices = dates.apply(self.price_as_of)
        high_low = dates.apply(self.rolling_52w_high_low)
        week52_high = high_low.apply(lambda hl: hl[0])
        week52_low = high_low.apply(lambda hl: hl[1])
        trailing_div = dates.apply(self.trailing_dividend_per_share)

        with np.errstate(divide="ignore", invalid="ignore"):
            bvps = np.where(shares > 0, equity / shares, np.nan)
            eps = np.where(shares > 0, net_income / shares, np.nan)
            sales_per_share = np.where(shares > 0, revenue / shares, np.nan)
            market_cap = prices * shares
            pe_ratio = np.where(eps > 0, prices / eps, np.nan)  # guard against negative/zero EPS
            price_to_book = np.where(bvps > 0, prices / bvps, np.nan)
            price_to_sales = np.where(sales_per_share > 0, prices / sales_per_share, np.nan)
            dividend_yield = np.where(prices > 0, trailing_div / prices, 0.0)
            ebitda = op_income + depreciation

        return pd.DataFrame({
            "market_cap": market_cap, "pe_ratio": pe_ratio, "dividend_yield": dividend_yield,
            "week52_high": week52_high, "week52_low": week52_low, "book_value_per_share": bvps,
            "ebitda": ebitda, "price_to_sales": price_to_sales, "price_to_book": price_to_book,
            "current_price": prices,
        })
