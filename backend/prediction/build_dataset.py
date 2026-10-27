"""
Assemble the final dataset: slide a window across each ticker's price
history, and for every valid window produce one sample:

    X : (IMG_HEIGHT, IMG_WIDTH, 3) - fundamentals+news+price up to time t
    Y : (IMG_HEIGHT, IMG_WIDTH)    - price chart at t+1 ... t+k (the target)

Unlike the pipeline's first version, fundamentals here are computed
POINT-IN-TIME as of each window's end date (via pit_fundamentals.py,
respecting real SEC filing dates - no look-ahead), and news uses GDELT's
daily tone timeline matched to each window's actual date range - not a
single reused snapshot or a mostly-empty recent-articles-only feed.

Reads from data/raw/{prices,sec_facts,dividends,news_tone}.csv. Run
fetch_prices.py, fetch_fundamentals_sec.py, and fetch_news_gdelt.py first
(all need real internet - see README.md).

Two-pass design: pass 1 computes point-in-time fundamentals for every
(ticker, window) up front, to establish global normalization stats (so the
fundamentals channel is comparable across tickers/dates); pass 2 renders
and saves every sample.

Usage:
    python build_dataset.py
"""

import numpy as np
import pandas as pd

import config
from pit_fundamentals import TickerFundamentalsIndex, normalize_date_column
from render_channels import render_price_chart, render_fundamentals_channel, render_news_channel, stack_channels


def get_window_specs(prices: pd.DataFrame) -> list[dict]:
    """
    All valid (input window, target window) index ranges for one ticker's
    price history, without rendering anything yet.

    Each valid window START produces one spec PER horizon in
    config.HORIZON_DAYS_OPTIONS - same 1-year input, different forecast
    distance. A start is only used if there's enough future data for the
    LARGEST horizon, so every window start yields the full, consistent set
    of horizons (no ticker/date silently missing the longer-range targets).
    """
    n = len(prices)
    window, stride = config.INPUT_WINDOW_DAYS, config.STRIDE_DAYS
    max_horizon = max(config.HORIZON_DAYS_OPTIONS)
    specs = []
    for start in range(0, n - window - max_horizon + 1, stride):
        for horizon in config.HORIZON_DAYS_OPTIONS:
            specs.append({
                "input_start_idx": start, "input_end_idx": start + window,
                "target_start_idx": start + window, "target_end_idx": start + window + horizon,
                "horizon_days": horizon,
            })
    return specs


def main():
    import training_config as tc
    tc.refuse_rebuild_or_retrain()
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data ...")
    prices_all = pd.read_csv(config.PRICES_CSV)
    prices_all["Date"] = normalize_date_column(prices_all["Date"])
    sec_facts_all = pd.read_csv(config.SEC_FACTS_CSV) if config.SEC_FACTS_CSV.exists() else pd.DataFrame(
        columns=["Ticker", "Concept", "End", "Filed", "Val", "Form", "FiscalPeriod"]
    )
    dividends_all = pd.read_csv(config.DIVIDENDS_CSV) if config.DIVIDENDS_CSV.exists() else pd.DataFrame(
        columns=["Ticker", "Date", "DividendPerShare"]
    )
    news_tone_all = pd.read_csv(config.NEWS_TONE_CSV) if config.NEWS_TONE_CSV.exists() else pd.DataFrame(
        columns=["Ticker", "Date", "Tone"]
    )

    # ---------- Pass 1: compute point-in-time fundamentals for every window ----------
    print("\nPass 1/2: computing point-in-time fundamentals for every window ...")
    per_ticker_data = {}   # ticker -> {"prices":..., "specs":..., "idx":..., "fund_df":...}
    all_fund_rows = []

    for ticker in config.TICKERS:
        ticker_prices = prices_all[prices_all["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if ticker_prices.empty:
            continue

        specs = get_window_specs(ticker_prices)
        if not specs:
            continue

        ticker_facts = sec_facts_all[sec_facts_all["Ticker"] == ticker]
        ticker_divs = dividends_all[dividends_all["Ticker"] == ticker]
        fidx = TickerFundamentalsIndex(ticker, ticker_facts, ticker_prices, ticker_divs)

        # "as of" date for each window = the last day of the INPUT window
        # (i.e. everything up to and including time t) - never the target.
        as_of_dates = [ticker_prices.iloc[s["input_end_idx"] - 1]["Date"].strftime("%Y-%m-%d") for s in specs]
        fund_df = fidx.compute_for_dates(as_of_dates)

        per_ticker_data[ticker] = {"prices": ticker_prices, "specs": specs, "fund_df": fund_df}
        all_fund_rows.append(fund_df)
        print(f"  {ticker}: {len(specs)} windows")

    if not per_ticker_data:
        print("No tickers had usable price data. Did you run fetch_prices.py first?")
        return

    all_fundamentals = pd.concat(all_fund_rows, ignore_index=True)
    reference_stats = {}
    for field in config.FUNDAMENTALS_FIELDS:
        col = pd.to_numeric(all_fundamentals[field], errors="coerce").dropna()
        reference_stats[field] = (float(col.min()), float(col.max())) if len(col) else (0.0, 1.0)
    print(f"\nComputed reference stats for normalization across {len(all_fundamentals)} total (ticker, window) fundamentals rows")

    coverage = all_fundamentals["ebitda"].notna().mean() * 100
    print(f"Fundamentals coverage: {coverage:.1f}% of windows have real SEC-derived data (rest are pre-first-filing, correctly NaN -> neutral)")

    # ---------- Pass 2: render and save every sample ----------
    print("\nPass 2/2: rendering and saving samples ...")

    # CHECKPOINTED: if a manifest from a previous (interrupted) run already
    # exists, skip any ticker that's already fully represented in it, and
    # continue sample_id numbering from where it left off. Each ticker's
    # rows are appended to the manifest CSV immediately after that ticker
    # finishes - not accumulated in memory and written once at the very
    # end - so an interruption loses at most one ticker's worth of work,
    # not the whole run.
    already_done_tickers = set()
    sample_idx = 0
    if config.DATASET_MANIFEST.exists():
        existing_manifest = pd.read_csv(config.DATASET_MANIFEST)
        if not existing_manifest.empty:
            already_done_tickers = set(existing_manifest["ticker"].unique())
            sample_idx = int(existing_manifest["sample_id"].max()) + 1
            print(f"Resuming: {len(already_done_tickers)} tickers already in the manifest, "
                  f"continuing from sample_id {sample_idx}.")

    manifest_path_exists = config.DATASET_MANIFEST.exists()

    for ticker, data in per_ticker_data.items():
        if ticker in already_done_tickers:
            continue

        ticker_prices = data["prices"]
        specs = data["specs"]
        fund_df = data["fund_df"]
        ticker_news = news_tone_all[news_tone_all["Ticker"] == ticker]
        ticker_manifest_rows = []

        # Group specs by their shared input window (same input_start_idx) -
        # every horizon variant of the same window reuses the same X.
        specs_by_start = {}
        for i, spec in enumerate(specs):
            specs_by_start.setdefault(spec["input_start_idx"], []).append((i, spec))

        for input_start_idx, grouped in specs_by_start.items():
            first_i, first_spec = grouped[0]
            input_slice = ticker_prices.iloc[first_spec["input_start_idx"]: first_spec["input_end_idx"]]

            window_dates = input_slice["Date"].dt.strftime("%Y-%m-%d").tolist()
            window_start, window_end = window_dates[0], window_dates[-1]
            window_news = ticker_news[(ticker_news["Date"] >= window_start) & (ticker_news["Date"] <= window_end)]

            # X is identical across all horizons sharing this window - render once.
            price_ch = render_price_chart(input_slice)
            fund_row = fund_df.iloc[first_i].to_dict()
            fund_ch = render_fundamentals_channel(fund_row, reference_stats=reference_stats)
            news_ch = render_news_channel(
                window_news["Tone"].tolist(), window_news["Date"].tolist(), window_dates,
            )
            X = stack_channels(price_ch, fund_ch, news_ch)
            X_u8 = (np.clip(X, 0, 1) * 255).astype(np.uint8)

            # Y differs per horizon - render once per horizon in this group.
            for i, spec in grouped:
                target_slice = ticker_prices.iloc[spec["target_start_idx"]: spec["target_end_idx"]]
                target_start = target_slice["Date"].iloc[0].strftime("%Y-%m-%d")
                target_end = target_slice["Date"].iloc[-1].strftime("%Y-%m-%d")

                Y = render_price_chart(target_slice)
                Y_u8 = (np.clip(Y, 0, 1) * 255).astype(np.uint8)

                fname = f"sample_{sample_idx:06d}.npz"
                np.savez_compressed(config.SAMPLES_DIR / fname, X=X_u8, Y=Y_u8)

                ticker_manifest_rows.append({
                    "sample_id": sample_idx, "file": fname, "ticker": ticker,
                    "window_start": window_start, "window_end": window_end,
                    "horizon_days": spec["horizon_days"],
                    "target_start": target_start, "target_end": target_end,
                    "fundamentals_as_of": window_end,
                    "had_real_fundamentals": bool(pd.notna(fund_row.get("ebitda"))),
                    "news_articles_in_window": len(window_news),
                })
                sample_idx += 1

        # Append this ticker's rows to the manifest NOW, not at the end of
        # the whole run - this is what actually makes the checkpoint real.
        ticker_df = pd.DataFrame(ticker_manifest_rows)
        ticker_df.to_csv(config.DATASET_MANIFEST, mode="a", header=not manifest_path_exists, index=False)
        manifest_path_exists = True
        print(f"  {ticker}: {len(ticker_manifest_rows)} samples -> appended to manifest")

    manifest = pd.read_csv(config.DATASET_MANIFEST)

    print(f"\nBuilt {len(manifest)} samples total -> {config.SAMPLES_DIR}")
    print(f"Manifest -> {config.DATASET_MANIFEST}")
    print(f"\nSamples with real point-in-time fundamentals: {manifest['had_real_fundamentals'].mean() * 100:.1f}%")
    print(f"Samples with >=1 matched news day: {(manifest['news_articles_in_window'] > 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()