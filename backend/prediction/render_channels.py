"""
Render each modality (price chart, fundamentals, news) as a single-channel
image of shape (IMG_HEIGHT, IMG_WIDTH), so they can be stacked into one
multi-channel tensor per timestep.

Every function here is pure (no network calls) and can be unit-tested with
synthetic data — see test_render_channels.py.
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import config

def _fig_to_grayscale_array(fig, width: int, height: int) -> np.ndarray:
    """Render a matplotlib figure to a (height, width) float32 array in [0, 1]."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("L").resize((width, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _resample_for_readability(ohlcv: pd.DataFrame, max_bars: int = 60) -> pd.DataFrame:
    """
    If ohlcv has more rows than max_bars, resample to weekly OHLC bars so
    the chart stays visually readable at IMG_WIDTH=128px. A full 252-day
    year squeezed into 128px as daily candles would be under half a pixel
    per candle - unreadable. Resampled to weekly, that's ~52 bars, which
    renders cleanly.

    Requires a 'Date' column. If absent (legacy callers), returns ohlcv
    unchanged rather than failing - but every caller in this pipeline now
    passes Date, since real point-in-time windows need it anyway.
    """
    if "Date" not in ohlcv.columns or len(ohlcv) <= max_bars:
        return ohlcv

    df = ohlcv.set_index(pd.to_datetime(ohlcv["Date"]))
    weekly = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return weekly.reset_index(drop=True)


def render_price_chart(
    ohlcv: pd.DataFrame,
    width: int = config.IMG_WIDTH,
    height: int = config.IMG_HEIGHT,
) -> np.ndarray:
    """
    Render an OHLCV window as a candlestick-style chart image.

    ohlcv: DataFrame with columns Open, High, Low, Close (and ideally Date,
           for automatic weekly resampling on long windows), sorted by
           date, covering exactly the window to render (input window or
           target horizon — same function works for both, at any length).

    The y-axis is set from this window's own high and low, plus 5% padding.
    Two futures with the same shape and different amplitude therefore render
    to the same image.
    """
    if ohlcv.empty:
        return np.zeros((height, width), dtype=np.float32)

    ohlcv = _resample_for_readability(ohlcv)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    x = np.arange(len(ohlcv))
    opens = ohlcv["Open"].to_numpy()
    highs = ohlcv["High"].to_numpy()
    lows = ohlcv["Low"].to_numpy()
    closes = ohlcv["Close"].to_numpy()

    up = closes >= opens
    colors = np.where(up, "white", "black")  # rendered to grayscale, so encode
                                              # direction as light/dark, not RGB
    # wick
    ax.vlines(x, lows, highs, color="black", linewidth=0.6)
    # body
    body_bottom = np.minimum(opens, closes)
    body_height = np.abs(closes - opens)
    body_height = np.where(body_height < 1e-6, 1e-6, body_height)  # avoid zero-height bars
    ax.bar(x, body_height, bottom=body_bottom, width=0.6, color=colors, edgecolor="black", linewidth=0.4)

    ax.set_xlim(-1, len(ohlcv))
    lo, hi = float(np.min(lows)), float(np.max(highs))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.set_ylim(lo - pad, hi + pad)

    return _fig_to_grayscale_array(fig, width, height)


def render_fundamentals_channel(
    fundamentals: dict,
    width: int = config.IMG_WIDTH,
    height: int = config.IMG_HEIGHT,
    reference_stats: dict | None = None,
) -> np.ndarray:
    """
    Render a fundamentals snapshot as a horizontal-bar heatmap: one bar per
    field in config.FUNDAMENTALS_FIELDS, height proportional to the
    normalized value. This turns a numeric vector into a fixed, comparable
    visual pattern.

    reference_stats: optional {field: (min, max)} for normalization across
    the whole dataset, so bars are comparable across tickers/dates. If not
    given, falls back to a simple per-sample min-max (weaker, but usable
    for a first end-to-end test).
    """
    fields = config.FUNDAMENTALS_FIELDS
    values = []
    for f in fields:
        v = fundamentals.get(f)
        values.append(float(v) if v is not None else 0.0)
    values = np.array(values, dtype=np.float32)

    if reference_stats:
        norm = []
        for f, v in zip(fields, values):
            lo, hi = reference_stats.get(f, (0.0, 1.0))
            norm.append(0.0 if hi <= lo else np.clip((v - lo) / (hi - lo), 0, 1))
        norm = np.array(norm, dtype=np.float32)
    else:
        lo, hi = values.min(), values.max()
        norm = np.zeros_like(values) if hi <= lo else (values - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.bar(np.arange(len(fields)), norm, width=0.8, color="black")
    ax.set_xlim(-0.5, len(fields) - 0.5)
    ax.set_ylim(0, 1)

    return _fig_to_grayscale_array(fig, width, height)


def render_news_channel(
    tone_values: list[float],
    dates: list[str],
    window_dates: list[str],
    width: int = config.IMG_WIDTH,
    height: int = config.IMG_HEIGHT,
) -> np.ndarray:
    """
    Render SEC 8-K-derived daily tone scores over the input window as a
    horizontal intensity strip (one column-band per day in window_dates),
    repeated vertically to fill the channel. Days with no 8-K tone get
    neutral (0.5) intensity rather than being left blank or fabricated.

    tone_values / dates: parallel lists of daily tone (from fetch_news_sec8k.py)
    and its date (str, 'YYYY-MM-DD'), already filtered to the ticker.
    Tone is clipped to [-10, 10] and rescaled to [0, 1] here.
    window_dates: the exact list of trading-day date strings the input
    window spans, in order — defines the x-axis of the strip.
    """
    daily_tone = dict(zip(dates, tone_values))

    strip = np.full(len(window_dates), 0.5, dtype=np.float32)  # neutral default
    for i, d in enumerate(window_dates):
        if d in daily_tone and not np.isnan(daily_tone[d]):
            clipped = np.clip(daily_tone[d], -10, 10)
            strip[i] = (clipped + 10.0) / 20.0  # map [-10, 10] -> [0, 1]

    strip_2d = np.tile(strip[np.newaxis, :], (10, 1))
    img = Image.fromarray((strip_2d * 255).astype(np.uint8)).resize((width, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def stack_channels(price_ch: np.ndarray, fundamentals_ch: np.ndarray, news_ch: np.ndarray) -> np.ndarray:
    """Stack three (H, W) channels into one (H, W, 3) array."""
    return np.stack([price_ch, fundamentals_ch, news_ch], axis=-1)
