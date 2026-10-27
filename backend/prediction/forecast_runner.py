"""
Programmatic chart forecast for the web API (non-interactive).

Primary path for the website: **live inference** — fetch recent OHLCV via
yfinance, render price plus point-in-time SEC fundamentals and Form 8-K tone,
run the diffusion model, return a PNG.

Optional offline path: if `data/processed/manifest_split.csv` + samples
exist, use the prebuilt dataset window instead (research/eval path).

Checkpoint: checkpoints_stable/latest.pt only (epoch 25, the released weights).
Do not prefer best.pt — a later training run must not replace the demo.
(BASE_CHANNELS=24, DIFFUSION_TIMESTEPS=1000 — must match the shipped weights).
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROCESSED_DIR = _SCRIPT_DIR / "data" / "processed"
_SAMPLES_DIR = _PROCESSED_DIR / "samples"
_CHECKPOINT_DIR = _SCRIPT_DIR / "checkpoints_stable"

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

HORIZON_OPTIONS = [5, 21, 63, 126, 189, 252]
BASE_CHANNELS = 24
DIFFUSION_TIMESTEPS = 1000
# Full 1000-step DDPM is slow for UI; DDIM with ~150 steps keeps chart structure.
LIVE_INFERENCE_STEPS = 150
INPUT_WINDOW_DAYS = 252

# One DDIM draw is one sample from the model's distribution — it carries the
# model's signal *plus* whatever that particular noise seed contributed, which
# is why single-draw forecasts look arbitrary and change on every refresh.
# Drawing several and averaging their extracted paths cancels the per-draw
# noise and, as a bonus, the spread across draws is a real measure of how much
# the model actually agrees with itself. Drawn as one batch, so this costs far
# less than N sequential runs. Tune with AINVESTIFY_FORECAST_ENSEMBLE.
def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.environ.get(name, "").strip() or default)))
    except (TypeError, ValueError):
        return default


ENSEMBLE_SAMPLES = _env_int("AINVESTIFY_FORECAST_ENSEMBLE", 5, 1, 16)
# Expected high-low range of a driftless random walk over T is ~1.596·σ·√T.
# The generated frame is self-normalised to the target window's own high-low
# range (see the module note in `_ensemble_paths`), so this is what converts
# "position within the frame" back into "percent of price".
_RANGE_OVER_SIGMA = 1.596
# Trading days of real history drawn behind the forecast, as a multiple of
# the forecast horizon. Constant across horizons so every timeframe has the
# same history:forecast proportion — 1W and 1Y are then the same chart at
# different zoom levels, instead of each horizon having its own character.
# At 5:1 the forecast occupies ~17% of the plot width: clearly attached to
# real history (which is the point of drawing it this way) and still large
# enough to read.
#
# The floor is deliberately set to exactly 5 × the shortest horizon (5 days)
# so it never actually binds. An earlier 40-bar floor made 1W render at
# 7.6:1 while every other horizon sat at 3.9:1 — i.e. the one visible
# inconsistency across the six tabs.
_HISTORY_TO_FORECAST_RATIO = 5.0
_MIN_HISTORY_BARS = 25

# Live diffusion conditioning uses the same point-in-time inputs as training:
# Yahoo prices, SEC XBRL fundamentals filed on or before the window end, and
# Form 8-K item-code tone inside the window. A day with no 8-K stays neutral.
LIVE_CONDITIONING_NOTE = (
    "Fundamentals are the SEC values that were public on the window end. "
    "Form 8-K tone is the item-code score of filings in the window. "
    "A day with no filing stays neutral."
)

# The website serves the diffusion model so those two channels are actually used.
# Set AINVESTIFY_FORECAST_MODE=cone to serve the volatility cone instead.
FORECAST_MODE = os.environ.get("AINVESTIFY_FORECAST_MODE", "model").strip().lower()
# One band, not three. +/-1 sigma: measured coverage 67.8-72.5% against its 68%
# target across five horizons and ~34,000 windows, so "about two times in three"
# is a claim the chart can actually stand behind.
CONE_LEVEL = 0.68

# Uvicorn runs sync routes in a threadpool. Concurrent MPS inference can hang
# the whole process ("socket hang up" through the Vite proxy). Serialize + prefer CPU.
_INFER_LOCK = threading.Lock()
_MODEL_BUNDLE: dict[str, Any] | None = None
_RESULT_CACHE: dict[tuple[str, int], dict[str, Any]] = {}
_RESULT_CACHE_MAX = 32

# The released demo weights. best.pt is a training artifact and is not loaded.
CHECKPOINT_CANDIDATES = (
    _CHECKPOINT_DIR / "latest.pt",
)


def _import_config():
    import config

    return config


def _pick_checkpoint() -> Path | None:
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return None


def get_forecast_capabilities() -> dict[str, Any]:
    manifest_path = _PROCESSED_DIR / "manifest_split.csv"
    checkpoint = _pick_checkpoint()
    checkpoint_exists = checkpoint is not None
    manifest_exists = manifest_path.exists()
    return {
        "horizons_days": HORIZON_OPTIONS,
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_exists,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "checkpoint_exists": checkpoint_exists,
        "samples_dir": str(_SAMPLES_DIR),
        "live_inference": checkpoint_exists,
        "ready": checkpoint_exists,  # live path only needs the checkpoint
        "mode": "dataset" if (checkpoint_exists and manifest_exists) else ("live" if checkpoint_exists else "unavailable"),
        "conditioning_note": (
            LIVE_CONDITIONING_NOTE
            if FORECAST_MODE == "model"
            else "This cone is built from adjusted prices only. It does not read fundamentals or Form 8-K filings."
        ),
    }


def _get_device(torch_mod):
    """Pick inference device. Default CPU for web stability (override with AINVESTIFY_FORECAST_DEVICE)."""
    preferred = (os.environ.get("AINVESTIFY_FORECAST_DEVICE") or "cpu").strip().lower()
    if preferred == "cuda" and torch_mod.cuda.is_available():
        return torch_mod.device("cuda")
    if preferred == "mps" and getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        return torch_mod.device("mps")
    if preferred == "auto":
        if torch_mod.cuda.is_available():
            return torch_mod.device("cuda")
        if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
            return torch_mod.device("mps")
    return torch_mod.device("cpu")


def _get_model_bundle(torch_mod, ConditionalUNet, GaussianDiffusion):
    """Load checkpoint once; reuse across requests under _INFER_LOCK."""
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    checkpoint = _pick_checkpoint()
    if checkpoint is None:
        raise RuntimeError("Chart forecast checkpoint missing.")
    device = _get_device(torch_mod)
    ckpt = torch_mod.load(checkpoint, map_location=device, weights_only=False)
    film_x = any(k.startswith("x_encoder.") for k in ckpt["model_state"])
    model = ConditionalUNet(
        num_horizons=len(HORIZON_OPTIONS),
        base_channels=BASE_CHANNELS,
        film_x=film_x,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    diffusion = GaussianDiffusion(timesteps=DIFFUSION_TIMESTEPS, device=device)
    for attr in [
        "betas",
        "alphas_cumprod",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas",
        "posterior_variance",
    ]:
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    _MODEL_BUNDLE = {"model": model, "diffusion": diffusion, "device": device, "torch": torch_mod}
    return _MODEL_BUNDLE


def _load_torch_stack():
    try:
        import torch
        from diffusion import GaussianDiffusion
        from model import ConditionalUNet
    except ImportError as exc:
        raise RuntimeError(
            "Chart forecast dependencies missing. Install torch (+ backend/prediction deps)."
        ) from exc
    return torch, ConditionalUNet, GaussianDiffusion


def _select_sample(manifest, ticker: str, horizon_days: int):
    rows = manifest[
        (manifest["ticker"] == ticker.upper()) & (manifest["horizon_days"] == horizon_days)
    ].sort_values("window_end", ascending=False)
    if rows.empty:
        rows = manifest[manifest["ticker"] == ticker.upper()].sort_values("window_end", ascending=False)
    if rows.empty:
        raise ValueError(f"No samples for ticker {ticker.upper()} in manifest.")
    if horizon_days not in HORIZON_OPTIONS:
        raise ValueError(f"horizon_days must be one of {HORIZON_OPTIONS}")
    match = rows[rows["horizon_days"] == horizon_days]
    return (match if not match.empty else rows).iloc[0]


def _candle_count_for_horizon(horizon_days: int) -> int:
    """How many candles `render_price_chart` actually drew for this horizon
    during training. Mirrors `_resample_for_readability`'s own rule in
    render_channels.py exactly: <=60 rows stays daily (one candle per
    trading day); >60 rows gets resampled to weekly bars first. A trading
    -day window resamples to roughly `horizon_days * 0.2` calendar weeks."""
    if horizon_days <= 60:
        return max(2, int(horizon_days))
    return max(2, round(horizon_days * 0.2))


def _analyze_generated(generated_tensor, horizon_days: int) -> dict[str, Any]:
    """Shared per-candle read of the model's raw (1, 1, H, W) tensor in
    [-1, 1] — the one piece of ground truth every downstream consumer
    (`_ensemble_paths` for the price path, `_generated_to_png` for the
    fallback bitmap's colouring) builds on, so they can never disagree
    about which way the shape is headed.

    IMPORTANT: candle boundaries are computed from actual candle geometry
    (`_candle_count_for_horizon`), not naive equal pixel columns. An
    earlier version smoothed over a fixed 21px window and judged color per
    column — for wide-candle horizons (1W's 5 candles span ~85px each at
    the 512px display size) that window is much narrower than a single
    candle, so a candle's own internal noise (the wick sitting off-center
    from the body, model grain) could flip the reading partway through one
    candle, making it look like two half-width candles side by side. Slice
    edges here are aligned exactly to how `render_price_chart` laid the
    candles out (`xlim(-1, count)`, one candle per integer position), so
    every candle is read as exactly one unit no matter how wide it is on
    screen.
    """
    from PIL import Image, ImageFilter

    gen_u8 = (
        ((generated_tensor[0].squeeze(0).cpu().numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    )

    # Soften per-pixel diffusion grain at native resolution, then upsample
    # with LANCZOS (not matplotlib's default nearest-ish resampling) to a
    # fixed size — identical PNG dimensions every request, no more layout
    # jitter from content-dependent bbox cropping.
    img = Image.fromarray(gen_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=0.6))
    DISPLAY_PX = 512
    img = img.resize((DISPLAY_PX, DISPLAY_PX), Image.LANCZOS)

    # intensity: 1.0 = light/background, 0.0 = dark/ink. Training convention
    # (render_channels.py): up-candle bodies render white, down-candle
    # bodies + all wicks render black, default background is white.
    intensity = np.asarray(img, dtype=np.float32) / 255.0
    ink_weight = np.clip(1.0 - intensity, 0.0, 1.0)
    rows = np.arange(DISPLAY_PX, dtype=np.float32)[:, None]

    # Slice boundaries aligned to render_price_chart's own geometry
    # (`ax.set_xlim(-1, count)`, candle i centered at data-x = i) rather
    # than naive equal-width columns, so a boundary never lands inside a
    # candle's body even at the edges of the image.
    count = _candle_count_for_horizon(horizon_days)
    edges = np.empty(count + 1, dtype=np.float64)
    edges[0], edges[-1] = 0.0, float(DISPLAY_PX)
    for j in range(1, count):
        edges[j] = (j + 0.5) / (count + 1) * DISPLAY_PX

    slice_level = np.full(count, DISPLAY_PX / 2.0, dtype=np.float32)
    slice_ink = np.zeros(count, dtype=np.float32)
    for i in range(count):
        c0, c1 = int(round(edges[i])), max(int(round(edges[i])) + 1, int(round(edges[i + 1])))
        w = ink_weight[:, c0:c1]
        total = float(w.sum())
        slice_ink[i] = total
        if total > 1.0:
            slice_level[i] = float((rows * w).sum()) / total  # weighted-avg row; lower = higher price

    overall_mean = float(np.average(slice_level, weights=np.maximum(slice_ink, 1e-6)))

    return {
        "intensity": intensity,
        "ink_weight": ink_weight,
        "count": count,
        "edges": edges,
        "slice_level": slice_level,
        "slice_ink": slice_ink,
        "overall_mean": overall_mean,
        "display_px": DISPLAY_PX,
    }


def _generated_to_png(analysis: dict[str, Any]) -> bytes:
    """Render the shared per-candle analysis as a polished, in-palette PNG.

    This used to go straight through a bare matplotlib imshow: flat
    black/white/gray, a slate-colored spine that doesn't match the site's
    palette, and `bbox_inches="tight"` cropping to content (a different
    PNG size every request). Fixed with PIL upstream in `_analyze_generated`
    (denoise + upsample to a fixed size); this function just colors it.

    Color: a generated sample has no literal per-candle open/close (no
    structured OHLC survives diffusion sampling), so there's no ground
    truth "was this candle up or down" the way a real chart has. Instead
    this reads the shape the way a trader would glance at it — colors each
    candle green if its ink sits higher (price up) than the previous
    candle's and red if lower, matching the real green/red price chart
    elsewhere in the app instead of one flat color for the whole image.
    """
    from PIL import Image

    intensity = analysis["intensity"]
    ink_weight = analysis["ink_weight"]
    count = analysis["count"]
    edges = analysis["edges"]
    slice_level = analysis["slice_level"]
    overall_mean = analysis["overall_mean"]
    DISPLAY_PX = analysis["display_px"]

    # Color each candle by comparing it to the *previous* candle (the
    # closest proxy we have to "up or down" without literal OHLC); the
    # first candle compares against the overall mean level instead.
    green = np.array([15, 157, 88], dtype=np.float32)  # #0F9D58 --success
    red = np.array([217, 48, 37], dtype=np.float32)  # #D93025 --destructive
    bg = np.array([255, 255, 255], dtype=np.float32)  # #FFFFFF --card

    col_color = np.empty((DISPLAY_PX, 3), dtype=np.float32)
    for i in range(count):
        reference = overall_mean if i == 0 else slice_level[i - 1]
        color = green if slice_level[i] <= reference else red
        c0, c1 = int(round(edges[i])), max(int(round(edges[i])) + 1, int(round(edges[i + 1])))
        col_color[c0:c1] = color

    rgb = (
        col_color[None, :, :] * ink_weight[..., None]
        + bg[None, None, :] * intensity[..., None]
    )
    out = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), mode="RGB")

    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _ensemble_paths(generated_batch, horizon_days: int) -> dict[str, Any]:
    """Extract one normalised price *path* per ensemble member and reduce
    them to a mean path plus a per-point spread.

    `generated_batch` is (K, 1, H, W) — K independent DDIM draws for the
    same conditioning. Each draw is read by `_analyze_generated` into a
    per-candle ink level, then flipped into "level" units where 1.0 is the
    top of the frame (high price) and 0.0 the bottom.

    Each member is re-anchored to its *own* first candle before averaging.
    Members disagree about absolute placement within the frame (that
    placement is meaningless anyway — see below), so averaging raw levels
    would blur real shape into mush; averaging shapes-relative-to-their-
    own-start keeps the shape and cancels only the noise.

    IMPORTANT — what the vertical axis does and does not mean. The training
    targets were rendered by `render_price_chart`, which auto-scales
    `ax.set_ylim` to each target window's own min/max. A +5% move and a
    +50% move therefore render to a byte-identical image. So the frame
    carries *no* absolute scale: the model can express the shape of the
    path, but its magnitude has to be supplied from outside, by
    `_paths_to_prices` below.
    """
    k = int(generated_batch.shape[0])
    per_member = []
    first_analysis = None
    for i in range(k):
        analysis = _analyze_generated(generated_batch[i : i + 1], horizon_days)
        if first_analysis is None:
            first_analysis = analysis
        # slice_level is a row index (0 = top row = high price); flip it so
        # bigger number = higher price, and normalise to the frame.
        level = 1.0 - (np.asarray(analysis["slice_level"], dtype=np.float64) / float(analysis["display_px"]))
        per_member.append(level - level[0])  # re-anchor each member to its own start

    paths = np.vstack(per_member)  # (K, count)
    return {
        "analysis": first_analysis,
        "count": int(paths.shape[1]),
        "members": k,
        "mean_path": paths.mean(axis=0),
        "std_path": paths.std(axis=0) if k > 1 else np.zeros(paths.shape[1]),
    }


def _paths_to_prices(
    ensemble: dict[str, Any],
    last_price: float | None,
    daily_volatility: float | None,
    horizon_days: int,
) -> dict[str, Any] | None:
    """Give the model's (unitless) path a price axis.

    The model supplies the *shape*; this supplies the *scale*, because the
    scale was normalised out of the training targets (see `_ensemble_paths`).
    The conversion is deliberately simple and stated rather than tuned:

      frame height  ~=  1.596 · daily_vol · sqrt(horizon)

    so a move of `d` in normalised frame units is a move of `d × frame
    height` in percent. The path is already re-anchored at 0, so multiplying
    by the last close places the first step on the last observed price.

    The uncertainty band combines two independent sources in quadrature:
    how much the ensemble members disagree with each other (model
    uncertainty — small band means the model is confident, not that it is
    right), and the stock's own sqrt-time diffusion of price (market
    uncertainty, which grows with distance into the forecast). Reporting
    only the first would imply far more precision than exists.

    Returns None when there's no live price/volatility to anchor on (the
    offline dataset path), rather than inventing a number.
    """
    if not last_price or last_price <= 0:
        return None
    if not daily_volatility or daily_volatility <= 0:
        return None

    count = int(ensemble["count"])
    if count < 2:
        return None

    mean_path = np.asarray(ensemble["mean_path"], dtype=np.float64)
    std_path = np.asarray(ensemble["std_path"], dtype=np.float64)

    vol = float(daily_volatility)
    horizon = max(int(horizon_days), 1)
    # Percent-of-price that one full frame height represents.
    frame_pct = _RANGE_OVER_SIGMA * vol * float(np.sqrt(horizon))
    frame_pct = float(min(frame_pct, 1.5))  # ceiling: a 150%-tall frame is already extreme

    # Trading days elapsed at each candle (candles are evenly spaced across
    # the horizon, whether they're daily or weekly bars).
    days = np.array([horizon * (i + 1) / count for i in range(count)], dtype=np.float64)

    prices = last_price * (1.0 + mean_path * frame_pct)
    model_sigma = last_price * std_path * frame_pct
    market_sigma = last_price * vol * np.sqrt(days)
    total_sigma = np.sqrt(model_sigma**2 + market_sigma**2)

    band_low = np.maximum(prices - total_sigma, 0.0)
    band_high = prices + total_sigma

    # Sanity cap on the headline number: the shape can be extreme, the
    # stock's own history says how extreme is plausible.
    expected_move_pct = min(vol * float(np.sqrt(horizon)), 2.0)
    raw_change = float(prices[-1] / last_price - 1.0)
    predicted_pct_change = float(np.clip(raw_change, -2.0 * expected_move_pct, 2.0 * expected_move_pct))
    predicted_price = last_price * (1.0 + predicted_pct_change)

    # How much the ensemble members disagreed about the endpoint, as a
    # percent of price — a direct, honest readout of model confidence.
    spread_pct = float(std_path[-1] * frame_pct * 100.0)

    return {
        "last_price": round(float(last_price), 4),
        "predicted_price": round(float(predicted_price), 4),
        "predicted_pct_change": round(predicted_pct_change * 100.0, 2),
        "range_low": round(float(band_low[-1]), 4),
        "range_high": round(float(band_high[-1]), 4),
        "expected_move_pct": round(expected_move_pct * 100.0, 2),
        "ensemble_members": int(ensemble["members"]),
        "ensemble_spread_pct": round(spread_pct, 2),
        "prices": [round(float(p), 4) for p in prices],
        "band_low": [round(float(p), 4) for p in band_low],
        "band_high": [round(float(p), 4) for p in band_high],
        # Per-point uncertainty, kept split so the two sources stay legible:
        # `model_sigma` is how much the ensemble members disagreed at this
        # point (small = the model is self-consistent, which is NOT the same
        # as being right); `total_sigma` folds in the stock's own sqrt-time
        # price diffusion and is the half-width actually drawn as the band.
        "model_sigma": [round(float(s), 4) for s in model_sigma],
        "total_sigma": [round(float(s), 4) for s in total_sigma],
        "note": (
            "Illustrative estimate. The model supplies the shape of the path; "
            "the scale comes from this stock's own historical volatility, because "
            "magnitude was normalised out of the training targets. Not a "
            "calibrated price target."
        ),
    }


def _forecast_offsets(horizon_days: int, count: int) -> list[int]:
    """How many trading days after the anchor each forecast point sits.

    Point i lands `(i+1) * horizon / count` trading days out. Kept separate from
    `_forecast_dates` because the cone needs these offsets as *numbers* — the
    uncertainty at a point depends on how far out it is, so the band and the
    date label have to be derived from one shared definition or they drift.
    """
    count = max(1, int(count))
    horizon = max(1, int(horizon_days))

    offsets: list[int] = []
    for i in range(count):
        off = max(1, round((i + 1) * horizon / count))
        # Strictly increasing: at horizons where count > horizon the naive
        # rounding repeats a day, which would stack two points on one date.
        if offsets and off <= offsets[-1]:
            off = offsets[-1] + 1
        offsets.append(off)
    return offsets


def _forecast_dates(anchor_iso: str, horizon_days: int, count: int) -> list[str]:
    """The real trading day each generated candle refers to.

    Horizons are quoted in *trading* days (5 = one week, 252 = one year) and
    the candles are laid out evenly across them, so point i sits
    `(i+1) * horizon / count` trading days after the last real bar. An
    earlier version spread the points evenly over *calendar* dates instead,
    which put "day 3 of a 5-day forecast" on a Saturday — meaningless for a
    market, and wrong by up to two days.

    Business days are used as the trading-day proxy: correct except for
    market holidays, which would need an exchange calendar to handle and
    would shift only the tail of long horizons.

    Shared by the PNG renderer and the JSON series so the static image and
    the interactive chart can never disagree about where a point sits.
    """
    import pandas as pd

    offsets = _forecast_offsets(horizon_days, count)
    anchor = pd.Timestamp(anchor_iso)
    bdays = pd.bdate_range(start=anchor + pd.Timedelta(days=1), periods=offsets[-1])
    return [bdays[o - 1].date().isoformat() for o in offsets]


def _build_series(
    *,
    history_dates: list[str],
    history_closes: list[float],
    forecast: dict[str, Any],
    anchor_iso: str,
    horizon_days: int,
    max_history_points: int = 400,
) -> dict[str, Any]:
    """The same data the PNG plots, as JSON, so the frontend can draw it as
    an interactive chart instead of a flat image.

    History is downsampled to at most `max_history_points` for payload size
    — at 1Y the raw history is ~1260 daily bars, far more than any chart
    width can resolve. The final point is always kept regardless of where
    the stride lands, because it is the anchor the forecast attaches to;
    dropping it would visibly disconnect the two lines.
    """
    prices = forecast["prices"]
    dates = _forecast_dates(anchor_iso, horizon_days, len(prices))

    n = len(history_dates)
    if n > max_history_points:
        # Ceiling division, not floor: with n=630 and a 400 cap, `n // cap`
        # is 1 and nothing gets dropped at all. Any n between cap and 2×cap
        # would silently pass through unsampled.
        stride = max(1, -(-n // max_history_points))
        keep = sorted(set([*range(0, n, stride), n - 1]))
    else:
        keep = list(range(n))

    model_sigma = forecast.get("model_sigma") or [0.0] * len(prices)
    total_sigma = forecast.get("total_sigma") or [0.0] * len(prices)

    return {
        "history": [{"date": history_dates[i], "close": history_closes[i]} for i in keep],
        "forecast": [
            {
                "date": d,
                "price": p,
                "low": lo,
                "high": hi,
                # `sigma` is the half-width of the band at this point (model
                # disagreement and market diffusion combined); `model_sigma`
                # is the ensemble-disagreement part alone, which is the only
                # piece the model itself has an opinion about.
                "sigma": sig,
                "model_sigma": msig,
            }
            for d, p, lo, hi, sig, msig in zip(
                dates, prices, forecast["band_low"], forecast["band_high"], total_sigma, model_sigma
            )
        ],
        "anchor_date": history_dates[-1] if history_dates else anchor_iso,
        "anchor_price": forecast["last_price"],
    }


def _render_forecast_chart(
    *,
    ticker: str,
    horizon_days: int,
    history_dates: list[str],
    history_closes: list[float],
    forecast: dict[str, Any],
    window_end: str,
    target_end: str,
) -> bytes:
    """Draw the forecast as an actual chart: real history, a real price
    axis, the model's mean path, and a shaded uncertainty band.

    Replaces showing the raw generated bitmap. The bitmap is the model's
    signal *plus* diffusion grain rendered at 128×128 and blown up — it is
    the least readable form of the very same information this function
    plots properly. Everything drawn here comes from the same extracted
    per-candle levels the bitmap encoded, so nothing is invented; it is
    the same forecast, legible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.ticker import FuncFormatter

    INK = "#1A1A1A"
    MUTED = "#6B7280"
    GRID = "#E5E7EB"
    HIST = "#374151"
    GREEN = "#0F9D58"
    RED = "#D93025"

    prices = forecast["prices"]
    band_low = forecast["band_low"]
    band_high = forecast["band_high"]
    last_price = forecast["last_price"]

    hist_x = pd.to_datetime(pd.Series(history_dates))
    # Anchor on the last *plotted* history point, not on window_end, so the
    # forecast always starts exactly where the drawn line ends — no visual
    # gap if the two ever disagree.
    anchor_x = pd.Timestamp(history_dates[-1]) if history_dates else pd.Timestamp(window_end)
    fc_x = [
        pd.Timestamp(d)
        for d in _forecast_dates(anchor_x.date().isoformat(), int(horizon_days), len(prices))
    ]

    # Join the forecast onto the last real point so the line is continuous
    # rather than floating away from the history.
    line_x = [anchor_x, *fc_x]
    line_y = [last_price, *prices]
    fill_lo = [last_price, *band_low]
    fill_hi = [last_price, *band_high]

    # A cone has a deliberately flat centre, so colouring it green ("up") or red
    # ("down") would imply a direction it is explicitly not claiming. Neutral
    # slate instead; green/red stay for the model path, which does claim one.
    is_cone = forecast.get("method") == "volatility_cone"
    NEUTRAL = "#475569"
    up = prices[-1] >= last_price
    accent = NEUTRAL if is_cone else (GREEN if up else RED)

    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=100)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.plot(hist_x, history_closes, color=HIST, linewidth=1.5, solid_joinstyle="round", zorder=3)

    # One uncertainty band, drawn as three nested fills of the SAME band so its
    # edge fades instead of ending in a hard line. The fractions are a rendering
    # detail, not probability levels — the band itself is +/-1 sigma at every
    # point, widening with distance.
    for frac, alpha in ((1.0, 0.07), (0.66, 0.07), (0.33, 0.07)):
        lo = [p - (p - l) * frac for p, l in zip(line_y, fill_lo)]
        hi = [p + (h - p) * frac for p, h in zip(line_y, fill_hi)]
        ax.fill_between(line_x, lo, hi, color=accent, alpha=alpha, linewidth=0, zorder=1)

    ax.plot(
        line_x, line_y, color=accent, linewidth=2.0,
        linestyle=(0, (5, 2.5)), solid_capstyle="round", dash_capstyle="round", zorder=4,
    )

    # Endpoint markers, each with a white ring so they stay legible wherever
    # they land on top of the band.
    for x, y, c in ((anchor_x, last_price, INK), (line_x[-1], line_y[-1], accent)):
        ax.plot([x], [y], marker="o", markersize=7.5, color="#FFFFFF", zorder=5)
        ax.plot([x], [y], marker="o", markersize=4.5, color=c, zorder=6)

    ax.axvline(anchor_x, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)), alpha=0.55, zorder=2)
    ax.annotate(
        "now",
        xy=(anchor_x, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -11),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
    )

    # History wash, added AFTER autoscaling so it can be filled all the way
    # down to the axis floor. Filling to the data minimum instead leaves a
    # hard horizontal edge floating above the axis, which is the single thing
    # that most made this look unfinished. Matplotlib has no gradient fill, so
    # the fade is stacked bands: strongest just under the line, vanishing at
    # the baseline.
    y_lo, y_hi = ax.get_ylim()
    _FADE_STEPS = 7
    for i in range(_FADE_STEPS):
        top_f = 1.0 - i / _FADE_STEPS
        bot_f = 1.0 - (i + 1) / _FADE_STEPS
        y_top = [y_lo + (c - y_lo) * top_f for c in history_closes]
        y_bot = [y_lo + (c - y_lo) * bot_f for c in history_closes]
        ax.fill_between(
            hist_x, y_bot, y_top, color=INK,
            alpha=0.05 * (1.0 - i / _FADE_STEPS) + 0.008,
            linewidth=0, zorder=0,
        )
    ax.set_ylim(y_lo, y_hi)  # the fills can nudge autoscale; pin it back

    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)

    # Decide the tick format ONCE, from the approximate tick *spacing* rather
    # than the price magnitude. Deciding per tick value mixes "$85.00" and
    # "$115" on one axis; deciding by magnitude gives a $270 stock useless
    # trailing zeros while a stock in a tight range loses distinct ticks.
    plotted = [*history_closes, *fill_lo, *fill_hi]
    step = (max(plotted) - min(plotted)) / 5.0
    dec = 0 if step >= 1.0 else (1 if step >= 0.1 else 2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, d=dec: f"${v:,.{d}f}"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.tick_params(colors=MUTED, labelsize=8, length=0)

    horizon_label = {5: "1W", 21: "1M", 63: "3M", 126: "6M", 189: "9M", 252: "1Y"}.get(
        int(horizon_days), f"{horizon_days}d"
    )
    pct = forecast["predicted_pct_change"]
    # Typographic hierarchy instead of one long title line: label small and
    # muted, the number itself large and in the trend colour, supporting
    # detail small again underneath.
    title = f"{ticker} · {horizon_label} " + ("outcome range" if is_cone else "forecast")
    ax.set_title(title, loc="left", fontsize=9, color=MUTED, pad=26)
    # The figure itself stays in ink; the trend colour rides on the % change
    # beside it. Colouring the headline number means the text is carrying
    # series identity, which leaves nothing for the actual marks to say.
    ax.annotate(
        f"${prices[-1]:,.2f}",
        xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
        fontsize=15, fontweight="semibold", color=INK, va="bottom",
    )
    # The cone's centre is flat by construction, so "+0.00%" would be a
    # meaningless label dressed up as a result. Show nothing there instead.
    if not is_cone:
        ax.annotate(
            f"{pct:+.2f}%",
            xy=(0.0, 1.0), xycoords="axes fraction", xytext=(72, 11), textcoords="offset points",
            fontsize=9.5, fontweight="medium", color=accent, va="bottom",
        )
    # The subtitle has to describe whichever forecast produced this chart. The
    # cone has no ensemble to report agreement over — its honest summary is the
    # width of the band and the probability attached to it.
    if is_cone:
        # Escape the dollar signs: matplotlib reads a PAIR of unescaped '$' as
        # mathtext and silently swallows everything between them, which turned
        # "between $179.64 and $357.27" into "between 179.64and357.27".
        unc = forecast.get("uncertainty_pct") or [forecast.get("expected_move_pct", 0.0)]
        subtitle = (f"·   by then ±{float(unc[-1]):.1f}%: \\${band_low[-1]:,.2f} – "
                    f"\\${band_high[-1]:,.2f}   ·   about 2 times in 3")
    else:
        subtitle = (f"·   from ${last_price:,.2f}   ·   {forecast['ensemble_members']} samples, "
                    f"±{forecast['ensemble_spread_pct']:.1f}% agreement")
    ax.annotate(
        subtitle,
        xy=(0.0, 1.0), xycoords="axes fraction", xytext=(126, 11), textcoords="offset points",
        fontsize=8.5, color=MUTED, va="bottom",
    )

    fig.tight_layout(pad=1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _seed_for(ticker: str, horizon_days: int, window_end: str) -> int:
    """Stable seed per (ticker, horizon, last data date).

    Without this, every cache miss draws fresh noise and the "forecast" for
    the same stock visibly changes on each refresh — which correctly reads
    to a user as the model having no real opinion. Keying the seed to the
    last date of real data means the forecast is reproducible across server
    restarts and only moves when new market data actually arrives.
    """
    key = f"{ticker}|{int(horizon_days)}|{window_end}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def _cone_estimate(
    closes: np.ndarray,
    horizon_days: int,
    count: int,
) -> dict[str, Any]:
    """The honest forecast: a flat centre and a calibrated band around it.

    NO MODEL IS INVOLVED, and that is the point. Three measurements led here:

      backtest.py       the diffusion model has no measurable skill (direction
                        38.9%, p=0.48; predictions vary 8-16x less than reality)
      sensitivity.py    it does not read its input — rewriting the price history
                        moves the output 0.11x as much as re-rolling the noise,
                        so 98.9% of any two forecasts' difference is the seed
      predictability.py nothing about the PATH is predictable anyway. Over 30
                        tickers and 18 years: return OOS R2 -0.030, normalised
                        shape -0.003, direction worse than "always say up".
                        Only volatility scores, at +0.453.

    So the centre is flat — a random walk's optimal point forecast, and better
    out of sample than any fitted alternative measured — and all the information
    lives in the WIDTH, which is genuinely forecastable.

    Bands come from `volatility.forecast_variance_path`: today's volatility
    mean-reverting toward its long-run level, with Student-t quantiles for fat
    tails, applied in log space so a band can never imply a negative price.
    `backtest_cone.py` scores this at 1.8pp mean coverage error out of sample
    against 2.5pp for the trailing-std method it replaces.
    """
    from volatility import (
        EWMA_LAMBDA,
        GARCH_PERSISTENCE,
        LONG_RUN_DAYS,
        STUDENT_T_DF,
        _MAX_ANNUAL_VOL,
        _MIN_ANNUAL_VOL,
        _TRADING_DAYS,
        _effective_df,
        _t_quantile,
        ewma_variance,
        horizon_sigma,
    )

    closes = np.asarray(closes, dtype=float)
    closes = closes[np.isfinite(closes) & (closes > 0)]
    if closes.size < 60:
        raise ValueError(f"need at least 60 closes to build a cone, got {closes.size}")

    last_price = float(closes[-1])
    rets = np.diff(np.log(closes))
    offsets = _forecast_offsets(horizon_days, count)

    sigma_full = horizon_sigma(rets, max(offsets))
    sigma = np.array([sigma_full[o - 1] for o in offsets], dtype=float)

    # ONE band, and one uncertainty number PER FUTURE DAY.
    #
    # Uncertainty is not a property of the horizon, it is a property of each
    # point in it: day 1 of a week is a far safer bet than day 5. Reporting a
    # single figure for the whole tab threw that away and made a 1-year view look
    # as knowable at month one as at month twelve.
    #
    # Three nested probability bands were dropped for the same reason they were
    # added — legibility. Reading a chart should not require deciding which of
    # three shaded regions to believe. What survives is the +/-1 sigma band,
    # measured at 67.8-72.5% coverage against its 68% target across five horizons
    # and 34,000 windows (`backtest_cone.py`), i.e. "about two times in three".
    k = np.array([_t_quantile(0.5 + CONE_LEVEL / 2.0, _effective_df(o)) for o in offsets])
    low = last_price * np.exp(-k * sigma)
    high = last_price * np.exp(+k * sigma)
    # The band is lognormal, so it is slightly wider upward than downward. The
    # per-point figure is the half-width, and `low`/`high` carry the true ends —
    # a single "+/-" would quietly misstate the downside at long horizons.
    unc_pct = (high - low) / 2.0 / last_price * 100.0

    prices = [round(last_price, 4)] * len(offsets)
    # `sigma_full` is CUMULATIVE return sd, so sigma_full[-1]**2 is the total
    # variance over `h` days and sigma_full[-1]**2 / h is the mean daily
    # variance. Differencing sigma_full instead — which is what this line used to
    # do — differences standard deviations rather than variances and lands an
    # order of magnitude low (2.5% instead of 27% for AAPL).
    h_days = max(offsets)
    annual_vol = float(sigma_full[-1]) * float(np.sqrt(252.0 / h_days))

    finite = rets[np.isfinite(rets)]
    lr_n = min(int(finite.size), int(LONG_RUN_DAYS))
    long_run_var = float(np.var(finite[-lr_n:], ddof=1))
    ewma_var = float(ewma_variance(finite))
    today_var = 0.75 * ewma_var + 0.25 * long_run_var
    var_lo = (_MIN_ANNUAL_VOL / np.sqrt(_TRADING_DAYS)) ** 2
    var_hi = (_MAX_ANNUAL_VOL / np.sqrt(_TRADING_DAYS)) ** 2
    day1, day_h = int(offsets[0]), int(offsets[-1])

    def _ann_pct(daily_var: float) -> float:
        return round(float(np.sqrt(max(daily_var, 0.0) * _TRADING_DAYS) * 100.0), 2)

    calculation = {
        "ewma_lambda": EWMA_LAMBDA,
        "ewma_variance": round(ewma_var, 8),
        "ewma_vol_pct": _ann_pct(ewma_var),
        "long_run_days": lr_n,
        "long_run_variance": round(long_run_var, 8),
        "long_run_vol_pct": _ann_pct(long_run_var),
        "blend_fast": 0.75,
        "blend_long_run": 0.25,
        "today_variance": round(today_var, 8),
        "today_vol_pct": _ann_pct(float(np.clip(today_var, var_lo, var_hi))),
        "variance_clipped": bool(
            today_var < var_lo or today_var > var_hi or long_run_var < var_lo or long_run_var > var_hi
        ),
        "persistence": GARCH_PERSISTENCE,
        "student_t_df": STUDENT_T_DF,
        "confidence_level": CONE_LEVEL,
        "n_closes": int(closes.size),
        "n_log_returns": int(finite.size),
        "day1": {
            "day": day1,
            "df": round(_effective_df(day1), 2),
            "k": round(float(k[0]), 4),
            "sigma": round(float(sigma[0]), 6),
            "low": round(float(low[0]), 4),
            "high": round(float(high[0]), 4),
        },
        "horizon": {
            "day": day_h,
            "df": round(_effective_df(day_h), 2),
            "k": round(float(k[-1]), 4),
            "sigma": round(float(sigma[-1]), 6),
            "low": round(float(low[-1]), 4),
            "high": round(float(high[-1]), 4),
        },
    }

    return {
        "last_price": round(last_price, 4),
        "predicted_price": round(last_price, 4),
        "predicted_pct_change": 0.0,
        "range_low": round(float(low[-1]), 4),
        "range_high": round(float(high[-1]), 4),
        "expected_move_pct": round(float(unc_pct[-1]), 2),
        "annualised_vol_pct": round(annual_vol * 100.0, 2),
        "confidence_level": CONE_LEVEL,
        "prices": prices,
        "band_low": [round(float(v), 4) for v in low],
        "band_high": [round(float(v), 4) for v in high],
        # The per-point series the panel actually reads.
        "uncertainty_pct": [round(float(v), 2) for v in unc_pct],
        "sigma_log": [round(float(s), 6) for s in sigma],
        "total_sigma": [round(float(last_price * s), 4) for s in sigma],
        "model_sigma": [0.0] * len(offsets),
        "trading_days_out": [int(o) for o in offsets],
        "method": "volatility_cone",
        "calculation": calculation,
        "note": (
            "This is a range of outcomes, not a prediction. The centre line is "
            "today's price held flat, because no tested method forecasts direction "
            "better than that. The bands are calibrated from this stock's own "
            "volatility and are the part backed by evidence: measured coverage is "
            "within about 2 points of the stated probability over 18 years."
        ),
    }


def _fetch_live_history(
    ticker: str,
    min_bars: int = INPUT_WINDOW_DAYS,
    auto_adjust: bool = False,
):
    """Daily OHLC from Yahoo, cleaned, Date as a tz-naive column.

    Shared by the diffusion path and the cone path so they can never disagree
    about what "this stock's history" is — a divergence there would make the
    calibration measured in `backtest_cone.py` inapplicable to what ships.

    `auto_adjust` matters more than it looks. The cone estimates volatility from
    returns, and an UNADJUSTED series shows a fake drop on every ex-dividend
    date, which inflates measured volatility for dividend payers. The cone was
    calibrated on adjusted data (`predictability.py` fetches with
    auto_adjust=True), so it asks for adjusted here too — otherwise the shipped
    band is not the band that was measured. Yahoo applies adjustments backwards,
    so the most recent close is identical either way and the price shown to the
    user does not change. Defaults to False purely to leave the older diffusion
    path byte-identical to what it was trained and rendered against.
    """
    import pandas as pd
    import yfinance as yf

    sym = ticker.upper().replace(".", "-")
    end = datetime.utcnow().date()
    # ~2000 calendar days ≈ 1380 trading days — enough for the 1Y horizon's 5:1
    # history (252 × 5 = 1260 bars) and for the cone's 2-year long-run variance.
    # Stocks with a shorter listing history simply get everything available.
    start = end - timedelta(days=2000)
    hist = yf.download(
        sym,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=auto_adjust,
        threads=False,
    )
    if hist is None or hist.empty:
        raise ValueError(f"No price history available for {ticker.upper()}.")

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0] if isinstance(c, tuple) else c for c in hist.columns]

    need = ["Open", "High", "Low", "Close"]
    missing = [c for c in need if c not in hist.columns]
    if missing:
        raise ValueError(f"Price history for {ticker.upper()} missing columns: {missing}")

    hist = hist.dropna(subset=need).copy()
    hist = hist.reset_index()
    date_col = "Date" if "Date" in hist.columns else hist.columns[0]
    hist = hist.rename(columns={date_col: "Date"})
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)

    if len(hist) < min_bars:
        raise ValueError(
            f"Need at least {min_bars} trading days for {ticker.upper()}; got {len(hist)}."
        )
    return hist


def _finite_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _cached_sec_facts(ticker: str, cik: str):
    import pandas as pd
    from fetch_fundamentals_sec import SEC_FACTS_CACHE_DIR, extract_facts_table, fetch_company_facts

    SEC_FACTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = SEC_FACTS_CACHE_DIR / f"{ticker}.csv"
    if cache.exists() and cache.stat().st_size > 0:
        table = pd.read_csv(cache)
        if "Filed" in table.columns and len(table):
            return table
    raw = fetch_company_facts(cik)
    if raw is None:
        raise RuntimeError(f"SEC company facts are unavailable for {ticker}.")
    table = extract_facts_table(ticker, raw)
    table.to_csv(cache, index=False)
    return table


def _cached_8k_tone(ticker: str, cik: str, window_start: str, window_end: str):
    import pandas as pd
    from fetch_news_sec8k import SEC_8K_CACHE_DIR, extract_8k_rows, fetch_json

    SEC_8K_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = SEC_8K_CACHE_DIR / f"{ticker}.csv"
    if cache.exists() and cache.stat().st_size > 0:
        frame = pd.read_csv(cache)
    else:
        main = fetch_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
        if main is None:
            raise RuntimeError(f"SEC submissions are unavailable for {ticker}.")
        rows = extract_8k_rows(ticker, main.get("filings", {}).get("recent", {}))
        earliest = min((row["Date"] for row in rows if row.get("Date")), default=None)
        for older in main.get("filings", {}).get("files", []) or []:
            if earliest is not None and earliest <= window_start:
                break
            name = older.get("name")
            if not name:
                continue
            older_json = fetch_json(f"https://data.sec.gov/submissions/{name}")
            if older_json is not None:
                rows.extend(extract_8k_rows(ticker, older_json))
                earliest = min((row["Date"] for row in rows if row.get("Date")), default=earliest)
        frame = pd.DataFrame(rows, columns=["Ticker", "Date", "Tone"])
        if len(frame):
            frame = frame.groupby(["Ticker", "Date"], as_index=False)["Tone"].mean()
        frame.to_csv(cache, index=False)
    if frame.empty or "Date" not in frame.columns:
        return [], []
    frame["Date"] = frame["Date"].astype(str).str.slice(0, 10)
    window = frame[(frame["Date"] >= window_start) & (frame["Date"] <= window_end)]
    return window["Date"].tolist(), [float(v) for v in window["Tone"].tolist()]


def _live_dividends(ticker: str):
    import pandas as pd
    import yfinance as yf

    series = yf.Ticker(ticker).dividends
    if series is None or len(series) == 0:
        return pd.DataFrame(columns=["Date", "DividendPerShare"])
    dates = pd.to_datetime(series.index, utc=True).tz_localize(None)
    return pd.DataFrame({"Date": dates, "DividendPerShare": series.to_numpy(dtype=float)})


def _live_sec_channels(ticker: str, prices, window_dates: list[str], as_of: str):
    """Point-in-time fundamentals and 8-K tone for one live window.

    Fundamentals use only facts whose Filed date is on or before `as_of`.
    Tone uses 8-K item codes whose filing date falls inside the window.
    """
    from fetch_fundamentals_sec import fetch_cik_map
    from pit_fundamentals import TickerFundamentalsIndex

    config = _import_config()
    symbol = ticker.upper().replace(".", "-")
    cik = fetch_cik_map().get(symbol)
    if not cik:
        raise RuntimeError(f"No SEC company record for {symbol}.")
    facts = _cached_sec_facts(symbol, cik)
    index = TickerFundamentalsIndex(symbol, facts, prices, _live_dividends(symbol))
    row = index.compute_for_dates([as_of]).iloc[0]
    fundamentals = {field: _finite_or_none(row.get(field)) for field in config.FUNDAMENTALS_FIELDS}
    tone_dates, tone_values = _cached_8k_tone(symbol, cik, window_dates[0], window_dates[-1])
    return fundamentals, tone_dates, tone_values


def _build_live_conditioning(ticker: str, horizon_days: int) -> dict[str, Any]:
    """Build X from live prices plus point-in-time SEC fundamentals and 8-K tone."""
    import pandas as pd
    from render_channels import render_fundamentals_channel, render_news_channel, render_price_chart, stack_channels

    config = _import_config()
    hist = _fetch_live_history(ticker, INPUT_WINDOW_DAYS)

    window = hist.iloc[-INPUT_WINDOW_DAYS:].copy()
    window_start = window["Date"].iloc[0].date().isoformat()
    window_end = window["Date"].iloc[-1].date().isoformat()
    # Approximate target end as calendar days ≈ trading days * 1.4
    target_end = (window["Date"].iloc[-1] + timedelta(days=int(horizon_days * 1.45))).date().isoformat()

    # Anchor + scale for the price-estimate feature: today's real close and
    # this stock's own realized daily volatility over the input window.
    last_price = float(window["Close"].iloc[-1])
    daily_returns = window["Close"].pct_change().dropna()
    daily_volatility = float(daily_returns.std()) if len(daily_returns) >= 2 else None

    # Real history to plot the forecast as a continuation of, rather than as
    # a chart floating on its own. Length is a fixed MULTIPLE of the horizon
    # (not a fixed number of days), so every timeframe renders with the same
    # history-to-forecast proportion and therefore the same visual character
    # — 1W and 1Y look like the same chart at different zoom levels, instead
    # of 1W being mostly history and 1Y being mostly forecast.
    # Drawn from `hist` (the full fetch), not `window` (the model's 252-day
    # input), so long horizons aren't capped at one year of context.
    hist_n = int(
        min(len(hist), max(_MIN_HISTORY_BARS, round(horizon_days * _HISTORY_TO_FORECAST_RATIO)))
    )
    hist_slice = hist.iloc[-hist_n:]
    history_dates = [d.date().isoformat() for d in hist_slice["Date"]]
    history_closes = [round(float(c), 4) for c in hist_slice["Close"]]

    price_ch = render_price_chart(window[["Date", "Open", "High", "Low", "Close"]])
    window_dates = [d.date().isoformat() for d in window["Date"]]
    fund_values, tone_dates, tone_values = _live_sec_channels(ticker, hist, window_dates, window_end)
    fund_ch = render_fundamentals_channel(fund_values)
    news_ch = render_news_channel(tone_values, tone_dates, window_dates)

    X = stack_channels(price_ch, fund_ch, news_ch)
    X_u8 = (np.clip(X, 0.0, 1.0) * 255.0).astype(np.uint8)

    return {
        "X_u8": X_u8,
        "ticker": ticker.upper(),
        "horizon_days": int(horizon_days),
        "window_start": window_start,
        "window_end": window_end,
        "target_start": window_end,
        "target_end": target_end,
        "source": "live_yfinance",
        "img_size": int(config.IMG_HEIGHT),
        "last_price": last_price,
        "daily_volatility": daily_volatility,
        "history_dates": history_dates,
        "history_closes": history_closes,
    }


def _load_dataset_conditioning(ticker: str, horizon_days: int) -> dict[str, Any]:
    import pandas as pd

    config = _import_config()
    manifest = pd.read_csv(_PROCESSED_DIR / "manifest_split.csv")
    sample_row = _select_sample(manifest, ticker, horizon_days)
    data = np.load(_SAMPLES_DIR / sample_row.file)
    return {
        "X_u8": data["X"],
        "ticker": sample_row.ticker,
        "horizon_days": int(sample_row.horizon_days),
        "window_start": str(sample_row.window_start),
        "window_end": str(sample_row.window_end),
        "target_start": str(sample_row.target_start),
        "target_end": str(sample_row.target_end),
        "source": "dataset",
        "img_size": int(config.IMG_HEIGHT),
        # No live OHLCV anchor in the prebuilt dataset sample — price
        # estimate is skipped (not fabricated) for this path, and the
        # display falls back to the raw generated bitmap.
        "last_price": None,
        "daily_volatility": None,
        "history_dates": [],
        "history_closes": [],
    }


def _sample_ddim(diffusion, model, X_tensor, horizon_tensor, img_size: int, num_steps: int):
    """DDIM reverse process — correct fewer-step sampling (unlike naive DDPM skips)."""
    return diffusion.ddim_sample(
        model,
        X_tensor,
        horizon_tensor,
        img_size=img_size,
        channels=1,
        num_steps=num_steps,
        eta=0.0,
        guidance_scale=1.0,
    )

def run_cone_forecast(ticker: str, horizon_days: int = 63) -> dict[str, Any]:
    """The default path: an outcome cone built from price history alone.

    Needs no checkpoint, no torch and no GPU, so it cannot fail the way the
    diffusion path does — no OpenMP crash, no interpreter-version trap, and it
    returns in well under a second instead of running 150 DDIM steps.
    """
    if horizon_days not in HORIZON_OPTIONS:
        raise ValueError(f"horizon_days must be one of {HORIZON_OPTIONS}")

    sym = ticker.strip().upper()
    t0 = time.time()
    # 2y of long-run variance needs ~504 bars; ask for a bit more so the
    # estimate isn't built from the bare minimum.
    hist = _fetch_live_history(sym, min_bars=260, auto_adjust=True)
    closes_all = hist["Close"].to_numpy(dtype=float)
    dates_all = [d.date().isoformat() for d in hist["Date"]]

    # ONE POINT PER TRADING DAY — 5 for a week, 252 for a year.
    #
    # The diffusion path used `_candle_count_for_horizon`, which resampled to
    # weekly bars beyond 60 days because that is how the training images were
    # drawn. The cone inherits none of that: it evaluates the variance term
    # structure at whatever offsets it is asked for, so a 3-month view was
    # reporting 13 weekly points whose first entry sat 5 days out. Uncertainty is
    # a per-day quantity, so the series is now per-day at every horizon.
    count = int(horizon_days)
    estimate = _cone_estimate(closes_all, int(horizon_days), count)
    estimate["calculation"].update({
        "symbol": sym,
        "vendor": "Yahoo Finance",
        "download": "yfinance.download",
        "adjusted": True,
        "history_start": dates_all[0],
        "history_end": dates_all[-1],
        "bars_fetched": int(len(dates_all)),
        "min_bars_required": 260,
        "lookback_calendar_days": 2000,
    })

    # Same 5:1 context rule the diffusion chart used, so switching modes doesn't
    # change how much history the chart shows.
    hist_n = int(min(len(closes_all),
                     max(_MIN_HISTORY_BARS, round(horizon_days * _HISTORY_TO_FORECAST_RATIO))))
    history_dates = dates_all[-hist_n:]
    history_closes = [round(float(c), 4) for c in closes_all[-hist_n:]]
    window_end = dates_all[-1]

    series = _build_series(
        history_dates=history_dates,
        history_closes=history_closes,
        forecast=estimate,
        anchor_iso=window_end,
        horizon_days=int(horizon_days),
    )
    # Attach the per-day uncertainty to each point. This is the number the panel
    # leads with, so it travels with the point rather than being recomputed from
    # low/high in the client — one definition, one place.
    for i, pt in enumerate(series["forecast"]):
        pt["uncertainty_pct"] = estimate["uncertainty_pct"][i]
        pt["days_out"] = estimate["trading_days_out"][i]

    png_bytes = _render_forecast_chart(
        ticker=sym,
        horizon_days=int(horizon_days),
        history_dates=history_dates,
        history_closes=history_closes,
        forecast=estimate,
        window_end=window_end,
        target_end=series["forecast"][-1]["date"],
    )

    return {
        "ticker": sym,
        "horizon_days": int(horizon_days),
        "window_start": dates_all[max(0, len(dates_all) - INPUT_WINDOW_DAYS)],
        "window_end": window_end,
        "target_start": window_end,
        "target_end": series["forecast"][-1]["date"],
        "source": "live_yfinance",
        "inference_seconds": round(time.time() - t0, 2),
        "png_bytes": png_bytes,
        "render_mode": "cone",
        "forecast_mode": "cone",
        "ensemble_members": 0,
        "price_estimate": estimate,
        "series": series,
        "conditioning_note": (
            "This cone is built from adjusted prices only. It does not read fundamentals or Form 8-K filings."
        ),
        "cached": False,
    }


def run_forecast(ticker: str, horizon_days: int = 63) -> dict[str, Any]:
    """Generate a forecast PNG for ticker + horizon (live yfinance or offline dataset)."""
    if horizon_days not in HORIZON_OPTIONS:
        raise ValueError(f"horizon_days must be one of {HORIZON_OPTIONS}")

    sym = ticker.strip().upper()
    cache_key = (sym, int(horizon_days))
    cached = _RESULT_CACHE.get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    if FORECAST_MODE != "model":
        result = run_cone_forecast(sym, horizon_days)
        if len(_RESULT_CACHE) >= _RESULT_CACHE_MAX:
            _RESULT_CACHE.pop(next(iter(_RESULT_CACHE)))
        _RESULT_CACHE[cache_key] = {k: v for k, v in result.items() if k != "cached"}
        return result

    caps = get_forecast_capabilities()
    if not caps["checkpoint_exists"]:
        raise RuntimeError(
            "Chart forecast not ready: model checkpoint missing "
            "(expected backend/prediction/checkpoints_stable/latest.pt)."
        )

    # Build conditioning outside the lock (yfinance / disk I/O).
    if caps["manifest_exists"]:
        try:
            cond = _load_dataset_conditioning(sym, horizon_days)
        except ValueError:
            cond = _build_live_conditioning(sym, horizon_days)
    else:
        cond = _build_live_conditioning(sym, horizon_days)

    torch_mod, ConditionalUNet, GaussianDiffusion = _load_torch_stack()

    with _INFER_LOCK:
        cached = _RESULT_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

        bundle = _get_model_bundle(torch_mod, ConditionalUNet, GaussianDiffusion)
        model = bundle["model"]
        diffusion = bundle["diffusion"]
        device = bundle["device"]

        X_u8 = cond["X_u8"]
        horizon_idx = HORIZON_OPTIONS.index(int(cond["horizon_days"]))
        X_one = (
            torch_mod.from_numpy(X_u8.astype(np.float32) / 255.0) * 2.0 - 1.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        # Same conditioning, K independent noise draws — batched, so this is
        # one pass through the network per DDIM step rather than K passes.
        k = max(1, int(ENSEMBLE_SAMPLES))
        X_tensor = X_one.repeat(k, 1, 1, 1)
        horizon_tensor = torch_mod.full((k,), horizon_idx, device=device, dtype=torch_mod.long)

        # Reproducible: same ticker + horizon + last data date => same forecast.
        torch_mod.manual_seed(_seed_for(cond["ticker"], int(cond["horizon_days"]), str(cond["window_end"])))

        t0 = time.time()
        with torch_mod.no_grad():
            Y_generated = _sample_ddim(
                diffusion,
                model,
                X_tensor,
                horizon_tensor,
                img_size=int(cond["img_size"]),
                num_steps=LIVE_INFERENCE_STEPS,
            )
        elapsed = time.time() - t0

        ensemble = _ensemble_paths(Y_generated, int(cond["horizon_days"]))
        price_estimate = _paths_to_prices(
            ensemble,
            cond.get("last_price"),
            cond.get("daily_volatility"),
            int(cond["horizon_days"]),
        )

        # Preferred output: a real chart with a real price axis. Falls back
        # to the recoloured generated bitmap only when there's no live price
        # anchor to build a price axis from (offline dataset path).
        series = None
        if price_estimate is not None and cond.get("history_closes"):
            png_bytes = _render_forecast_chart(
                ticker=cond["ticker"],
                horizon_days=int(cond["horizon_days"]),
                history_dates=list(cond["history_dates"]),
                history_closes=list(cond["history_closes"]),
                forecast=price_estimate,
                window_end=str(cond["window_end"]),
                target_end=str(cond["target_end"]),
            )
            render_mode = "chart"
            # Same data as the PNG, as JSON — the frontend draws this
            # interactively and only falls back to the image if it's absent.
            series = _build_series(
                history_dates=list(cond["history_dates"]),
                history_closes=list(cond["history_closes"]),
                forecast=price_estimate,
                anchor_iso=str(cond["window_end"]),
                horizon_days=int(cond["horizon_days"]),
            )
        else:
            png_bytes = _generated_to_png(ensemble["analysis"])
            render_mode = "generated_bitmap"

        result = {
            "ticker": cond["ticker"],
            "horizon_days": int(cond["horizon_days"]),
            "window_start": cond["window_start"],
            "window_end": cond["window_end"],
            "target_start": cond["target_start"],
            # Prefer the last *actual* forecast trading day over the
            # conditioning's rough `horizon * 1.45` calendar estimate, so the
            # date shown to the user is the date the chart really ends on.
            "target_end": (
                series["forecast"][-1]["date"]
                if series and series.get("forecast")
                else cond["target_end"]
            ),
            "source": cond["source"],
            "inference_seconds": round(elapsed, 2),
            "png_bytes": png_bytes,
            "render_mode": render_mode,
            "ensemble_members": int(ensemble["members"]),
            "price_estimate": price_estimate,
            "series": series,
            "forecast_mode": "model",
            "conditioning_note": LIVE_CONDITIONING_NOTE,
            "cached": False,
        }
        if len(_RESULT_CACHE) >= _RESULT_CACHE_MAX:
            _RESULT_CACHE.pop(next(iter(_RESULT_CACHE)))
        _RESULT_CACHE[cache_key] = {k: v for k, v in result.items() if k != "cached"}
        return result


def _cli_main() -> int:
    """CLI entry for subprocess isolation from the FastAPI process."""
    import argparse
    import base64
    import json
    import warnings

    # Keep stdout pure JSON for the parent process.
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Run chart forecast and print JSON to stdout.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--horizon-days", type=int, default=63)
    args = parser.parse_args()

    try:
        result = run_forecast(args.ticker, horizon_days=args.horizon_days)
        png_bytes = result.pop("png_bytes")
        out = {
            **result,
            "image_base64": base64.b64encode(png_bytes).decode("ascii"),
            "content_type": "image/png",
        }
        sys.stdout.write(json.dumps(out))
        sys.stdout.flush()
        return 0
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
        # Also emit a JSON error object so the parent can surface a clear message.
        sys.stdout.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
