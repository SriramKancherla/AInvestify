"""Per-horizon held-out evaluation of the chart-diffusion checkpoint.

This is the harness the paper's per-horizon results table needs and the repo
did not have. It reports, for every forecast horizon:

    MAE, RMSE, SSIM, mean uncertainty, corr(U, |err|)

for the model *and* for the three baselines the paper declares but never
scores against — flat, drift, and persistence-chart. A result table without
the baselines is not interpretable, since a persistence chart is already a
strong image-space predictor.

Honest scoping, all of which lands in the output JSON:

  * Channels 1 and 2 are neutral placeholders here, exactly as
    `forecast_runner._build_live_conditioning` serves them in production.
    Building real point-in-time fundamentals and 8-K tone needs the SEC
    pipeline (`fetch_fundamentals_sec.py`, `fetch_news_sec8k.py`) to have
    been run. So this measures the price-channel regime, which is also the
    regime the deployed API actually runs in.
  * As-of dates default to a window after the estimated train/val cutoff of
    the shipped checkpoint (config.START_DATE 2015 .. END_DATE 2025 with a
    70/15/15 chronological split puts the test region at roughly 2023-08
    onward). Pass --as-of-start/--as-of-end to change it. The assumption is
    recorded in the JSON rather than hidden.
  * Targets are rendered by the same `render_price_chart` used at dataset
    build time, so <= 60 sessions stay daily and longer ones become weekly
    bars. Cross-horizon numbers therefore compare images of differing candle
    density; `candles` is reported per horizon so that is visible.

    python evaluate_horizons.py --tickers AAPL MSFT JPM XOM --dates 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import config  # noqa: E402
from render_channels import (  # noqa: E402
    render_fundamentals_channel,
    render_news_channel,
    render_price_chart,
    stack_channels,
)

HORIZONS = config.HORIZON_DAYS_OPTIONS
WINDOW = config.INPUT_WINDOW_DAYS
BASE_CHANNELS = 24          # must match the shipped weights, not training_config's default
DIFFUSION_TIMESTEPS = 1000

DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "NVDA", "PG"]
# After the estimated test-split boundary, and early enough that a full
# 252-session future still exists to score the one-year horizon against.
DEFAULT_ASOF_START = "2024-01-31"
DEFAULT_ASOF_END = "2025-06-30"


# ----------------------------------------------------------------- data
def fetch(ticker: str, cache: Path) -> pd.DataFrame | None:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{ticker}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"])
        return df
    import yfinance as yf

    hist = yf.download(ticker, start="2012-01-01", progress=False,
                       auto_adjust=False, threads=False)
    if hist is None or hist.empty:
        return None
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0] if isinstance(c, tuple) else c for c in hist.columns]
    hist = hist.dropna(subset=["Open", "High", "Low", "Close"]).reset_index()
    hist = hist.rename(columns={hist.columns[0]: "Date"})
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    hist.to_csv(path, index=False)
    return hist


def as_of_dates(hist: pd.DataFrame, start: str, end: str, n: int) -> list[int]:
    """Row indices of n evenly spaced sessions in [start, end] that have both
    a full input window behind them and the longest horizon ahead."""
    lo = max(WINDOW, int((hist["Date"] < pd.Timestamp(start)).sum()))
    hi = min(len(hist) - max(HORIZONS) - 1,
             int((hist["Date"] <= pd.Timestamp(end)).sum()) - 1)
    if hi <= lo:
        return []
    return [int(round(x)) for x in np.linspace(lo, hi, n)]


def candle_count(h: int) -> int:
    """Bars the renderer actually draws: <= 60 sessions stay daily, longer
    windows become weekly. Mirrors `_resample_for_readability`."""
    return h if h <= 60 else max(2, round(h * 0.2))


def conditioning(hist: pd.DataFrame, i: int) -> np.ndarray:
    """X for as-of row i. Channels 1-2 neutral, matching the production path."""
    window = hist.iloc[i - WINDOW + 1:i + 1]
    price = render_price_chart(window[["Date", "Open", "High", "Low", "Close"]])
    fund = render_fundamentals_channel({f: 0.0 for f in config.FUNDAMENTALS_FIELDS})
    news = render_news_channel([], [], [d.date().isoformat() for d in window["Date"]])
    X = stack_channels(price, fund, news)
    return (np.clip(X, 0.0, 1.0) * 255.0).astype(np.uint8)


# ----------------------------------------------------------------- baselines
def _synthetic_ohlc(closes: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    """Wrap a close series as an OHLC frame the renderer accepts."""
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "Date": dates.to_numpy(),
        "Open": opens,
        "High": np.maximum(opens, closes),
        "Low": np.minimum(opens, closes),
        "Close": closes,
    })


def baseline_charts(hist: pd.DataFrame, i: int, h: int) -> dict[str, np.ndarray]:
    """Flat, drift and persistence-chart references rendered like the target."""
    fut_dates = hist["Date"].iloc[i + 1:i + 1 + h]
    last = float(hist["Close"].iloc[i])

    flat = _synthetic_ohlc(np.full(h, last), fut_dates)

    trail = hist["Close"].iloc[i - h:i + 1].to_numpy(dtype=float)
    mu = float(np.mean(np.diff(np.log(trail)))) if len(trail) > 2 else 0.0
    drift = _synthetic_ohlc(last * np.exp(mu * np.arange(1, h + 1)), fut_dates)

    persistence = hist.iloc[i - h + 1:i + 1][["Date", "Open", "High", "Low", "Close"]]

    return {
        "flat": render_price_chart(flat),
        "drift": render_price_chart(drift),
        "persistence": render_price_chart(persistence),
    }


# ----------------------------------------------------------------- metrics
def score(pred01: np.ndarray, true01: np.ndarray) -> dict[str, float]:
    """MAE/RMSE on 8-bit intensities and SSIM, matching the paper's units."""
    from skimage.metrics import structural_similarity as ssim

    p = np.clip(pred01, 0, 1) * 255.0
    t = np.clip(true01, 0, 1) * 255.0
    err = np.abs(p - t)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(((p - t) ** 2).mean())),
        "ssim": float(ssim(t.astype(np.uint8), p.astype(np.uint8), data_range=255)),
        "_err": err,
    }


def unc_err_corr(unc: np.ndarray, err: np.ndarray) -> float:
    a, b = unc.ravel(), err.ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------- model
def load_model():
    import torch
    from diffusion import GaussianDiffusion
    from model import ConditionalUNet

    ckpt_dir = _SCRIPT_DIR / "checkpoints_stable"
    path = ckpt_dir / "best.pt" if (ckpt_dir / "best.pt").exists() else ckpt_dir / "latest.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ConditionalUNet(num_horizons=len(HORIZONS), base_channels=BASE_CHANNELS)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {"checkpoint": path.name, "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "hyperparameters": ckpt.get("hyperparameters")}
    return torch, model, GaussianDiffusion(timesteps=DIFFUSION_TIMESTEPS, device="cpu"), meta


def sample_all_horizons(torch, model, diffusion, X_u8, k: int, steps: int, seed: int):
    """One batch covering every horizon x every seed, so the 150 denoising
    steps are paid once instead of once per horizon."""
    torch.manual_seed(seed)
    X = torch.from_numpy(X_u8).float().permute(2, 0, 1) / 127.5 - 1.0
    n = len(HORIZONS) * k
    Xb = X.unsqueeze(0).repeat(n, 1, 1, 1)
    hb = torch.arange(len(HORIZONS)).repeat_interleave(k)
    out = diffusion.ddim_sample(model, Xb, hb, img_size=X_u8.shape[0],
                                channels=1, num_steps=steps)
    arr = ((out.squeeze(1).numpy() + 1.0) / 2.0).clip(0, 1)
    return {h: arr[j * k:(j + 1) * k] for j, h in enumerate(HORIZONS)}


# ----------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--dates", type=int, default=4, help="as-of dates per ticker")
    ap.add_argument("--as-of-start", default=DEFAULT_ASOF_START)
    ap.add_argument("--as-of-end", default=DEFAULT_ASOF_END)
    ap.add_argument("-k", "--ensemble", type=int, default=10)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--out", default=str(_SCRIPT_DIR / "eval_horizons_results.json"))
    args = ap.parse_args()

    torch, model, diffusion, meta = load_model()
    print(f"checkpoint {meta['checkpoint']} epoch={meta['epoch']} "
          f"val_loss={meta['val_loss']:.5f} hp={meta['hyperparameters']}")

    cache = _SCRIPT_DIR / "data" / "eval_cache"
    rows: list[dict] = []
    t0 = time.time()

    for ticker in args.tickers:
        hist = fetch(ticker, cache)
        if hist is None:
            print(f"  {ticker}: no data, skipped")
            continue
        idxs = as_of_dates(hist, args.as_of_start, args.as_of_end, args.dates)
        if not idxs:
            print(f"  {ticker}: no usable as-of dates in range, skipped")
            continue

        for i in idxs:
            as_of = hist["Date"].iloc[i].date().isoformat()
            X_u8 = conditioning(hist, i)
            draws = sample_all_horizons(torch, model, diffusion, X_u8,
                                        args.ensemble, args.steps, seed=abs(hash(as_of)) % 10000)

            for h in HORIZONS:
                fut = hist.iloc[i + 1:i + 1 + h]
                if len(fut) < h:
                    continue
                true01 = render_price_chart(fut[["Date", "Open", "High", "Low", "Close"]])

                d = draws[h]
                mean01, unc = d.mean(0), d.std(0)
                m = score(mean01, true01)
                base = {name: score(img, true01)
                        for name, img in baseline_charts(hist, i, h).items()}

                rows.append({
                    "ticker": ticker, "as_of": as_of, "horizon": h,
                    "candles": candle_count(h),
                    "mae": m["mae"], "rmse": m["rmse"], "ssim": m["ssim"],
                    "mean_u": float(unc.mean() * 255.0),
                    "corr_u_err": unc_err_corr(unc, m["_err"]),
                    "pairwise_disagreement": float(np.mean([
                        np.abs(d[a] - d[b]).mean() * 255.0
                        for a in range(len(d)) for b in range(a + 1, len(d))])),
                    **{f"{n}_{k2}": base[n][k2] for n in base for k2 in ("mae", "rmse", "ssim")},
                })
            print(f"  {ticker} {as_of}: done  ({time.time() - t0:.0f}s elapsed)")

    if not rows:
        print("no samples evaluated")
        return 1

    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    by_h = df.groupby("horizon")[list(num)].mean().drop(columns=["horizon"], errors="ignore")

    print(f"\n{'h':>5} {'n':>4} {'MAE':>7} {'RMSE':>7} {'SSIM':>7} "
          f"{'meanU':>7} {'corr':>7} | {'persSSIM':>9} {'flatSSIM':>9} {'driftSSIM':>9}")
    for h in HORIZONS:
        if h not in by_h.index:
            continue
        r = by_h.loc[h]
        n = int((df.horizon == h).sum())
        print(f"{h:>5} {n:>4} {r.mae:>7.2f} {r.rmse:>7.2f} {r.ssim:>7.3f} "
              f"{r.mean_u:>7.2f} {r.corr_u_err:>7.3f} | {r.persistence_ssim:>9.3f} "
              f"{r.flat_ssim:>9.3f} {r.drift_ssim:>9.3f}")
    o = df[list(num)].mean()
    print(f"{'mean':>5} {len(df):>4} {o.mae:>7.2f} {o.rmse:>7.2f} {o.ssim:>7.3f} "
          f"{o.mean_u:>7.2f} {o.corr_u_err:>7.3f} | {o.persistence_ssim:>9.3f} "
          f"{o.flat_ssim:>9.3f} {o.drift_ssim:>9.3f}")

    beats = {name: float((df.ssim > df[f"{name}_ssim"]).mean())
             for name in ("persistence", "flat", "drift")}
    print("\nshare of samples where the model's SSIM beats the baseline:")
    for name, v in beats.items():
        print(f"  vs {name:<12} {v * 100:5.1f}%")

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "checkpoint": meta,
        "cohort": {
            "tickers": args.tickers, "dates_per_ticker": args.dates,
            "as_of_start": args.as_of_start, "as_of_end": args.as_of_end,
            "n_samples": len(df), "ensemble_k": args.ensemble,
            "ddim_steps": args.steps,
        },
        "caveats": [
            "Channels 1 and 2 are neutral placeholders, matching the deployed "
            "live-inference path; real PIT fundamentals and 8-K tone require "
            "the SEC pipeline to have been run.",
            "As-of window is assumed to sit after the shipped checkpoint's "
            "train/val cutoff, estimated from config.START_DATE/END_DATE and "
            "make_splits' 70/15/15 chronological split.",
            "render_price_chart resamples windows longer than 60 sessions to "
            "weekly bars, so candle density differs across horizons.",
            "Targets are rendered with per-window y-axis auto-scaling, so "
            "these image metrics score shape only, not magnitude.",
        ],
        "by_horizon": {int(h): by_h.loc[h].to_dict() for h in by_h.index},
        "overall": o.to_dict(),
        "beats_baseline_ssim_rate": beats,
        "samples": rows,
    }
    for r in payload["samples"]:
        r.pop("_err", None)
    Path(args.out).write_text(json.dumps(payload, indent=1, default=float))
    print(f"\nwrote {args.out}  ({len(df)} samples, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
