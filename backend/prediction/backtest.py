"""
Point-in-time backtest for the chart-forecast model.

Answers one question: **does this model have any predictive skill, or does it
just produce plausible-looking chart shapes?**

Nothing else in this project measures that. Training reports an epsilon-MSE,
which is a denoising loss, not a forecasting loss -- a model can score well on
it by learning what stock charts look like in general while learning nothing
about the mapping from a particular input window to its particular future.
This script tests the thing that actually matters instead: run the model as of
a date in the past, then compare what it said against what really happened.

WHY 2026 DATES ARE A CLEAN TEST
config.END_DATE is 2025-12-31, so no 2026 outcome was ever in the training set.
Any as-of date in 2026 gives a genuinely out-of-sample result. (The 252-day
*input* window may reach back into the training range -- that is fine and
unavoidable; what must be unseen is the target, and it is.)

THE BASELINE IS THE WHOLE POINT
A forecast is only worth having if it beats the cheap alternative. Two are
measured here:

  flat      -- "the price does not move from here" (a random walk's optimal
               point forecast, and a genuinely hard benchmark over short
               horizons)
  drift     -- "the recent trend continues at the same rate"

A model that cannot beat `flat` has no demonstrated skill, no matter how
convincing its charts look. That comparison is reported first and plainly.

WHAT IS MEASURED
  direction   -- did the sign of the predicted move match reality?
                 (vs a 50% coin flip, with a binomial p-value)
  MAE / RMSE  -- absolute error in percent, against both baselines
  coverage    -- how often the actual close fell inside the +-1 sigma band.
                 A calibrated band should capture ~68%. Much less means the
                 band lies about its own confidence; much more means it is so
                 wide it says nothing.
  correlation -- between predicted and actual returns across all windows.
                 This is the sharpest signal: near zero means no relationship
                 at all, whatever the individual charts looked like.

Usage:
    # the single case: forecast the first week of April 2026 from end of March
    python backtest.py --ticker AAPL --as-of 2026-03-31 --horizon 5

    # a real skill measurement -- many tickers, many dates
    python backtest.py --sweep

    # results are always written to backtest_results.json as well
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Reuse the SHIPPED inference path rather than reimplementing it. If this
# import list ever drifts from forecast_runner, the backtest stops testing
# what the website actually serves -- which would make it worse than useless.
import forecast_runner as fr  # noqa: E402

DEFAULT_SWEEP_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "JPM", "XOM", "JNJ", "PG", "WMT", "KO",
]

# Abort a sweep after this many consecutive failures. A run where every window
# fails identically is an environment problem, not 216 separate bad windows.
_MAX_CONSECUTIVE_FAILURES = 3


_REQUIRED = ("torch", "matplotlib", "yfinance", "pandas", "numpy")


def _find_working_interpreter() -> Path | None:
    """Look for another interpreter that can actually import everything.

    Checks candidates by RUNNING them, not by inspecting site-packages — a
    directory being present proves nothing when the failure is a compiled
    extension refusing to load. This project's .venv turned out to hold two
    interpreters (3.11 and 3.13) with different package sets, and `bin/python`
    points at the one WITHOUT torch, so "use the venv" was useless advice.
    """
    import subprocess

    venv = _SCRIPT_DIR.parent.parent / ".venv"
    candidates: list[Path] = []
    if (venv / "bin").is_dir():
        candidates += sorted((venv / "bin").glob("python3.*"))

    here = Path(sys.executable).resolve()
    probe = "import " + ", ".join(_REQUIRED)
    for cand in candidates:
        try:
            if cand.resolve() == here:
                continue
            r = subprocess.run([str(cand), "-c", probe], capture_output=True, timeout=180)
            if r.returncode == 0:
                return cand
        except Exception:
            continue
    return None


def preflight(script: str | None = None) -> None:
    """Import everything the run needs, ONCE, before doing any work.

    Without this, an environment problem surfaces inside the per-window
    try/except and gets re-raised and re-printed for every single window in the
    sweep — hundreds of identical tracebacks burying the one line that matters.
    Fail fast, say what's wrong, say exactly how to fix it.

    `script` is the entry point being run, used only for the "re-run with" hint.
    It defaults to this file; callers in other modules must pass their own, or
    the hint names the wrong script — worse than no hint at all.
    """
    missing: list[tuple[str, str]] = []
    for name in _REQUIRED:
        try:
            __import__(name)
        except Exception as exc:  # ImportError, or a C-extension ABI failure
            missing.append((name, f"{type(exc).__name__}: {exc}"))
    if not missing:
        try:
            import render_channels  # noqa: F401
        except Exception as exc:
            missing.append(("render_channels", f"{type(exc).__name__}: {exc}"))
    if not missing:
        return

    joined = " ".join(m[1] for m in missing).lower()
    print("\nCannot run: this interpreter is missing or has broken dependencies.\n")
    print(f"  interpreter   {sys.executable}")
    print(f"  version       {sys.version.split()[0]}\n")
    for name, err in missing:
        print(f"  {name:<14} {err.splitlines()[0][:150]}")

    if "numpy" in joined or "_array_api" in joined or "initialization failed" in joined:
        print(
            "\n  Looks like a NumPy 1.x / 2.x ABI mismatch: a compiled package was built"
            "\n  against NumPy 1.x but NumPy 2.x is installed. That is an environment"
            "\n  problem — nothing in this script can work around it."
        )

    print("\n  Looking for an interpreter that can run this ...", flush=True)
    working = _find_working_interpreter()
    if working:
        args = " ".join(sys.argv[1:])
        entry = Path(script or __file__).resolve()
        print(f"\n  Found one. Re-run with:\n\n    {working} {entry} {args}\n")
        print(
            "  NOTE: your backend spawns the forecast subprocess with sys.executable, so it"
            "\n  needs this same interpreter. If uvicorn is started with a python that lacks"
            "\n  torch, the prediction endpoint fails too.\n"
        )
    else:
        print(
            "\n  None found. Install the missing packages into the interpreter you use for"
            "\n  the backend, or point it at one that already has them.\n"
        )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLCV, columns flattened, Date as a tz-naive column."""
    import yfinance as yf

    sym = ticker.upper().replace(".", "-")
    hist = yf.download(
        sym, start=start, end=end, progress=False, auto_adjust=False, threads=False
    )
    if hist is None or hist.empty:
        raise ValueError(f"No price history for {ticker}.")
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0] if isinstance(c, tuple) else c for c in hist.columns]

    need = ["Open", "High", "Low", "Close"]
    missing = [c for c in need if c not in hist.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")

    hist = hist.dropna(subset=need).reset_index()
    date_col = "Date" if "Date" in hist.columns else hist.columns[0]
    hist = hist.rename(columns={date_col: "Date"})
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    return hist


def build_conditioning_as_of(hist: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int) -> dict[str, Any]:
    """The live conditioning builder, frozen to a past date.

    NO LOOK-AHEAD: every row strictly after `as_of` is dropped before anything
    is computed, so the window, the last price and the volatility estimate all
    see exactly what a user standing on that date would have seen.
    """
    from render_channels import (
        render_fundamentals_channel,
        render_news_channel,
        render_price_chart,
        stack_channels,
    )

    config = fr._import_config()

    past = hist[hist["Date"] <= as_of]
    if len(past) < fr.INPUT_WINDOW_DAYS:
        raise ValueError(
            f"Only {len(past)} trading days on or before {as_of.date()}; "
            f"need {fr.INPUT_WINDOW_DAYS}."
        )

    window = past.iloc[-fr.INPUT_WINDOW_DAYS:].copy()
    last_price = float(window["Close"].iloc[-1])
    returns = window["Close"].pct_change().dropna()
    daily_vol = float(returns.std()) if len(returns) >= 2 else None

    price_ch = render_price_chart(window[["Date", "Open", "High", "Low", "Close"]])
    fund_ch = render_fundamentals_channel({f: 0.0 for f in config.FUNDAMENTALS_FIELDS})
    news_ch = render_news_channel([], [], [d.date().isoformat() for d in window["Date"]])
    X = stack_channels(price_ch, fund_ch, news_ch)

    return {
        "X_u8": (np.clip(X, 0.0, 1.0) * 255.0).astype(np.uint8),
        "window_start": window["Date"].iloc[0].date().isoformat(),
        "window_end": window["Date"].iloc[-1].date().isoformat(),
        "last_price": last_price,
        "daily_volatility": daily_vol,
        "img_size": int(config.IMG_HEIGHT),
        "window": window,
    }


# ---------------------------------------------------------------------------
# One forecast
# ---------------------------------------------------------------------------
def forecast_as_of(ticker: str, hist: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int) -> dict[str, Any]:
    """Run the shipped ensemble path as of `as_of`. Returns the price path."""
    import torch

    cond = build_conditioning_as_of(hist, as_of, horizon_days)

    torch_mod, ConditionalUNet, GaussianDiffusion = fr._load_torch_stack()
    bundle = fr._get_model_bundle(torch_mod, ConditionalUNet, GaussianDiffusion)
    model, diffusion, device = bundle["model"], bundle["diffusion"], bundle["device"]

    horizon_idx = fr.HORIZON_OPTIONS.index(int(horizon_days))
    X_one = (
        torch_mod.from_numpy(cond["X_u8"].astype(np.float32) / 255.0) * 2.0 - 1.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    k = max(1, int(fr.ENSEMBLE_SAMPLES))
    X_tensor = X_one.repeat(k, 1, 1, 1)
    horizon_tensor = torch_mod.full((k,), horizon_idx, device=device, dtype=torch_mod.long)

    # Same seeding rule as production, so a backtested forecast is exactly the
    # forecast the site would have shown on that date.
    torch_mod.manual_seed(fr._seed_for(ticker.upper(), int(horizon_days), cond["window_end"]))

    with torch.no_grad():
        Y = fr._sample_ddim(
            diffusion, model, X_tensor, horizon_tensor,
            img_size=int(cond["img_size"]), num_steps=fr.LIVE_INFERENCE_STEPS,
        )

    ensemble = fr._ensemble_paths(Y, int(horizon_days))
    est = fr._paths_to_prices(
        ensemble, cond["last_price"], cond["daily_volatility"], int(horizon_days)
    )
    if est is None:
        raise RuntimeError("Price mapping returned None (missing price/volatility anchor).")

    est["window_start"] = cond["window_start"]
    est["window_end"] = cond["window_end"]
    est["count"] = int(ensemble["count"])
    est["window"] = cond["window"]
    return est


def actual_path(hist: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int, count: int):
    """The real closes at the same trading-day offsets the forecast points sit at.

    Uses ACTUAL trading days from the price data rather than business days, so
    market holidays cannot silently shift a comparison by a day.
    Returns (dates, closes) or (None, None) if the future hasn't happened yet.
    """
    future = hist[hist["Date"] > as_of].reset_index(drop=True)
    offsets: list[int] = []
    for i in range(count):
        off = max(1, round((i + 1) * horizon_days / count))
        if offsets and off <= offsets[-1]:
            off = offsets[-1] + 1
        offsets.append(off)
    if len(future) < offsets[-1]:
        return None, None
    rows = [future.iloc[o - 1] for o in offsets]
    return [r["Date"].date().isoformat() for r in rows], [float(r["Close"]) for r in rows]


def baselines(window: pd.DataFrame, last_price: float, horizon_days: int, count: int) -> dict[str, list[float]]:
    """The forecasts worth beating.

    flat  -- price stays where it is. Optimal point forecast under a random
             walk, and the benchmark a real forecast has to clear.
    drift -- the last 60 trading days' average daily return, extrapolated.
    """
    steps = [ (i + 1) * horizon_days / count for i in range(count) ]
    recent = window["Close"].pct_change().dropna().tail(60)
    mu = float(recent.mean()) if len(recent) else 0.0
    return {
        "flat": [last_price] * count,
        "drift": [last_price * float(np.exp(mu * s)) for s in steps],
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_one(est: dict[str, Any], actual: list[float], horizon_days: int) -> dict[str, Any]:
    last = float(est["last_price"])
    pred = np.asarray(est["prices"], dtype=float)
    act = np.asarray(actual, dtype=float)
    lo = np.asarray(est["band_low"], dtype=float)
    hi = np.asarray(est["band_high"], dtype=float)

    base = baselines(est["window"], last, horizon_days, len(pred))
    flat = np.asarray(base["flat"], dtype=float)
    drift = np.asarray(base["drift"], dtype=float)

    def pct_err(p):
        return np.abs(p - act) / act * 100.0

    pred_ret = float(pred[-1] / last - 1.0)
    act_ret = float(act[-1] / last - 1.0)

    return {
        "predicted_end": float(pred[-1]),
        "actual_end": float(act[-1]),
        "last_price": last,
        "predicted_pct": pred_ret * 100.0,
        "actual_pct": act_ret * 100.0,
        "direction_correct": bool(np.sign(pred_ret) == np.sign(act_ret)) if act_ret != 0 else None,
        "mae_pct": float(pct_err(pred).mean()),
        "mae_pct_flat": float(pct_err(flat).mean()),
        "mae_pct_drift": float(pct_err(drift).mean()),
        "endpoint_err_pct": float(abs(pred[-1] - act[-1]) / act[-1] * 100.0),
        "endpoint_err_pct_flat": float(abs(flat[-1] - act[-1]) / act[-1] * 100.0),
        "coverage": float(np.mean((act >= lo) & (act <= hi))),
        "beat_flat": bool(pct_err(pred).mean() < pct_err(flat).mean()),
        "beat_drift": bool(pct_err(pred).mean() < pct_err(drift).mean()),
    }


def binomial_p_two_sided(successes: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test, no scipy dependency.

    Computed in log space via lgamma. The direct `comb(n, k) * p**k * ...` form
    is fine for the couple of dozen windows a sweep produces, but `comb` returns
    an unbounded int while `p**k` underflows to a float, and their product raises
    OverflowError somewhere past n of a few hundred — which is exactly the range
    `predictability.py` scores in. Same answer, no ceiling.
    """
    if n == 0:
        return 1.0
    from math import exp, lgamma, log

    if not 0 <= successes <= n:
        # lgamma of a non-positive integer raises an opaque "math domain error";
        # a caller that swapped the argument order deserves to be told so.
        raise ValueError(f"successes ({successes}) must be between 0 and n ({n}) inclusive")
    if not 0.0 < p < 1.0:
        return 1.0 if successes == (0 if p == 0.0 else n) else 0.0

    log_p, log_q = log(p), log(1.0 - p)
    log_nfac = lgamma(n + 1)

    def logpmf(k: int) -> float:
        return log_nfac - lgamma(k + 1) - lgamma(n - k + 1) + k * log_p + (n - k) * log_q

    obs = logpmf(successes)
    # Relative tolerance in log space, matching the original's intent of counting
    # outcomes "no more likely than the observed one" without dropping ties.
    return float(min(1.0, sum(exp(lk) for lk in map(logpmf, range(n + 1)) if lk <= obs + 1e-9)))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_single(ticker: str, as_of: str, horizon_days: int, est: dict[str, Any],
                  dates: list[str], actual: list[float], scores: dict[str, Any]) -> None:
    hl = "=" * 78
    print(f"\n{hl}\n {ticker}  |  forecast made as of {as_of}  |  horizon {horizon_days} trading days\n{hl}")
    print(f"  Model input window : {est['window_start']} -> {est['window_end']}  "
          f"({fr.INPUT_WINDOW_DAYS} trading days)")
    print(f"  Last close on {est['window_end']}: ${est['last_price']:,.2f}")
    print(f"  Ensemble           : {est['ensemble_members']} samples, "
          f"+-{est['ensemble_spread_pct']:.2f}% endpoint disagreement")

    print(f"\n  {'date':<12} {'predicted':>11} {'actual':>11} {'error':>9} {'error %':>9} {'in band':>8}")
    print(f"  {'-'*12} {'-'*11} {'-'*11} {'-'*9} {'-'*9} {'-'*8}")
    for d, p, a, lo, hi in zip(dates, est["prices"], actual, est["band_low"], est["band_high"]):
        err = p - a
        inb = "yes" if lo <= a <= hi else "NO"
        print(f"  {d:<12} {p:>11,.2f} {a:>11,.2f} {err:>+9,.2f} {err/a*100:>+8.2f}% {inb:>8}")

    print(f"\n  Predicted move : {scores['predicted_pct']:+.2f}%   "
          f"(${est['last_price']:,.2f} -> ${scores['predicted_end']:,.2f})")
    print(f"  Actual move    : {scores['actual_pct']:+.2f}%   "
          f"(${est['last_price']:,.2f} -> ${scores['actual_end']:,.2f})")
    print(f"  Direction      : {'CORRECT' if scores['direction_correct'] else 'WRONG'}")
    print(f"  Band coverage  : {scores['coverage']*100:.0f}% of days fell inside +-1 sigma")

    print(f"\n  {'':<26}{'mean abs error':>16}")
    print(f"  {'this model':<26}{scores['mae_pct']:>15.2f}%")
    print(f"  {'baseline: flat (no move)':<26}{scores['mae_pct_flat']:>15.2f}%")
    print(f"  {'baseline: drift continues':<26}{scores['mae_pct_drift']:>15.2f}%")
    verdict = "BEAT" if scores["beat_flat"] else "LOST TO"
    print(f"\n  => the model {verdict} the flat baseline on this window.")
    print("     (One window proves nothing either way -- run --sweep for a real answer.)")


def report_sweep(results: list[dict[str, Any]]) -> dict[str, Any]:
    hl = "=" * 78
    print(f"\n{hl}\n SWEEP SUMMARY -- {len(results)} forecasts\n{hl}")
    if not results:
        print("  No completed windows (not enough elapsed data). Try earlier --as-of dates.")
        return {}

    by_h: dict[int, list[dict[str, Any]]] = {}
    for r in results:
        by_h.setdefault(r["horizon_days"], []).append(r)

    print(f"\n  {'horizon':>8} {'n':>4} {'dir. acc':>9} {'p-value':>8} {'MAE%':>7} "
          f"{'flat%':>7} {'drift%':>7} {'beat flat':>10} {'coverage':>9} {'corr':>7}")
    print(f"  {'-'*8} {'-'*4} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*10} {'-'*9} {'-'*7}")

    summary = {}
    for h in sorted(by_h):
        rs = by_h[h]
        n = len(rs)
        dirs = [r["direction_correct"] for r in rs if r["direction_correct"] is not None]
        acc = float(np.mean(dirs)) if dirs else float("nan")
        pval = binomial_p_two_sided(int(sum(dirs)), len(dirs)) if dirs else float("nan")
        mae = float(np.mean([r["mae_pct"] for r in rs]))
        maef = float(np.mean([r["mae_pct_flat"] for r in rs]))
        maed = float(np.mean([r["mae_pct_drift"] for r in rs]))
        beat = float(np.mean([r["beat_flat"] for r in rs]))
        cov = float(np.mean([r["coverage"] for r in rs]))
        pr = [r["predicted_pct"] for r in rs]
        ar = [r["actual_pct"] for r in rs]
        corr = float(np.corrcoef(pr, ar)[0, 1]) if n > 2 and np.std(pr) > 0 and np.std(ar) > 0 else float("nan")
        print(f"  {h:>8} {n:>4} {acc*100:>8.1f}% {pval:>8.3f} {mae:>7.2f} {maef:>7.2f} "
              f"{maed:>7.2f} {beat*100:>9.0f}% {cov*100:>8.0f}% {corr:>7.3f}")
        summary[h] = {
            "n": n, "direction_accuracy": acc, "direction_p_value": pval,
            "mae_pct": mae, "mae_pct_flat": maef, "mae_pct_drift": maed,
            "beat_flat_rate": beat, "coverage": cov, "return_correlation": corr,
        }

    all_dirs = [r["direction_correct"] for r in results if r["direction_correct"] is not None]
    overall_acc = float(np.mean(all_dirs)) if all_dirs else float("nan")
    overall_p = binomial_p_two_sided(int(sum(all_dirs)), len(all_dirs)) if all_dirs else float("nan")
    beat_rate = float(np.mean([r["beat_flat"] for r in results]))
    cov = float(np.mean([r["coverage"] for r in results]))

    print(f"\n{hl}\n HOW TO READ THIS\n{hl}")
    print(f"  Direction accuracy overall : {overall_acc*100:.1f}%  (p = {overall_p:.3f} vs a coin flip)")
    print(f"  Beat the flat baseline in  : {beat_rate*100:.0f}% of windows")
    print(f"  Band coverage              : {cov*100:.0f}%  (a calibrated +-1 sigma band captures ~68%)")
    print()
    if overall_p > 0.05:
        print("  Direction is NOT statistically distinguishable from a coin flip.")
    else:
        print(f"  Direction IS statistically distinguishable from chance (p = {overall_p:.3f}).")
    if beat_rate < 0.5:
        print("  The model loses to 'assume no change' more often than it wins.")
        print("  That is the finding that matters: on this evidence it has no usable skill,")
        print("  regardless of how convincing the individual charts look.")
    else:
        print("  The model beats 'assume no change' more often than not -- worth pursuing.")
    if cov < 0.5:
        print("  Band is too NARROW: it claims more confidence than the errors justify.")
    elif cov > 0.85:
        print("  Band is too WIDE: it captures almost everything, so it says almost nothing.")
    else:
        print("  Band width is in a reasonable range.")

    return {
        "overall_direction_accuracy": overall_acc,
        "overall_direction_p_value": overall_p,
        "beat_flat_rate": beat_rate,
        "coverage": cov,
        "by_horizon": summary,
    }


def plot_single(ticker, as_of, horizon_days, est, dates, actual, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window = est["window"]
    tail = window.iloc[-max(25, horizon_days * 5):]
    hx = pd.to_datetime(tail["Date"])
    fx = pd.to_datetime(pd.Series(dates))
    anchor = hx.iloc[-1]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
    ax.plot(hx, tail["Close"], color="#374151", lw=1.5, label="History (model input)")
    ax.fill_between([anchor, *fx], [est["last_price"], *est["band_low"]],
                    [est["last_price"], *est["band_high"]], color="#2563EB", alpha=0.12, lw=0)
    ax.plot([anchor, *fx], [est["last_price"], *est["prices"]], color="#2563EB",
            lw=2, ls=(0, (5, 2.5)), label="Forecast")
    ax.plot([anchor, *fx], [est["last_price"], *actual], color="#0F9D58", lw=2, label="What actually happened")
    ax.axvline(anchor, color="#6B7280", lw=0.8, ls=(0, (2, 3)), alpha=0.6)

    ax.set_title(f"{ticker} · forecast as of {as_of} · {horizon_days} trading days ahead", fontsize=10)
    ax.grid(True, axis="y", color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.tick_params(labelsize=8, colors="#6B7280", length=0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\n  Chart written to {out_path}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--as-of", default="2026-03-31", help="Forecast is made as if standing on this date.")
    ap.add_argument("--horizon", type=int, default=5, choices=fr.HORIZON_OPTIONS)
    ap.add_argument("--sweep", action="store_true", help="Many tickers x dates x horizons.")
    ap.add_argument("--sweep-tickers", default=",".join(DEFAULT_SWEEP_TICKERS))
    ap.add_argument("--sweep-horizons", default="5,21,63")
    ap.add_argument("--sweep-dates", default="", help="Comma-separated as-of dates; default = monthly through 2026.")
    ap.add_argument("--out", default=str(_SCRIPT_DIR / "backtest_results.json"))
    args = ap.parse_args()

    if fr._pick_checkpoint() is None:
        print("No checkpoint found under checkpoints_stable/. Nothing to test.")
        return 1

    preflight()

    results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"generated_at": _now_iso()}

    if not args.sweep:
        as_of = pd.Timestamp(args.as_of)
        hist = fetch_history(
            args.ticker,
            (as_of - timedelta(days=2000)).date().isoformat(),
            (as_of + timedelta(days=int(args.horizon * 2.2) + 30)).date().isoformat(),
        )
        est = forecast_as_of(args.ticker, hist, as_of, args.horizon)
        dates, actual = actual_path(hist, as_of, args.horizon, est["count"])
        if actual is None:
            print(f"Not enough trading days after {args.as_of} to score a {args.horizon}-day horizon yet.")
            return 1
        scores = score_one(est, actual, args.horizon)
        report_single(args.ticker, args.as_of, args.horizon, est, dates, actual, scores)
        plot_single(args.ticker, args.as_of, args.horizon, est, dates, actual,
                    _SCRIPT_DIR / f"backtest_{args.ticker}_{args.as_of}_{args.horizon}d.png")
        payload["single"] = {
            "ticker": args.ticker, "as_of": args.as_of, "horizon_days": args.horizon,
            "dates": dates, "predicted": est["prices"], "actual": actual,
            "band_low": est["band_low"], "band_high": est["band_high"],
            **{k: v for k, v in scores.items()},
        }
    else:
        tickers = [t.strip().upper() for t in args.sweep_tickers.split(",") if t.strip()]
        horizons = [int(h) for h in args.sweep_horizons.split(",") if h.strip()]
        if args.sweep_dates.strip():
            dates_list = [d.strip() for d in args.sweep_dates.split(",") if d.strip()]
        else:
            # Month-ends through 2026, far enough back that outcomes exist.
            dates_list = [d.date().isoformat() for d in pd.date_range("2026-01-30", periods=6, freq="ME")]

        total = len(tickers) * len(horizons) * len(dates_list)
        done = 0
        consecutive_failures = 0
        aborted = False
        for ticker in tickers:
            if aborted:
                break
            try:
                hist = fetch_history(ticker, "2019-01-01", _today())
            except Exception as exc:
                print(f"  {ticker}: skipped ({exc})")
                done += len(horizons) * len(dates_list)
                continue
            for d in dates_list:
                if aborted:
                    break
                as_of = pd.Timestamp(d)
                for h in horizons:
                    done += 1
                    try:
                        est = forecast_as_of(ticker, hist, as_of, h)
                        dts, act = actual_path(hist, as_of, h, est["count"])
                        if act is None:
                            continue  # future hasn't happened yet — not a failure
                        sc = score_one(est, act, h)
                        sc.update({"ticker": ticker, "as_of": d, "horizon_days": h})
                        results.append(sc)
                        consecutive_failures = 0
                        print(f"  [{done}/{total}] {ticker} {d} +{h}d  "
                              f"pred {sc['predicted_pct']:+.2f}%  actual {sc['actual_pct']:+.2f}%  "
                              f"{'OK ' if sc['direction_correct'] else 'MISS'}", flush=True)
                    except Exception as exc:
                        consecutive_failures += 1
                        print(f"  [{done}/{total}] {ticker} {d} +{h}d  failed: "
                              f"{type(exc).__name__}: {exc}", flush=True)
                        if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                            # Every window failing the same way is one problem,
                            # not N problems. Stop and say so.
                            print(
                                f"\n  Aborting: {consecutive_failures} windows failed in a row with the "
                                f"same error.\n  This is an environment or setup problem, not a property "
                                f"of the data.\n  Fix the error above and re-run.\n"
                            )
                            aborted = True
                            break
        payload["aborted_early"] = aborted
        payload["sweep"] = results
        payload["summary"] = report_sweep(results)

    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n  Results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
