"""
What about a future price path is predictable AT ALL — with no neural network?

`sensitivity.py` showed the diffusion model does not read its input (input effect
= 0.11x the noise floor; 99% of any two forecasts' difference is the random seed).
That is consistent with two very different causes:

  (a) the conditioning path can't carry the signal  -> fix the architecture
  (b) the training TARGET contains almost no signal, so ignoring the input is the
      correct thing to have learned                 -> fix the data pipeline

Under (b), no architecture change produces skill. So before committing days to a
retrain, measure the ceiling: how much of the future is predictable from the past
under each candidate target definition, using plain regression on real prices.

Whatever a linear model on sensible features cannot find, a diffusion model
rendering the same information as a 128x128 bitmap is very unlikely to find. This
is a generous upper bound on the achievable signal, not a lower bound.

TARGETS COMPARED (horizon h trading days from t)
  return          close[t+h]/close[t] - 1
                  What a FIXED-RETURN-SCALE rendering would encode. The
                  documented rebuild plan aims here.
  direction       sign of the above — scored against the BASE RATE (equities
                  drift up, so "always say up" already beats a coin flip) and
                  against the date-block null, never against 50%.
  shape           the path close[t+1..t+h] min-max normalised to [0,1].
                  What the CURRENT pipeline encodes: build_dataset.py renders
                  targets with render_price_chart, which auto-scales ax.set_ylim
                  to each target window's own min/max, so only shape survives.
  abs_return      |return| — size of the move regardless of direction.
  log_volatility  log of the realised daily vol over t+1..t+h. What finance says
                  IS predictable, because volatility clusters.

If log_volatility scores well and return/shape score ~0, the answer is that the
predictable part of a price path is its WIDTH, not its direction or its shape —
and the volatility cone already in the panel (69% measured coverage against a 68%
target) is the correct product, with the model contributing nothing.

HOW IT AVOIDS FOOLING ITSELF
  * non-overlapping windows (step = horizon), so samples don't share days
  * time-ordered split with a gap, so no future leaks into training
  * ridge alpha chosen on an inner split of TRAIN only
  * out-of-sample R^2 against the TRAIN mean — the number a real forecaster
    could have beaten, so 0 means "no better than assuming the average"
  * a SHUFFLE NULL that permutes whole dates, keeping all tickers on a date
    together. Stocks move together, so a naive per-row shuffle would break that
    correlation and make the null far too easy to beat. Every R^2 is reported
    against this null, and the p-value is the fraction of shuffles that did
    better than the real data.
  * a MINIMUM EFFECT SIZE as well as a p-value, because at thousands of windows
    an R^2 of 0.001 is "significant" and still worthless.
  * direction compared to the free "always guess the commoner sign" baseline,
    not to 50%, and never with an exact binomial over correlated windows.

Usage:
    python predictability.py                          # real prices, 21d
    python predictability.py --horizon 5
    python predictability.py --synthetic              # self-test, see below

SELF-TEST
`--synthetic` runs the identical pipeline on generated series with GARCH-style
volatility clustering and NO predictable drift. The right answer is known in
advance: volatility should score well above the null, return and shape should sit
on it. If the script cannot reproduce that, its verdict on real data means
nothing — so run it once before believing anything here.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# NOTE: deliberately NOT using backtest.binomial_p_two_sided here. An exact
# binomial treats every window as an independent flip, which is false for a
# panel where all tickers share a date. Direction is scored against the
# date-block null and the base rate instead — see score_target.

_TTY = sys.stdout.isatty()


def _progress(msg: str, done: bool = False) -> None:
    """Overwrite-in-place counter, but only on a real terminal. Piped or
    redirected, \\r doesn't overwrite, so hundreds of counter updates end up
    interleaved with the results table and the output can't be read or pasted."""
    if not _TTY:
        return
    print(("\r" + " " * 30 + "\r") if done else f"\r{msg}", end="", flush=True)

PANEL = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "ADBE",
    "JPM", "BAC", "GS", "V", "XOM", "CVX", "COP",
    "JNJ", "PFE", "UNH", "MRK", "PG", "KO", "PEP", "WMT", "COST",
    "CAT", "BA", "HON", "T", "VZ", "NEE",
]
TARGETS = ("return", "shape", "abs_return", "log_volatility")
_FEATURE_LOOKBACKS = (5, 21, 63, 126, 252)

# Statistical significance is not the same as usable signal, and at these sample
# sizes the difference decides the verdict. The 5-day self-test throws thousands
# of windows at the null, which makes it tight enough that an R^2 of 0.0014 —
# 0.14% of variance explained, useless to anyone — clears p < 0.05. A result has
# to pass BOTH gates below to count.
#
# (That 0.0014 is not even noise: arithmetic returns of a log-normal process
# carry a Jensen term, exp(sigma^2/2) - 1, that grows with volatility. Since
# volatility is predictable, a sliver of arithmetic return is too. Real, correct,
# and far too small to trade or to build a product on.)
_USABLE_R2 = 0.01   # 1% of out-of-sample variance; below this, nothing to build on
_ALPHA = 0.05


def classify(r2: float, p: float) -> str:
    if p >= _ALPHA or r2 <= 0:
        return "no signal"
    return "SIGNAL" if r2 >= _USABLE_R2 else "too small to use"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fetch_panel(tickers: list[str], start: str, end: str, cache: Path) -> dict[str, pd.DataFrame]:
    """Daily closes per ticker, cached to CSV so re-runs cost nothing."""
    if cache.exists():
        raw = pd.read_csv(cache, parse_dates=["Date"])
        out = {t: g[["Date", "Close"]].reset_index(drop=True)
               for t, g in raw.groupby("ticker") if len(g) > 400}
        print(f"  loaded {len(out)} tickers from cache {cache.name}")
        return out

    import yfinance as yf

    frames, failed = [], []
    for i, t in enumerate(tickers, 1):
        sym = t.upper().replace(".", "-")
        try:
            h = yf.download(sym, start=start, end=end, progress=False,
                            auto_adjust=True, threads=False)
            if h is None or h.empty:
                failed.append(t)
                continue
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = [c[0] if isinstance(c, tuple) else c for c in h.columns]
            h = h.dropna(subset=["Close"]).reset_index()
            date_col = "Date" if "Date" in h.columns else h.columns[0]
            h = h.rename(columns={date_col: "Date"})
            h["Date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None)
            h["ticker"] = t
            frames.append(h[["Date", "Close", "ticker"]])
        except Exception as exc:
            failed.append(f"{t} ({type(exc).__name__})")
        _progress(f"  fetched {i}/{len(tickers)}")
    _progress("", done=True)
    print(f"  fetched {len(frames)}/{len(tickers)} tickers")
    if failed:
        print(f"  skipped: {', '.join(str(f) for f in failed)}")
    if not frames:
        raise RuntimeError("No price data fetched. Check the network and try again.")

    allf = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    allf.to_csv(cache, index=False)
    print(f"  cached to {cache.name}")
    return {t: g[["Date", "Close"]].reset_index(drop=True) for t, g in allf.groupby("ticker")}


def synthetic_panel(n_tickers: int = 30, n_days: int = 3800, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Series with GARCH-style volatility clustering and NO predictable drift.

    Ground truth by construction: today's volatility carries information about
    tomorrow's (the GARCH recursion), returns do not. A market factor is included
    so that, as in reality, all names move together on a given day — otherwise the
    date-block shuffle null would have nothing to preserve and the self-test would
    not exercise the part of the method most likely to be wrong.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-03-31", periods=n_days)
    omega, alpha, beta = 1e-6, 0.09, 0.90       # persistent but stationary
    market = np.zeros(n_days)
    sig2 = np.full(n_days, omega / (1 - alpha - beta))
    for i in range(1, n_days):
        sig2[i] = omega + alpha * market[i - 1] ** 2 + beta * sig2[i - 1]
        market[i] = rng.normal(0, np.sqrt(sig2[i]))

    out = {}
    for k in range(n_tickers):
        s2 = np.full(n_days, omega / (1 - alpha - beta))
        r = np.zeros(n_days)
        for i in range(1, n_days):
            s2[i] = omega + alpha * r[i - 1] ** 2 + beta * s2[i - 1]
            r[i] = 0.6 * market[i] + rng.normal(0, np.sqrt(s2[i]))
        out[f"SYN{k:02d}"] = pd.DataFrame(
            {"Date": dates, "Close": 100.0 * np.exp(np.cumsum(r))})
    return out


# ---------------------------------------------------------------------------
# Features and targets
# ---------------------------------------------------------------------------
def _features(closes: np.ndarray, t: int) -> np.ndarray | None:
    """Everything known at the close of day t. Never touches t+1 or later."""
    if t < max(_FEATURE_LOOKBACKS) + 1:
        return None
    logp = np.log(closes[: t + 1])
    rets = np.diff(logp)
    f: list[float] = []
    for lb in _FEATURE_LOOKBACKS:
        f.append(float(logp[-1] - logp[-1 - lb]))          # momentum
        f.append(float(np.std(rets[-lb:], ddof=1)))        # realised vol
    short, long = np.std(rets[-21:], ddof=1), np.std(rets[-252:], ddof=1)
    f.append(float(short / long) if long > 0 else 1.0)     # vol-of-vol regime
    f.append(float(np.mean(np.abs(rets[-21:]))))
    win = closes[t - 251: t + 1]
    rng_ = float(win.max() - win.min())
    f.append(float((closes[t] - win.min()) / rng_) if rng_ > 0 else 0.5)
    peak = np.maximum.accumulate(closes[t - 62: t + 1])
    f.append(float((closes[t] - peak[-1]) / peak[-1]))     # drawdown
    r63 = rets[-63:]
    sd = np.std(r63, ddof=1)
    f.append(float(np.mean((r63 - r63.mean()) ** 3) / sd**3) if sd > 0 else 0.0)
    arr = np.asarray(f, dtype=float)
    return arr if np.isfinite(arr).all() else None


def _targets(closes: np.ndarray, t: int, h: int) -> dict[str, Any] | None:
    """Everything about days t+1 .. t+h. Never touches day t or earlier."""
    if t + h >= len(closes):
        return None
    path = closes[t + 1: t + h + 1]
    if len(path) != h or not np.isfinite(path).all() or path.min() <= 0:
        return None
    ret = float(path[-1] / closes[t] - 1.0)
    lo, hi = float(path.min()), float(path.max())
    shape = (path - lo) / (hi - lo) if hi > lo else np.full(h, 0.5)
    daily = np.diff(np.log(np.concatenate([[closes[t]], path])))
    vol = float(np.std(daily, ddof=1))
    if vol <= 0:
        return None
    return {"return": ret, "abs_return": abs(ret),
            "log_volatility": float(np.log(vol)), "shape": shape.astype(float)}


def build_samples(panel: dict[str, pd.DataFrame], horizon: int) -> dict[str, Any]:
    """Non-overlapping windows on a shared date grid, so no two samples share a day."""
    X, dates, tickers = [], [], []
    ys: dict[str, list] = {k: [] for k in TARGETS}
    for tk, df in panel.items():
        closes = df["Close"].to_numpy(dtype=float)
        dts = df["Date"].to_numpy()
        start = max(_FEATURE_LOOKBACKS) + 1
        for t in range(start, len(closes) - horizon, horizon):   # step = horizon
            f = _features(closes, t)
            if f is None:
                continue
            y = _targets(closes, t, horizon)
            if y is None:
                continue
            X.append(f)
            for k in TARGETS:
                ys[k].append(y[k])
            dates.append(dts[t])
            tickers.append(tk)
    if not X:
        raise RuntimeError("No usable samples — not enough history.")
    return {"X": np.asarray(X), "dates": np.asarray(dates),
            "tickers": np.asarray(tickers),
            "y": {k: np.asarray(v, dtype=float) for k, v in ys.items()}}


# ---------------------------------------------------------------------------
# Honest out-of-sample scoring
# ---------------------------------------------------------------------------
def _ridge_fit(Xtr: np.ndarray, ytr: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    Xc, yc = Xtr - Xtr.mean(0), ytr - ytr.mean(0)
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ yc)
    return w, float(ytr.mean(0) - Xtr.mean(0) @ w) if ytr.ndim == 1 else 0.0


def _fit_predict(Xtr, ytr, Xte, alphas=(1.0, 10.0, 100.0, 1000.0, 1e4)) -> np.ndarray:
    """Ridge with alpha picked on an inner time split of TRAIN only."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
    cut = int(len(Xtr_s) * 0.75)
    best, best_err = alphas[0], np.inf
    if cut > 20 and len(Xtr_s) - cut > 10:
        for a in alphas:
            w, b = _ridge_fit(Xtr_s[:cut], ytr[:cut], a)
            err = float(np.mean((Xtr_s[cut:] @ w + b - ytr[cut:]) ** 2))
            if err < best_err:
                best, best_err = a, err
    w, b = _ridge_fit(Xtr_s, ytr, best)
    return Xte_s @ w + b


def _oos_r2(ytrue: np.ndarray, ypred: np.ndarray, train_mean: float) -> float:
    """R^2 against the TRAIN mean: 0 means no better than assuming the average."""
    ss_res = float(np.sum((ytrue - ypred) ** 2))
    ss_tot = float(np.sum((ytrue - train_mean) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _split(dates: np.ndarray, frac: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Time-ordered split with a gap, so no training window overlaps a test window."""
    uniq = np.unique(dates)
    cut = uniq[int(len(uniq) * frac)]
    gap = np.timedelta64(int(horizon * 1.6) + 5, "D")
    return dates <= cut - gap, dates > cut


def score_target(S: dict, name: str, horizon: int, frac: float) -> dict[str, float]:
    tr, te = _split(S["dates"], frac, horizon)
    y = S["y"][name]
    Xtr, Xte = S["X"][tr], S["X"][te]
    if y.ndim == 1:
        pred = _fit_predict(Xtr, Xte=Xte, ytr=y[tr])
        r2 = _oos_r2(y[te], pred, float(y[tr].mean()))
    else:  # shape: score each day of the path, then average
        r2s = [
            _oos_r2(y[te][:, j],
                    _fit_predict(Xtr, Xte=Xte, ytr=y[tr][:, j]),
                    float(y[tr][:, j].mean()))
            for j in range(y.shape[1])
        ]
        r2 = float(np.mean(r2s))
        pred = None
    out = {"r2": r2, "n_train": int(tr.sum()), "n_test": int(te.sum())}
    if name == "return" and pred is not None:
        # Direction needs TWO comparisons, and reporting only the first is how a
        # null result gets mistaken for skill:
        #   1. against the BASE RATE, not against 50%. Equities drift up, so a
        #      model that answers "up" every time already scores well above a
        #      coin flip while knowing nothing.
        #   2. against a null that respects the panel structure. An exact
        #      binomial over every window assumes each is an independent flip;
        #      they are not, because all tickers share a date and move together.
        #      That inflates significance by roughly sqrt(tickers per date).
        mask = np.abs(y[te]) > 1e-9
        yt, pr = y[te][mask], pred[mask]
        up = float(np.mean(yt > 0))
        out["direction_acc"] = float(np.mean(np.sign(pr) == np.sign(yt)))
        out["direction_base_rate"] = max(up, 1.0 - up)
        out["direction_pred_up_frac"] = float(np.mean(pr > 0))
        out["direction_n"] = int(mask.sum())
        out["direction_n_dates"] = int(len(np.unique(S["dates"][te][mask])))
    return out


def shuffle_null(S: dict, name: str, horizon: int, frac: float, n: int, seed: int) -> list[dict]:
    """Permute WHOLE DATES, keeping every ticker on a date together.

    Stocks are strongly correlated on any given day. Shuffling row by row would
    destroy that and produce a null far easier to beat than reality, which is
    exactly how a no-signal result gets mistaken for a real one.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(S["dates"])
    idx_by_date = {d: np.where(S["dates"] == d)[0] for d in uniq}
    out = []
    for i in range(n):
        perm = rng.permutation(len(uniq))
        order = np.empty(len(S["dates"]), dtype=int)
        for src, dst in enumerate(perm):
            a, b = idx_by_date[uniq[src]], idx_by_date[uniq[dst]]
            m = min(len(a), len(b))
            order[a[:m]] = b[:m]
            if len(a) > m:
                order[a[m:]] = a[m:]
        Sh = {**S, "y": {k: v[order] for k, v in S["y"].items()}}
        # Keep every metric, not just R^2, so direction is scored against the
        # same structure-preserving null in the same pass.
        out.append(score_target(Sh, name, horizon, frac))
        _progress(f"    null {i+1}/{n}")
    _progress("", done=True)
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default="2026-04-01")
    ap.add_argument("--synthetic", action="store_true",
                    help="Self-test on series with known answers instead of real prices.")
    ap.add_argument("--nulls", type=int, default=200)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--out", default=str(_SCRIPT_DIR / "predictability_results.json"))
    args = ap.parse_args()

    h = args.horizon
    print(f"\nHorizon {h} trading days | "
          + ("SYNTHETIC self-test" if args.synthetic else f"real prices {args.start} .. {args.end}"))

    if args.synthetic:
        panel = synthetic_panel()
        print(f"  {len(panel)} generated series with GARCH volatility clustering, no drift")
    else:
        panel = fetch_panel(PANEL, args.start, args.end,
                            _SCRIPT_DIR / f"price_panel_{args.start[:4]}_{args.end[:4]}.csv")

    S = build_samples(panel, h)
    n_dates = len(np.unique(S["dates"]))
    print(f"  {len(S['X'])} non-overlapping samples over {n_dates} distinct dates, "
          f"{S['X'].shape[1]} features")
    print(f"  effective independent sample size is closer to {n_dates} than to "
          f"{len(S['X'])} — stocks move together, which the null below preserves.\n")

    hl = "=" * 78
    print(hl)
    print(f" {'target':<16}{'OOS R2':>10}{'null mean':>11}{'null 95th':>11}{'p':>8}   verdict")
    print(hl)

    results, null_dir = {}, None
    for name in TARGETS:
        res = score_target(S, name, h, args.train_frac)
        nulls = shuffle_null(S, name, h, args.train_frac, args.nulls, seed=1)
        null = np.asarray([d["r2"] for d in nulls])
        p = float((null >= res["r2"]).mean())
        res.update({"null_mean": float(null.mean()),
                    "null_p95": float(np.percentile(null, 95)), "p_value": p})
        res["verdict"] = classify(res["r2"], p)
        if name == "return":
            null_dir = np.asarray([d["direction_acc"] for d in nulls if "direction_acc" in d])
        results[name] = res
        print(f" {name:<16}{res['r2']:>10.4f}{null.mean():>11.4f}"
              f"{np.percentile(null, 95):>11.4f}{p:>8.3f}   {res['verdict']}")

    print(hl)
    d = results["return"]
    if "direction_acc" in d:
        p_dir = float((null_dir >= d["direction_acc"]).mean()) if null_dir is not None and len(null_dir) else float("nan")
        d["direction_null_mean"] = float(null_dir.mean()) if null_dir is not None and len(null_dir) else None
        d["direction_p"] = p_dir
        beats_base = d["direction_acc"] > d["direction_base_rate"]
        print(f"\n  DIRECTION — {d['direction_n']} test windows spread over only "
              f"{d['direction_n_dates']} distinct dates")
        print(f"    model accuracy                  {d['direction_acc']*100:5.1f}%")
        print(f"    always guess the commoner sign  {d['direction_base_rate']*100:5.1f}%"
              "   <- free, needs no model")
        if d["direction_null_mean"] is not None:
            print(f"    shuffled-date null              {d['direction_null_mean']*100:5.1f}%"
                  f"   (p = {p_dir:.3f})")
        print(f"    model said 'up' on              {d['direction_pred_up_frac']*100:5.1f}%"
              " of windows")
        print("    => " + ("beats the free baseline" if beats_base else
                           "does NOT beat the free baseline — worse than a constant"))

    signal = {k: v for k, v in results.items() if v["verdict"] == "SIGNAL"}
    print(f"\n{hl}\n VERDICT\n{hl}")
    print(f"  \"SIGNAL\" means p < {_ALPHA} AND out-of-sample R2 >= {_USABLE_R2:.0%}. Anything")
    print("  significant but below that bar is reported as too small to use.\n")

    if args.synthetic:
        vol_ok = results["log_volatility"]["verdict"] == "SIGNAL"
        ret_ok = results["return"]["verdict"] != "SIGNAL"
        print("  Self-test. Correct answer, known by construction: volatility predictable,")
        print("  return not.\n")
        print(f"    volatility found usable            : {vol_ok}   (want True)")
        print(f"    return NOT claimed as usable signal: {ret_ok}   (want True)"
              + (f"  [{results['return']['verdict']}]" if not ret_ok else ""))
        print("\n  " + ("PASS — the method finds signal that exists and not signal that doesn't."
                        if (vol_ok and ret_ok) else
                        "FAIL — do not trust this script's verdict on real data."))
    elif not signal:
        print("  Nothing is predictable here, including volatility. That is a surprising")
        print("  result and more likely means something is wrong with this test than with")
        print("  markets — run --synthetic before acting on it.")
    elif "log_volatility" in signal and "return" not in signal and "shape" not in signal:
        print("  Only VOLATILITY is predictable. Neither the direction/size of the move nor")
        print("  the shape of the path carries signal a linear model can find.\n")
        print("  => The predictable part of a price path is its WIDTH, not its trajectory.")
        print("     Re-rendering targets on a fixed return scale would make magnitude")
        print("     representable, but there is little magnitude signal to represent, so")
        print("     that rebuild would not pay for itself either.")
        print("     The volatility cone already in the panel — 69% measured coverage against")
        print("     a 68% target — is the part that works, and it needs no model.")
    elif "return" in signal or "shape" in signal:
        print("  There IS recoverable signal in the path itself, which the current pipeline")
        print("  is failing to use. That makes the rebuild worth doing — and it should be")
        print("  aimed at whichever target scored above, not at the auto-scaled shape.")
    else:
        print("  Mixed result — read the table above rather than a one-line verdict.")

    print(f"\n{hl}")
    payload = {"horizon_days": h, "synthetic": args.synthetic,
               "n_samples": int(len(S["X"])), "n_dates": int(n_dates),
               "train_frac": args.train_frac, "nulls": args.nulls, "results": results}
    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n  Written to {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
