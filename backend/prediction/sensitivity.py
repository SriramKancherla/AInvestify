"""
Does the forecast model actually read its input?

The backtest showed the model has no measurable skill: its predictions vary
8-16x less than reality and correlate ~0 with outcomes. That is consistent with
two very different diagnoses, and they need different fixes:

  (a) The model DOES respond to its input, but the training targets were
      y-normalised per window so magnitude was destroyed. Fix = re-render
      targets on a fixed, invertible return scale and retrain.

  (b) The model largely IGNORES its input and emits an unconditional
      chart-shaped prior. Fix = the conditioning path itself. Re-rendering
      targets would not help, and a multi-day retrain would be aimed at the
      wrong thing.

This script separates them.

THE EXPERIMENT
Run the model on deliberately different conditioning inputs while holding the
random seed FIXED, so any change in output is attributable to the input alone.
Separately, hold the input fixed and vary only the seed. That second number is
the noise floor: how much the output moves for reasons that have nothing to do
with the data.

Then compare:

    signal / noise  =  (change caused by rewriting the input)
                       ----------------------------------------
                       (change caused by reshuffling the noise)

  >> 1  the input matters; diagnosis (a); the retrain plan is aimed correctly
  ~= 1  the input is being ignored; diagnosis (b); fix conditioning first

Both are measured on the ENSEMBLE MEAN PATH -- the thing the product actually
shows -- not on a single raw sample, so the number answers the question the
product cares about.

THE INPUT VARIANTS
  real          the true price window (baseline)
  inverted      prices mirrored about the window mean: every up move becomes a
                down move. The most direct opposite of the real input.
  reversed      the same prices in reverse time order
  flat          a constant price: no information at all
  other         a completely different company's window

If "inverted" and "flat" produce the same forecast as "real", the model is not
looking at the chart.

Usage:
    python sensitivity.py --ticker AAPL --as-of 2026-03-31 --horizon 21
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import forecast_runner as fr  # noqa: E402
from backtest import fetch_history, preflight  # noqa: E402

VARIANTS = ("real", "inverted", "reversed", "flat", "other")
CONTRAST_TICKER = "XOM"  # deliberately unlike a mega-cap tech name


# ---------------------------------------------------------------------------
def make_variant(window: pd.DataFrame, name: str, other: pd.DataFrame | None) -> pd.DataFrame:
    """Rewrite the input window. Every variant keeps the same dates, the same
    number of bars and a comparable price level — only the SHAPE changes, so a
    difference in output can't be blamed on the axis scale moving."""
    w = window.copy().reset_index(drop=True)
    cols = ["Open", "High", "Low", "Close"]

    if name == "real":
        return w

    if name == "inverted":
        # Reflect about the window's mean close. Highs and lows swap roles,
        # because the top of a bar becomes the bottom of the reflected bar.
        mid = float(w["Close"].mean())
        out = w.copy()
        out["Open"] = 2 * mid - w["Open"]
        out["Close"] = 2 * mid - w["Close"]
        out["High"] = 2 * mid - w["Low"]
        out["Low"] = 2 * mid - w["High"]
        return out

    if name == "reversed":
        out = w.copy()
        out[cols] = w[cols].iloc[::-1].to_numpy()
        return out

    if name == "flat":
        out = w.copy()
        last = float(w["Close"].iloc[-1])
        for c in cols:
            out[c] = last
        # A dead-flat series has zero range; give it a hairline so the renderer
        # has something to draw rather than a degenerate axis.
        out["High"] = last * 1.0005
        out["Low"] = last * 0.9995
        return out

    if name == "other":
        if other is None or len(other) < len(w):
            raise ValueError("contrast ticker has too little history")
        src = other.iloc[-len(w):].reset_index(drop=True)
        out = w.copy()
        # Rescale the contrast series to this stock's price level so the
        # difference is shape, not magnitude.
        scale = float(w["Close"].iloc[-1]) / float(src["Close"].iloc[-1])
        for c in cols:
            out[c] = src[c].to_numpy() * scale
        return out

    raise ValueError(f"unknown variant {name}")


def render_X(window: pd.DataFrame) -> np.ndarray:
    from render_channels import (
        render_fundamentals_channel,
        render_news_channel,
        render_price_chart,
        stack_channels,
    )

    config = fr._import_config()
    price_ch = render_price_chart(window[["Date", "Open", "High", "Low", "Close"]])
    fund_ch = render_fundamentals_channel({f: 0.0 for f in config.FUNDAMENTALS_FIELDS})
    news_ch = render_news_channel([], [], [d.date().isoformat() for d in window["Date"]])
    X = stack_channels(price_ch, fund_ch, news_ch)
    return (np.clip(X, 0.0, 1.0) * 255.0).astype(np.uint8)


def mean_path(X_u8: np.ndarray, seed: int, horizon: int, img_size: int) -> np.ndarray:
    """Ensemble-mean normalised path for one input at one seed."""
    # Use the runner's own torch handle rather than importing torch again, so
    # this test runs against exactly the stack production runs against.
    torch_mod, ConditionalUNet, GaussianDiffusion = fr._load_torch_stack()
    bundle = fr._get_model_bundle(torch_mod, ConditionalUNet, GaussianDiffusion)
    model, diffusion, device = bundle["model"], bundle["diffusion"], bundle["device"]

    k = max(1, int(fr.ENSEMBLE_SAMPLES))
    X_one = (
        torch_mod.from_numpy(X_u8.astype(np.float32) / 255.0) * 2.0 - 1.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)
    X_tensor = X_one.repeat(k, 1, 1, 1)
    hz_idx = fr.HORIZON_OPTIONS.index(int(horizon))
    hz_tensor = torch_mod.full((k,), hz_idx, device=device, dtype=torch_mod.long)

    torch_mod.manual_seed(seed)  # the whole experiment hinges on this being explicit
    with torch_mod.no_grad():
        Y = fr._sample_ddim(
            diffusion, model, X_tensor, hz_tensor,
            img_size=img_size, num_steps=fr.LIVE_INFERENCE_STEPS,
        )
    return np.asarray(fr._ensemble_paths(Y, int(horizon))["mean_path"], dtype=float)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--as-of", default="2026-03-31")
    ap.add_argument("--horizon", type=int, default=21, choices=fr.HORIZON_OPTIONS)
    ap.add_argument("--seeds", type=int, default=4, help="How many seeds to use for the noise floor.")
    ap.add_argument("--out", default=str(_SCRIPT_DIR / "sensitivity_results.json"))
    args = ap.parse_args()

    if fr._pick_checkpoint() is None:
        print("No checkpoint under checkpoints_stable/. Nothing to test.")
        return 1
    preflight(__file__)

    as_of = pd.Timestamp(args.as_of)
    config = fr._import_config()
    img_size = int(config.IMG_HEIGHT)

    print(f"\nFetching {args.ticker} and contrast ticker {CONTRAST_TICKER} ...")
    hist = fetch_history(args.ticker, (as_of - timedelta(days=2000)).date().isoformat(),
                         (as_of + timedelta(days=1)).date().isoformat())
    other_hist = fetch_history(CONTRAST_TICKER, (as_of - timedelta(days=2000)).date().isoformat(),
                               (as_of + timedelta(days=1)).date().isoformat())

    past = hist[hist["Date"] <= as_of]
    if len(past) < fr.INPUT_WINDOW_DAYS:
        print(f"Only {len(past)} trading days on or before {args.as_of}; need {fr.INPUT_WINDOW_DAYS}.")
        return 1
    window = past.iloc[-fr.INPUT_WINDOW_DAYS:].copy()
    other_window = other_hist[other_hist["Date"] <= as_of].iloc[-fr.INPUT_WINDOW_DAYS:].copy()

    # ---- build every input, and confirm they really are different images ----
    Xs: dict[str, np.ndarray] = {}
    for v in VARIANTS:
        Xs[v] = render_X(make_variant(window, v, other_window))
    print("\nConditioning images (price channel), difference from 'real':")
    base_px = Xs["real"][:, :, 0].astype(float)
    for v in VARIANTS:
        d = float(np.abs(Xs[v][:, :, 0].astype(float) - base_px).mean())
        print(f"  {v:<10} mean abs pixel difference {d:7.2f} / 255"
              + ("   <-- baseline" if v == "real" else ""))
    if min(float(np.abs(Xs[v][:, :, 0].astype(float) - base_px).mean())
           for v in VARIANTS if v != "real") < 1.0:
        print("\n  WARNING: a variant is nearly identical to the real input. The test would be")
        print("  vacuous — fix the transform before reading anything into the result.\n")

    total = len(VARIANTS) + max(0, args.seeds - 1) + 1  # +1 for the determinism check
    print(f"\nRunning {total} ensembles of {fr.ENSEMBLE_SAMPLES} samples "
          f"({fr.LIVE_INFERENCE_STEPS} DDIM steps each) ...")

    # ---- signal: vary the input, hold the seed ----
    paths: dict[str, np.ndarray] = {}
    for i, v in enumerate(VARIANTS, 1):
        print(f"  [{i}/{total}] input = {v:<10} (seed 0)", flush=True)
        paths[v] = mean_path(Xs[v], seed=0, horizon=args.horizon, img_size=img_size)

    # ---- self-check: same input, same seed must reproduce exactly ----
    # eta=0 makes DDIM deterministic given its initial noise, that noise is drawn
    # from the global RNG, and the model is in eval mode — so this should be zero
    # to floating-point. If it isn't, some other randomness is leaking in and the
    # noise floor below is not measuring what it claims to.
    print(f"  [{len(VARIANTS)+1}/{total}] input = real       (seed 0, repeat — determinism check)", flush=True)
    repeat_diff = rms(mean_path(Xs["real"], seed=0, horizon=args.horizon, img_size=img_size), paths["real"])
    if repeat_diff > 1e-9:
        print(f"\n  WARNING: the same input at the same seed gave a different answer "
              f"(difference {repeat_diff:.2e}).")
        print("  Something outside the seed is random. Treat the noise floor below as a")
        print("  lower bound and the ratios as optimistic.\n")

    # ---- noise floor: hold the input, vary the seed ----
    seed_paths = [paths["real"]]
    for j, s in enumerate(range(1, args.seeds), 1):
        print(f"  [{len(VARIANTS)+1+j}/{total}] input = real       (seed {s})", flush=True)
        seed_paths.append(mean_path(Xs["real"], seed=s, horizon=args.horizon, img_size=img_size))

    noise = [rms(seed_paths[a], seed_paths[b])
             for a in range(len(seed_paths)) for b in range(a + 1, len(seed_paths))]
    noise_floor = float(np.mean(noise)) if noise else float("nan")

    # How much the displayed path moves at all, top to bottom. Both the signal and
    # the noise are meaningless without this: 0.001 vs 0.0005 is "2x" but neither
    # is visible on a chart whose whole path spans 0.4.
    path_scale = float(np.ptp(paths["real"]))
    DEGENERATE = 1e-6

    hl = "=" * 76
    print(f"\n{hl}\n RESULT — {args.ticker}, as of {args.as_of}, +{args.horizon}d\n{hl}")
    print(f"\n  Scale reference: the real forecast path spans {path_scale:.4f} "
          f"top to bottom.")
    print(f"  Same input, same seed, run twice: {repeat_diff:.2e}  (0 = fully deterministic)")
    print(f"\n  Noise floor (same input, {args.seeds} different seeds):")
    print(f"    mean pairwise path difference = {noise_floor:.4f}"
          + (f"  ({noise_floor / path_scale * 100:.1f}% of the path's own span)" if path_scale > 0 else ""))
    print(f"    range = {min(noise):.4f} .. {max(noise):.4f}\n")
    print(f"  {'input variant':<12} {'path diff':>10} {'vs noise':>10} {'endpoint move':>15}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*15}")
    ratios, diffs = {}, {}
    for v in VARIANTS:
        if v == "real":
            continue
        d = rms(paths[v], paths["real"])
        diffs[v] = d
        ratios[v] = d / noise_floor if noise_floor > DEGENERATE else float("nan")
        end = paths[v][-1] - paths["real"][-1]
        shown = f"{ratios[v]:>9.2f}x" if np.isfinite(ratios[v]) else f"{'n/a':>10}"
        print(f"  {v:<12} {d:>10.4f} {shown} {end:>+15.4f}")

    print(f"\n{hl}\n VERDICT\n{hl}")

    # Degenerate cases first — a ratio is only meaningful if its denominator is.
    if noise_floor <= DEGENERATE and max(diffs.values()) <= DEGENERATE:
        print("  The model emits the SAME path no matter what. Neither rewriting the input")
        print("  nor changing the noise moved it at all. It is not a forecaster; it is a")
        print("  constant with a chart drawn around it.")
        print("\n  => Nothing about the target scale is the problem. Do not retrain against")
        print("     this architecture until the conditioning path is fixed and verified.")
        mean_ratio = 0.0
    elif noise_floor <= DEGENERATE:
        print("  The seed changes nothing but the input changes the output — sampling has")
        print("  collapsed to a single mode, yet conditioning is alive. The ratio is")
        print("  effectively infinite (zero denominator).")
        print("\n  => Conditioning works. The no-skill result is about magnitude, so the")
        print("     fixed-return-scale rebuild is aimed correctly. Note separately that the")
        print("     ensemble spread you show as uncertainty is not real spread.")
        mean_ratio = float("inf")
    else:
        worst = max(ratios.values())
        mean_ratio = float(np.mean(list(ratios.values())))
        print(f"  Rewriting the input moves the forecast {mean_ratio:.2f}x as much as")
        print(f"  reshuffling the random seed does (best case {worst:.2f}x).\n")
        _print_graded_verdict(mean_ratio)

    payload_extra = {"path_scale": path_scale, "repeat_diff": repeat_diff}
    return _finish(args, paths, ratios, noise_floor, mean_ratio, payload_extra)


def _print_graded_verdict(mean_ratio: float) -> None:
    if mean_ratio < 1.5:
        print("  The model is NOT meaningfully reading its input. Inverting the price")
        print("  history — turning every rally into a selloff — barely changes the")
        print("  forecast more than re-rolling the dice does.")
        print("\n  => Re-rendering targets on a fixed return scale will NOT fix this on its")
        print("     own. The conditioning path has to work first. Do not start a multi-day")
        print("     retrain against the current architecture.")
    elif mean_ratio < 3.0:
        print("  The input has a weak but real effect — larger than pure noise, much")
        print("  smaller than you would expect if the model were genuinely reading the")
        print("  chart. Consistent with conditioning that is present but underpowered")
        print("  (25 of 100 epochs, base_channels tuned down to 24 by a 3-epoch search).")
        print("\n  => Worth fixing the target scale AND training properly. Expect the two")
        print("     to be needed together.")
    else:
        print("  The model clearly responds to its input. The conditioning works.")
        print("\n  => The no-skill result is then about the destroyed magnitude scale, and")
        print("     the fixed-return-scale rebuild is aimed at the right problem.")


def _finish(args, paths, ratios, noise_floor, mean_ratio, extra: dict) -> int:
    payload = {
        "ticker": args.ticker, "as_of": args.as_of, "horizon_days": args.horizon,
        "ensemble_samples": fr.ENSEMBLE_SAMPLES, "seeds": args.seeds,
        "noise_floor": noise_floor,
        "variants": {v: {"path_diff": rms(paths[v], paths["real"]),
                         "ratio_vs_noise": (None if v == "real" or not np.isfinite(ratios.get(v, np.nan))
                                            else ratios[v]),
                         "path": [round(float(x), 6) for x in paths[v]]}
                     for v in VARIANTS},
        "mean_ratio": None if not np.isfinite(mean_ratio) else mean_ratio,
        **extra,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n  Written to {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
