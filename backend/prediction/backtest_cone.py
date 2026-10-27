"""
Does the outcome cone actually cover reality?

A confidence band is only worth drawing if its stated confidence is true. This
scores the cone from `volatility.py` against 18 years of real outcomes and
reports two numbers per level, because either alone can be gamed:

  coverage   how often the real price landed inside the band. An 80% band should
             capture 80% — no more, no less. Over-covering is not "safe", it
             means the band is too wide to say anything.
  width      the band's half-width as a percent of price. A band covering 95% by
             spanning +/-60% is technically honest and practically useless.

BASELINE
The same windows scored with what the panel shipped before: a trailing 252-day
standard deviation extrapolated as sigma*sqrt(t), with normal quantiles. That
assumes today's volatility persists unchanged for a year and that returns are
Gaussian. Both are false, and the comparison shows by how much.

Centre is FLAT (today's price) for both, because `predictability.py` measured the
out-of-sample R2 of a 21-day return forecast at -0.030 — worse than assuming no
change. There is nothing better to centre on.

WHY THERE IS A --tune MODE
The first version of this script picked its own parameters and then reported
calibration on the same windows, which is how a tuned number gets mistaken for a
measured one. `--tune` splits by DATE: parameters are chosen on 2008-2017 and
scored on 2018-2026, which the search never sees. The shipped constants in
volatility.py come from that split.

Usage:
    python backtest_cone.py                 # score shipped params, all horizons
    python backtest_cone.py --horizon 21
    python backtest_cone.py --tune          # choose params on early data, score on late
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from predictability import PANEL, fetch_panel  # noqa: E402
from volatility import (  # noqa: E402
    GARCH_PERSISTENCE,
    LONG_RUN_DAYS,
    _effective_df,
    _normal_quantile,
    _t_quantile,
    horizon_sigma,
)

LEVELS = (0.50, 0.80, 0.95)
HORIZONS = (5, 21, 63, 126, 252)
_MIN_HISTORY = 400
_SPLIT_DATE = "2018-01-01"      # tune before, validate after

_GRID_PERSISTENCE = (0.95, 0.97, 0.985, 0.99)
_GRID_LONG_RUN = (504, 756, 1260)      # 2, 3, 5 years


def score(panel: dict, horizon: int, persistence: float, long_run_days: int,
          date_from=None, date_to=None, levels=LEVELS) -> dict:
    """Walk every ticker on non-overlapping windows; score cone vs baseline."""
    hit_new = {lv: 0 for lv in levels}
    hit_old = {lv: 0 for lv in levels}
    w_new = {lv: [] for lv in levels}
    w_old = {lv: [] for lv in levels}
    n = 0

    k_new = {lv: _t_quantile(0.5 + lv / 2, _effective_df(horizon)) for lv in levels}
    k_old = {lv: _normal_quantile(0.5 + lv / 2) for lv in levels}
    lo_b = pd.Timestamp(date_from) if date_from else None
    hi_b = pd.Timestamp(date_to) if date_to else None

    for df in panel.values():
        closes = df["Close"].to_numpy(dtype=float)
        dates = pd.to_datetime(df["Date"]).to_numpy()
        if len(closes) < _MIN_HISTORY + horizon:
            continue
        logp = np.log(closes)
        for t in range(_MIN_HISTORY, len(closes) - horizon, horizon):
            d = pd.Timestamp(dates[t])
            if (lo_b is not None and d < lo_b) or (hi_b is not None and d >= hi_b):
                continue
            rets = np.diff(logp[: t + 1])
            if rets.size < 260 or not np.isfinite(rets[-260:]).all():
                continue
            last, actual = closes[t], closes[t + horizon]
            if last <= 0 or actual <= 0:
                continue

            # Cone: mean-reverting variance term structure, fat-tailed quantiles,
            # applied in log space so the band can never cross zero.
            sigma = float(horizon_sigma(rets, horizon, persistence, long_run_days)[-1])
            # Baseline: trailing std extrapolated as sqrt(t), normal, arithmetic.
            sigma_old = float(np.std(rets[-252:], ddof=1)) * np.sqrt(horizon)

            for lv in levels:
                lo = last * np.exp(-k_new[lv] * sigma)
                hi = last * np.exp(+k_new[lv] * sigma)
                hit_new[lv] += lo <= actual <= hi
                w_new[lv].append((hi - lo) / 2.0 / last)

                lo_o = max(last * (1 - k_old[lv] * sigma_old), 1e-9)
                hi_o = last * (1 + k_old[lv] * sigma_old)
                hit_old[lv] += lo_o <= actual <= hi_o
                w_old[lv].append((hi_o - lo_o) / 2.0 / last)
            n += 1

    if n == 0:
        raise RuntimeError(f"no usable windows at horizon {horizon}")
    return {
        "n": n,
        "new": {lv: {"coverage": hit_new[lv] / n,
                     "median_halfwidth_pct": float(np.median(w_new[lv]) * 100)} for lv in levels},
        "old": {lv: {"coverage": hit_old[lv] / n,
                     "median_halfwidth_pct": float(np.median(w_old[lv]) * 100)} for lv in levels},
    }


def mean_abs_error(panel, persistence, long_run_days, date_from=None, date_to=None,
                   horizons=HORIZONS, which="new") -> float:
    """Average |coverage - level| over every horizon and level."""
    errs = []
    for h in horizons:
        r = score(panel, h, persistence, long_run_days, date_from, date_to)
        errs += [abs(r[which][lv]["coverage"] - lv) for lv in LEVELS]
    return float(np.mean(errs))


def _report(panel, horizons, persistence, long_run_days, date_from, date_to, title) -> dict:
    hl = "=" * 78
    print(f"\n{hl}\n {title}\n{hl}")
    print("\n  'cone' = mean-reverting variance + fat tails, lognormal band")
    print("  'old'  = trailing 252d std * sqrt(t), normal, arithmetic (what shipped)")
    print("  Coverage should MATCH the level. Over-covering means too wide to be useful.\n")

    out, e_new, e_old, wins = {}, [], [], 0
    for h in horizons:
        r = score(panel, h, persistence, long_run_days, date_from, date_to)
        out[h] = r
        print(f"  horizon {h:>3}d   ({r['n']} non-overlapping windows)")
        print(f"    {'level':<7}{'cone cov':>10}{'cone ±%':>9}   {'old cov':>9}{'old ±%':>9}"
              f"   {'cone err':>9}{'old err':>9}   better")
        for lv in LEVELS:
            a, b = r["new"][lv], r["old"][lv]
            ea, eb = abs(a["coverage"] - lv), abs(b["coverage"] - lv)
            e_new.append(ea)
            e_old.append(eb)
            wins += ea < eb
            print(f"    {int(lv*100):>3}%  {a['coverage']*100:>9.1f}%{a['median_halfwidth_pct']:>9.1f}"
                  f"   {b['coverage']*100:>8.1f}%{b['median_halfwidth_pct']:>9.1f}"
                  f"   {ea*100:>8.1f}{eb*100:>8.1f}   {'cone' if ea < eb else 'old'}")
        print()

    total = len(e_new)
    print(hl)
    print(f"  mean |coverage error|   cone {np.mean(e_new)*100:.1f}pp   "
          f"old {np.mean(e_old)*100:.1f}pp")
    print(f"  worst                   cone {max(e_new)*100:.1f}pp   old {max(e_old)*100:.1f}pp")
    print(f"  cone better on {wins} of {total} horizon×level cells")
    print(hl)
    if wins == total:
        print("  The cone is better calibrated everywhere it was scored.")
    elif np.mean(e_new) < np.mean(e_old):
        print(f"  The cone is better ON AVERAGE but loses in {total - wins} cells — see the")
        print("  'better' column. It is not a uniform win and should not be sold as one.")
    else:
        print("  The cone is NOT better on average. Do not ship it over the baseline.")
    return {"results": out, "mean_err_new": float(np.mean(e_new)),
            "mean_err_old": float(np.mean(e_old)), "wins": wins, "cells": total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--tune", action="store_true",
                    help="Choose parameters on 2008-2017, then score on 2018-2026.")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default="2026-04-01")
    ap.add_argument("--out", default=str(_SCRIPT_DIR / "cone_calibration.json"))
    args = ap.parse_args()

    panel = fetch_panel(PANEL, args.start, args.end,
                        _SCRIPT_DIR / f"price_panel_{args.start[:4]}_{args.end[:4]}.csv")
    horizons = (args.horizon,) if args.horizon else HORIZONS
    phi, lrd = GARCH_PERSISTENCE, LONG_RUN_DAYS
    tuning = None

    if args.tune:
        print(f"\nTuning on windows before {_SPLIT_DATE}, validating on windows after.")
        print("The search never sees the validation period.\n")
        rows = []
        for p in _GRID_PERSISTENCE:
            for L in _GRID_LONG_RUN:
                e = mean_abs_error(panel, p, L, date_to=_SPLIT_DATE, horizons=horizons)
                rows.append((e, p, L))
                print(f"  persistence {p:<6} long-run {L//252}y   "
                      f"train mean |cov err| {e*100:5.2f}pp")
        rows.sort()
        _, phi, lrd = rows[0]
        base = mean_abs_error(panel, phi, lrd, date_to=_SPLIT_DATE,
                              horizons=horizons, which="old")
        print(f"\n  best on train: persistence {phi}, long-run {lrd//252}y "
              f"({rows[0][0]*100:.2f}pp)   baseline on same windows: {base*100:.2f}pp")
        print(f"  shipped constants are {GARCH_PERSISTENCE} / {LONG_RUN_DAYS//252}y")
        tuning = {"chosen_persistence": phi, "chosen_long_run_days": lrd,
                  "train_grid": [{"err": e, "persistence": p, "long_run_days": L}
                                 for e, p, L in rows]}
        title = f"VALIDATION — windows from {_SPLIT_DATE}, params chosen on earlier data"
        rep = _report(panel, horizons, phi, lrd, _SPLIT_DATE, None, title)
    else:
        title = (f"CONE CALIBRATION — {len(panel)} tickers, {args.start} to {args.end}, "
                 f"persistence {phi}, long-run {lrd//252}y")
        rep = _report(panel, horizons, phi, lrd, None, None, title)

    payload = {"tickers": len(panel), "levels": list(LEVELS), "persistence": phi,
               "long_run_days": lrd, "tuned": bool(args.tune), "tuning": tuning,
               "mean_err_cone": rep["mean_err_new"], "mean_err_old": rep["mean_err_old"],
               "cells_won": rep["wins"], "cells": rep["cells"],
               "results": {str(k): {m: ({str(lv): vv for lv, vv in v.items()}
                                        if isinstance(v, dict) else v)
                                    for m, v in r.items()}
                           for k, r in rep["results"].items()}}
    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n  Written to {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
