"""
Volatility forecasting for the outcome cone.

WHY THIS EXISTS
`predictability.py` measured what is actually forecastable in a price path, over
30 tickers and 18 years of out-of-sample windows:

    return          OOS R2  -0.030   (worse than assuming the average)
    shape           OOS R2  -0.003   (what the diffusion model was trained on)
    log_volatility  OOS R2  +0.453   (p = 0.000)

Direction did not beat "always guess the commoner sign". So the honest product is
a distribution of outcomes whose WIDTH is forecast, not a path whose direction is
guessed. This module forecasts that width.

WHAT IT IMPROVES ON
The panel previously scaled its band by a trailing standard deviation times
sqrt(t). That has one real flaw: it assumes today's volatility persists forever.
Volatility is persistent but MEAN-REVERTING, so a 1-year band built from a
turbulent month is far too wide, and one built from a calm month too narrow.

The fix is a variance term structure. Under a GARCH(1,1)-style recursion with
persistence phi = alpha + beta, the expected variance k days ahead is

    sigma2_k = LR + phi^k * (sigma2_now - LR)

and what a cone needs is the AVERAGE variance over the whole horizon, which sums
that geometric series in closed form:

    mean sigma2 over h days = LR + (sigma2_now - LR) * (1 - phi^h) / (h * (1 - phi))

At h=1 this is today's variance; as h grows it decays toward the long-run level
LR. phi is not fitted per stock — a fitted persistence on 250 noisy observations
is less reliable than the well-established equity value near 0.97, and an
unstable phi would make the cone jump around for no good reason.

FAT TAILS
Daily equity returns are not Gaussian; they have far more 3-sigma days than a
normal distribution allows. Using normal quantiles would make the outer bands too
narrow exactly when they matter. Quantiles here come from a Student-t with
STUDENT_T_DF degrees of freedom, rescaled to unit variance so the t only changes
tail SHAPE, never the forecast's overall width. Aggregation over many days pulls
the sum toward normality, so the effective degrees of freedom are relaxed with
horizon rather than held fixed.

Everything here is measured, not asserted — `backtest_cone.py` scores the
resulting coverage against 18 years of real outcomes.

Self-test:
    python volatility.py
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Daily variance persistence (alpha + beta of a GARCH(1,1) on equities). Near 1
# means slow mean reversion: shocks fade over months, not days. 0.99 gives a
# half-life of about 69 trading days, and sits in the middle of the 0.98-0.995
# range typically fitted to daily equity data — so it is a defensible prior, not
# a number bent to fit this panel.
GARCH_PERSISTENCE = 0.99
# Weight on the recent EWMA estimate when forming "today's" variance. The rest
# goes to a long sample estimate, which stops a single spike from dominating.
EWMA_LAMBDA = 0.94
# Degrees of freedom for the 1-day return distribution. ~4 is the usual estimate
# for daily equity returns; lower means fatter tails.
STUDENT_T_DF = 4.0
# Sensible bounds on annualised vol, to keep a data glitch from producing an
# absurd cone. 4% is quieter than any real equity; 250% is beyond 2008 or 2020.
_MIN_ANNUAL_VOL, _MAX_ANNUAL_VOL = 0.04, 2.50
_TRADING_DAYS = 252
# Sample window for the long-run variance the forecast reverts TOWARD.
# Both this and GARCH_PERSISTENCE were chosen on 2008-2017 and then scored on
# 2018-2026 by backtest_cone.py --tune, so they are not fitted to the numbers
# they are reported against.
LONG_RUN_DAYS = 2 * _TRADING_DAYS


def ewma_variance(returns: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """Exponentially weighted daily variance — the RiskMetrics estimator.

    Recent days count more, with a half-life of about 11 days at lam=0.94, so the
    estimate reacts to a regime change instead of averaging it away.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    w = lam ** np.arange(r.size - 1, -1, -1)     # oldest gets the smallest weight
    w /= w.sum()
    return float(np.sum(w * (r - r.mean()) ** 2))


def forecast_variance_path(
    returns: np.ndarray,
    horizon_days: int,
    persistence: float = GARCH_PERSISTENCE,
    long_run_days: int = None,
) -> np.ndarray:
    """Expected daily variance for each of the next `horizon_days` days.

    Blends a fast EWMA estimate with a long-run sample estimate, then mean-reverts
    the blend toward the long-run level at rate `persistence`.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 30:
        raise ValueError(f"need at least 30 returns to estimate volatility, got {r.size}")

    h = max(1, int(horizon_days))
    lr_days = int(long_run_days or LONG_RUN_DAYS)
    long_run = float(np.var(r[-min(r.size, lr_days):], ddof=1))
    fast = ewma_variance(r)
    if not np.isfinite(fast) or fast <= 0:
        fast = long_run
    # "Today's" variance leans on the fast estimate but keeps a foot in the long
    # sample, so one spike can't set the whole cone.
    now = 0.75 * fast + 0.25 * long_run

    lo = (_MIN_ANNUAL_VOL / np.sqrt(_TRADING_DAYS)) ** 2
    hi = (_MAX_ANNUAL_VOL / np.sqrt(_TRADING_DAYS)) ** 2
    now, long_run = float(np.clip(now, lo, hi)), float(np.clip(long_run, lo, hi))

    k = np.arange(1, h + 1, dtype=float)
    return long_run + (persistence**k) * (now - long_run)


def horizon_sigma(
    returns: np.ndarray,
    horizon_days: int,
    persistence: float = GARCH_PERSISTENCE,
    long_run_days: int = None,
) -> np.ndarray:
    """Cumulative return standard deviation at each day 1..h.

    Variance of a sum of independent daily returns is the sum of their variances,
    so this is the running square root of the forecast variance path — not
    sigma*sqrt(t), which would assume today's volatility never changes.
    """
    return np.sqrt(np.cumsum(
        forecast_variance_path(returns, horizon_days, persistence, long_run_days)))


@lru_cache(maxsize=4096)
def _t_quantile(p: float, df: float) -> float:
    """Student-t quantile, scaled to unit variance, with no scipy dependency.

    Memoised because it is pure and expensive: each call runs a 2,000-point
    Simpson integration inside a Newton loop, and a 1-year cone asks for one
    quantile per trading day. The cache turns 252 integrations into 252 on the
    first request and zero on every one after, across all tickers and horizons.

    Newton's method on the t CDF built from the regularised incomplete beta. The
    scaling by sqrt((df-2)/df) is what makes this a pure tail-shape change: the
    result has variance 1, so swapping normal for t never widens the cone
    overall, it only moves mass from the shoulders into the tails.
    """
    from math import exp, lgamma, pi, sqrt

    if df <= 2:
        raise ValueError("df must exceed 2 for the distribution to have finite variance")
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    if abs(p - 0.5) < 1e-12:
        return 0.0

    def pdf(x: float) -> float:
        c = exp(lgamma((df + 1) / 2) - lgamma(df / 2)) / sqrt(df * pi)
        return c * (1 + x * x / df) ** (-(df + 1) / 2)

    def cdf(x: float) -> float:
        # P(T <= x) via the incomplete beta identity, integrated numerically.
        # Simpson on a fixed grid: smooth integrand, and accuracy well past what
        # a chart band needs.
        n = 2000
        a, b = 0.0, abs(x)
        hstep = (b - a) / n
        s = pdf(a) + pdf(b)
        for i in range(1, n):
            s += pdf(a + i * hstep) * (4 if i % 2 else 2)
        half = s * hstep / 3
        return 0.5 + half if x > 0 else 0.5 - half

    # Newton from the normal quantile as a starting point.
    x = _normal_quantile(p)
    for _ in range(60):
        d = pdf(x)
        if d < 1e-14:
            break
        step = (cdf(x) - p) / d
        x -= step
        if abs(step) < 1e-10:
            break
    return x * sqrt((df - 2) / df)


def _normal_quantile(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9 accurate)."""
    from math import log, sqrt

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _effective_df(days: int, base_df: float = STUDENT_T_DF) -> float:
    """Degrees of freedom for a sum of `days` returns.

    A single day is sharply fat-tailed; a sum of many days is pulled toward
    normal by the central limit theorem, so holding df fixed would leave the
    1-year band as fat-tailed as the 1-day band, which is wrong. Growing df
    linearly in the number of days is the standard cheap approximation, capped
    where t and normal are indistinguishable anyway.
    """
    return float(min(base_df + 0.5 * max(0, days - 1), 120.0))


def band_multipliers(levels: tuple[float, ...], days: np.ndarray) -> dict[float, np.ndarray]:
    """Quantile multiplier per confidence level, per day.

    Returns {level: array of k}, so the band at that day is centre +/- k*sigma.
    Two-sided: level 0.80 uses the 0.90 quantile.
    """
    out: dict[float, np.ndarray] = {}
    for lv in levels:
        q = 0.5 + lv / 2.0
        out[lv] = np.array([_t_quantile(q, _effective_df(int(d))) for d in days], dtype=float)
    return out


# ---------------------------------------------------------------------------
def _self_test() -> int:
    print("volatility.py self-test\n" + "=" * 62)
    ok = True
    rng = np.random.default_rng(0)

    print("\n1. Quantiles")
    for p, want in ((0.975, 1.959964), (0.95, 1.644854), (0.84134, 1.0)):
        got = _normal_quantile(p)
        good = abs(got - want) < 1e-4
        ok &= good
        print(f"   normal q({p})      = {got:.6f}  want {want:.6f}  {'ok' if good else 'FAIL'}")
    # A unit-variance t must be tighter than normal in the middle and fatter far out.
    mid, tail = _t_quantile(0.75, 4.0), _t_quantile(0.999, 4.0)
    good = mid < _normal_quantile(0.75) and tail > _normal_quantile(0.999)
    ok &= good
    print(f"   t(4) q(0.75)={mid:.4f} < normal {_normal_quantile(0.75):.4f}, "
          f"q(0.999)={tail:.4f} > normal {_normal_quantile(0.999):.4f}  "
          f"{'ok' if good else 'FAIL'}  (fatter tails, same variance)")

    print("\n2. Quantiles match published t critical values")
    # Unit variance is an algebraic identity once the quantiles are right
    # (Var(T) = df/(df-2), so scaling by sqrt((df-2)/df) gives exactly 1). The
    # thing worth testing is therefore the quantiles themselves. Estimating the
    # variance numerically instead would fail at df=4 purely because that
    # integral converges too slowly for a finite grid — a property of the test,
    # not of the code.
    from math import sqrt as _sqrt
    TABLE = {
        (4.0, 0.75): 0.7407, (4.0, 0.90): 1.5332, (4.0, 0.95): 2.1318,
        (4.0, 0.975): 2.7764, (4.0, 0.99): 3.7469,
        (10.0, 0.90): 1.3722, (10.0, 0.95): 1.8125, (10.0, 0.975): 2.2281,
        (30.0, 0.95): 1.6973, (30.0, 0.975): 2.0423,
    }
    worst = 0.0
    for (df, p), want in TABLE.items():
        raw = _t_quantile(p, df) / _sqrt((df - 2) / df)   # undo the unit-variance scaling
        worst = max(worst, abs(raw - want))
    good = worst < 5e-3
    ok &= good
    print(f"   {len(TABLE)} table values, largest error {worst:.2e}  "
          f"{'ok' if good else 'FAIL'}")

    print("\n3. Mean reversion pulls a shock back down")
    calm = rng.normal(0, 0.005, 800)
    shocked = np.concatenate([calm, rng.normal(0, 0.05, 20)])   # a turbulent last month
    a = float(np.sqrt(forecast_variance_path(shocked, 5).mean()))
    b = float(np.sqrt(forecast_variance_path(shocked, 252).mean()))
    good = a > b
    ok &= good
    print(f"   after a shock: 5d avg vol {a*100:.2f}%/day, 252d avg vol {b*100:.2f}%/day"
          f"  {'ok' if good else 'FAIL'}")

    # The comparison that matters: a 1-year band must NOT be built by taking
    # today's elevated volatility and extrapolating it out a full year.
    naive = float(np.sqrt(ewma_variance(shocked) * 252))
    smart = float(horizon_sigma(shocked, 252)[-1])
    good = smart < naive
    ok &= good
    print(f"   1y sigma: extrapolate today's vol {naive*100:5.1f}%  vs  "
          f"mean-reverting {smart*100:5.1f}%  {'ok' if good else 'FAIL'}")

    print("\n4. sigma grows with horizon, sub-linearly in sqrt(t) after a shock")
    s = horizon_sigma(shocked, 252)
    good = bool(np.all(np.diff(s) > 0)) and s[-1] < s[0] * np.sqrt(252)
    ok &= good
    print(f"   monotone increasing: {bool(np.all(np.diff(s) > 0))}, "
          f"below the constant-vol sqrt(t) line: {bool(s[-1] < s[0]*np.sqrt(252))}  "
          f"{'ok' if good else 'FAIL'}")

    print("\n5. Coverage on data whose truth is known (constant-vol normal walk)")
    hits = {0.5: 0, 0.8: 0, 0.95: 0}
    trials = 4000
    for _ in range(trials):
        r = rng.normal(0, 0.012, 900)
        sig = horizon_sigma(r[:800], 21)[-1]
        actual = float(np.sum(r[800:821]))
        for lv in hits:
            k = _t_quantile(0.5 + lv / 2, _effective_df(21))
            hits[lv] += abs(actual) <= k * sig
    for lv, n in hits.items():
        got = n / trials
        good = abs(got - lv) < 0.05
        ok &= good
        print(f"   {int(lv*100)}% band covered {got*100:5.1f}%  want {int(lv*100)}%  "
              f"{'ok' if good else 'FAIL'}")

    print("\n" + "=" * 62)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
