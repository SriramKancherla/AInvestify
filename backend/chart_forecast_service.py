"""Chart-forecast API bridge to backend/prediction (the diffusion model pipeline).

Inference runs in a **subprocess** so PyTorch's OpenMP runtime never shares the
uvicorn process with XGBoost/sklearn (dual libomp → SIGSEGV on macOS).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from backend.scoremodel.engine import REPO_ROOT

PREDICTION_ROOT = REPO_ROOT / "backend" / "prediction"
FORECAST_RUNNER = PREDICTION_ROOT / "forecast_runner.py"
# Released weights only. A best.pt from a later training run is not the demo.
CHECKPOINT_CANDIDATES = (
    PREDICTION_ROOT / "checkpoints_stable" / "latest.pt",
)
MANIFEST_PATH = PREDICTION_ROOT / "data" / "processed" / "manifest_split.csv"
HORIZONS = [5, 21, 63, 126, 189, 252]

_SUBPROC_LOCK = threading.Lock()
_RESULT_CACHE: dict[tuple[str, int, str], dict[str, Any]] = {}
_RESULT_CACHE_MAX = 32
# Ensemble sampling draws several DDIM samples per request (batched, but
# still several times the work of a single draw), so allow more headroom
# than the old single-sample budget.
_FORECAST_TIMEOUT_S = 420
_CACHE_VERSION = "ensemble-chart-v4"


def _runner_fingerprint() -> str:
    """Identity of the rendering code currently on disk.

    The cache lives in the long-lived uvicorn process, but the code that
    actually produces the PNG (`forecast_runner.py`) is re-executed as a
    fresh subprocess on every request. So editing the renderer takes effect
    instantly *unless* a cached result short-circuits the call — which is
    exactly how three separate rendering fixes in this project shipped
    correctly and still appeared to change nothing, because the manual
    `_CACHE_VERSION` bump was forgotten each time.

    Folding the runner's size+mtime into the key removes the manual step:
    touch the renderer, and every cached entry for it is invalidated
    automatically. Falls back to the static version string if the file
    can't be stat'd, so a missing file degrades to the old behaviour
    rather than raising here.
    """
    try:
        st = FORECAST_RUNNER.stat()
        return f"{_CACHE_VERSION}:{st.st_size}:{int(st.st_mtime_ns)}"
    except OSError:
        return _CACHE_VERSION


def _forecast_mode() -> str:
    """Which forecast the runner will produce. Read per call rather than at
    import so the env var can be changed without worrying about load order."""
    return (os.environ.get("AINVESTIFY_FORECAST_MODE", "model") or "model").strip().lower()


def clear_forecast_cache() -> None:
    _RESULT_CACHE.clear()


def _cache_key(sym: str, horizon_days: int) -> tuple[str, int, str]:
    return (sym, int(horizon_days), _runner_fingerprint())


def _pick_checkpoint() -> Path | None:
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return None


def get_chart_forecast_status() -> dict[str, Any]:
    """Lightweight status — no torch import in the API process."""
    checkpoint = _pick_checkpoint()
    checkpoint_exists = checkpoint is not None
    manifest_exists = MANIFEST_PATH.exists()
    mode = _forecast_mode()
    cone = mode != "model"
    return {
        "model": "volatility_cone" if cone else "chart_forecast_diffusion",
        "forecast_mode": mode,
        "root": str(PREDICTION_ROOT),
        "horizons_days": HORIZONS,
        "manifest_path": str(MANIFEST_PATH),
        "manifest_exists": manifest_exists,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "checkpoint_exists": checkpoint_exists,
        "samples_dir": str(PREDICTION_ROOT / "data" / "processed" / "samples"),
        "live_inference": cone or checkpoint_exists,
        # The cone needs no checkpoint and no torch, so it is ready whenever the
        # runner file is present.
        "ready": True if cone else checkpoint_exists,
        "mode": (
            "cone"
            if cone
            else (
                "dataset"
                if (checkpoint_exists and manifest_exists)
                else ("live" if checkpoint_exists else "unavailable")
            )
        ),
        "conditioning_note": (
            "This cone is built from adjusted prices only. It does not read fundamentals or Form 8-K filings."
            if cone
            else (
                "Fundamentals are the SEC values that were public on the window end. "
                "Form 8-K tone is the item-code score of filings in the window. "
                "A day with no filing stays neutral."
            )
        ),
        "isolation": "subprocess",
    }


def get_chart_forecast_image(
    ticker: str, horizon_days: int = 63, refresh: bool = False
) -> dict[str, Any]:
    """`refresh=True` skips the cache read and regenerates — the escape hatch
    for "I changed something and want to see it now" without a restart."""
    sym = (ticker or "").strip().upper()
    if not sym:
        raise ValueError("Ticker is required.")
    if horizon_days not in HORIZONS:
        raise ValueError(f"horizon_days must be one of {HORIZONS}")
    if not FORECAST_RUNNER.exists():
        raise RuntimeError(f"Chart forecast runner not found at {FORECAST_RUNNER}")
    # Only the diffusion path needs a checkpoint. The cone is built from
    # price history alone.
    if _forecast_mode() == "model" and _pick_checkpoint() is None:
        raise RuntimeError(
            "Chart forecast not ready: model checkpoint missing "
            "(expected backend/prediction/checkpoints_stable/latest.pt). "
            "Unset AINVESTIFY_FORECAST_MODE, or set it to cone, if the checkpoint is missing."
        )

    cache_key = _cache_key(sym, horizon_days)
    if refresh:
        _RESULT_CACHE.pop(cache_key, None)
    else:
        cached = _RESULT_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

    env = os.environ.copy()
    env.setdefault("AINVESTIFY_FORECAST_DEVICE", "cpu")
    # Keep OpenMP quiet in the child; API process never loads torch.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    # Avoid inheriting a broken sandbox HTTP proxy that blocks Yahoo Finance.
    for proxy_key in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
        "SOCKS_PROXY", "SOCKS5_PROXY", "socks_proxy", "socks5_proxy",
    ):
        env.pop(proxy_key, None)

    cmd = [
        sys.executable,
        str(FORECAST_RUNNER),
        "--ticker",
        sym,
        "--horizon-days",
        str(int(horizon_days)),
    ]

    with _SUBPROC_LOCK:
        if not refresh:
            cached = _RESULT_CACHE.get(cache_key)
            if cached is not None:
                return {**cached, "cached": True}

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=_FORECAST_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Chart forecast timed out after {_FORECAST_TIMEOUT_S}s for {sym}."
            ) from exc

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        try:
            payload = json.loads(proc.stdout or "{}")
            if payload.get("error"):
                err = str(payload["error"])
        except json.JSONDecodeError:
            pass
        err = err or f"exit {proc.returncode}"
        err_lines = [ln for ln in err.splitlines() if ln.strip()][-8:]
        raise RuntimeError("Chart forecast failed: " + " | ".join(err_lines))

    raw = (proc.stdout or "").strip()
    if not raw:
        raise RuntimeError(
            "Chart forecast returned empty output. "
            + ((proc.stderr or "").strip()[-500:] or "No stderr.")
        )
    # Tolerate accidental log lines on stdout: parse the last JSON object.
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.rfind("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(raw[start : end + 1])
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Chart forecast returned invalid JSON: "
                    + raw[:200]
                    + ((" | stderr: " + proc.stderr.strip()[:300]) if proc.stderr else "")
                ) from exc
        else:
            raise RuntimeError(
                "Chart forecast returned invalid JSON: "
                + raw[:200]
                + ((" | stderr: " + proc.stderr.strip()[:300]) if proc.stderr else "")
            )

    if payload.get("error") and not payload.get("image_base64"):
        raise RuntimeError(str(payload["error"]))

    if not payload.get("image_base64"):
        raise RuntimeError(payload.get("error") or "Forecast response missing image.")

    result = {
        "ticker": payload.get("ticker", sym),
        "horizon_days": int(payload.get("horizon_days", horizon_days)),
        "window_start": payload.get("window_start"),
        "window_end": payload.get("window_end"),
        "target_start": payload.get("target_start"),
        "target_end": payload.get("target_end"),
        "source": payload.get("source"),
        "inference_seconds": payload.get("inference_seconds"),
        "image_base64": payload["image_base64"],
        "content_type": "image/png",
        # Default ("cone"): a flat centre with calibrated probability bands
        # derived from a volatility forecast — the only part of a price path
        # measured to be forecastable. Carries `bands`, `levels` and
        # `primary_level`. Under AINVESTIFY_FORECAST_MODE=model this is instead
        # the diffusion ensemble's mean shape, which has no measured skill.
        "price_estimate": payload.get("price_estimate"),
        # "cone", "chart" (model with a real price axis), or "generated_bitmap".
        "render_mode": payload.get("render_mode"),
        # Which forecast produced this, so the UI can label it honestly.
        "forecast_mode": payload.get("forecast_mode", "model"),
        "ensemble_members": payload.get("ensemble_members"),
        # History + forecast points, so the frontend can draw an interactive
        # chart. None on the bitmap fallback path.
        "series": payload.get("series"),
        "conditioning_note": payload.get(
            "conditioning_note",
            "Fundamentals are the SEC values that were public on the window end. "
            "Form 8-K tone is the item-code score of filings in the window. "
            "A day with no filing stays neutral.",
        ),
        "cached": False,
    }
    if len(_RESULT_CACHE) >= _RESULT_CACHE_MAX:
        _RESULT_CACHE.pop(next(iter(_RESULT_CACHE)))
    _RESULT_CACHE[cache_key] = {k: v for k, v in result.items() if k != "cached"}
    return result
