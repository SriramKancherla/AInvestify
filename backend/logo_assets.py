"""Resolve locally cached stock logos for PDF rendering."""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGOS_DIR = ROOT / "frontend" / "public" / "logos"
MANIFEST_PATH = LOGOS_DIR / "manifest.json"


@lru_cache(maxsize=1)
def _manifest() -> frozenset[str]:
    if not MANIFEST_PATH.is_file():
        return frozenset()
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return frozenset(str(s).strip().upper() for s in data if str(s).strip())
    except (OSError, json.JSONDecodeError):
        pass
    return frozenset()


def logo_path(ticker: str | None) -> Path | None:
    sym = (ticker or "").strip().upper()
    if not sym:
        return None
    path = LOGOS_DIR / f"{sym}.png"
    if path.is_file() and path.stat().st_size > 80:
        return path
    if sym in _manifest():
        return path if path.is_file() else None
    return path if path.is_file() else None


def logo_data_uri(ticker: str | None) -> str | None:
    path = logo_path(ticker)
    if not path:
        return None
    try:
        raw = path.read_bytes()
        if len(raw) < 80:
            return None
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except OSError:
        return None
