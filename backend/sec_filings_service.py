"""Recent SEC filings for a ticker, straight from EDGAR.

Reads SEC's free submissions API — no key, no scraping, no third-party
aggregator. The only requirement SEC imposes is a real contact in the
User-Agent header (see `config.SEC_USER_AGENT`); placeholder agents get 403'd
and repeat offenders get IP-banned, so that value is checked at startup rather
than left to fail mysteriously on the first request.

WHAT IT SHOWS AND WHY
Only the forms a person actually reads:

    10-K      annual report
    10-Q      quarterly report
    8-K       material event — the "something happened" filing
    DEF 14A   proxy statement (executive pay, board, shareholder votes)
    S-1       registration / IPO

Deliberately EXCLUDES Forms 3/4/5, the insider-transaction filings. For a large
company those are 70-90% of all filings by count — Apple files hundreds a year —
so including them would push every 10-K and 8-K off the first page. They are
genuinely useful, just a different feature with different UI needs.

Amendments (`10-K/A`, `8-K/A`) ARE included and flagged, because an amended
filing is often the more interesting one.

8-K item codes are decoded to plain English. "2.02" on its own is meaningless to
a reader; "Results of operations" is the whole point of the filing.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
import urllib.error
import urllib.request
import zlib
from typing import Any

from backend.scoremodel.engine import REPO_ROOT

PREDICTION_ROOT = REPO_ROOT / "backend" / "prediction"
_CACHE_DIR = PREDICTION_ROOT / "data" / "cache"
_CIK_MAP_CACHE = _CACHE_DIR / "sec_cik_map.json"

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"

# Prefix match, so "10-K" also catches "10-K/A".
KEY_FORMS = ("10-K", "10-Q", "8-K", "DEF 14A", "S-1")

FORM_LABELS = {
    "10-K": "Annual report",
    "10-Q": "Quarterly report",
    "8-K": "Material event",
    "DEF 14A": "Proxy statement",
    "S-1": "Registration statement",
}

# Plain-English 8-K item codes. Same code list as fetch_news_sec8k.py's tone map,
# but worded for a reader rather than scored for a model.
ITEM_LABELS = {
    "1.01": "Entered a material agreement",
    "1.02": "Terminated a material agreement",
    "1.03": "Bankruptcy or receivership",
    "1.04": "Mine safety report",
    "1.05": "Cybersecurity incident",
    "2.01": "Completed an acquisition or disposal",
    "2.02": "Results of operations",
    "2.03": "Took on a financial obligation",
    "2.04": "Obligation acceleration triggered",
    "2.05": "Exit or disposal costs",
    "2.06": "Material impairment",
    "3.01": "Delisting notice",
    "3.02": "Unregistered equity sale",
    "3.03": "Change to shareholder rights",
    "4.01": "Changed auditor",
    "4.02": "Prior financials not reliable",
    "5.02": "Officer or director change",
    "5.03": "Bylaws or fiscal year change",
    "5.04": "Benefit plan trading suspension",
    "5.05": "Code of ethics change",
    "5.07": "Shareholder vote results",
    "7.01": "Regulation FD disclosure",
    "8.01": "Other event",
    "9.01": "Financial statements and exhibits",
}

_TIMEOUT_S = 20
_CACHE_TTL_S = 6 * 3600      # filings appear a few times a month; 6h is plenty
_MIN_INTERVAL_S = 0.12       # SEC asks for <=10 req/s; this is ~8/s with margin

_lock = threading.Lock()
_last_request_at = 0.0
_cache: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}
_cik_map: dict[str, str] | None = None


class SecConfigError(RuntimeError):
    """SEC_USER_AGENT is still a placeholder. Raised instead of letting SEC
    return an opaque 403 that looks like a network fault."""


def _load_config():
    """Import prediction config as a PACKAGE module.

    `forecast_runner.py` reaches it by putting PREDICTION_ROOT on sys.path and
    doing `import config`, which is right for a standalone subprocess. Doing the
    same here would load a *second*, independent copy inside the API process —
    two module objects, two pandas imports, and an env change applying to one but
    not the other. Since this service already imports from `backend.*`, the
    package path is available; the sys.path form stays as a fallback for anyone
    running this file directly.
    """
    try:
        from backend.prediction import config as cfg
        return cfg
    except ImportError:
        import sys

        if str(PREDICTION_ROOT) not in sys.path:
            sys.path.insert(0, str(PREDICTION_ROOT))
        import config as cfg
        return cfg


def _user_agent() -> str:
    config = _load_config()
    ua = (getattr(config, "SEC_USER_AGENT", "") or "").strip()
    if not ua or "you@example.com" in ua or "@" not in ua:
        raise SecConfigError(
            "SEC_USER_AGENT in backend/prediction/config.py must contain a real "
            "contact email. SEC blocks placeholder user agents with HTTP 403."
        )
    return ua


def _get(url: str, retries: int = 3) -> Any:
    """One throttled, retried GET. SEC rate-limits by IP, and a burst from a
    page load plus a background job is exactly how a project gets banned, so the
    throttle is global to this process rather than per call site."""
    global _last_request_at
    headers = {"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"}

    last_err: Exception | None = None
    for attempt in range(retries):
        with _lock:
            gap = time.time() - _last_request_at
            if gap < _MIN_INTERVAL_S:
                time.sleep(_MIN_INTERVAL_S - gap)
            _last_request_at = time.time()
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                raw = resp.read()
                encoding = (resp.headers.get("Content-Encoding") or "").lower()
                if "gzip" in encoding or raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                elif "deflate" in encoding:
                    raw = zlib.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 403:
                raise SecConfigError(
                    "SEC returned 403. This almost always means the User-Agent is "
                    "not an acceptable contact, or this IP is rate-limited."
                ) from e
            last_err = e
            time.sleep((attempt + 1) * 1.5)
        except Exception as e:
            last_err = e
            time.sleep((attempt + 1) * 1.5)
    raise RuntimeError(f"SEC request failed after {retries} attempts: {last_err}")


def _load_cik_map() -> dict[str, str]:
    """Ticker -> zero-padded 10-digit CIK. Cached to disk; the file changes at
    most weekly and is ~1MB, so re-fetching it per request would be rude."""
    global _cik_map
    if _cik_map is not None:
        return _cik_map

    if _CIK_MAP_CACHE.exists():
        try:
            _cik_map = json.loads(_CIK_MAP_CACHE.read_text())
            if _cik_map:
                return _cik_map
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache: fall through and refetch

    raw = _get(SEC_TICKERS_URL)
    if not raw:
        raise RuntimeError("Could not load SEC ticker->CIK map.")
    mapping = {
        str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10)
        for v in raw.values()
        if v.get("ticker")
    }
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CIK_MAP_CACHE.write_text(json.dumps(mapping))
    except OSError:
        pass  # a read-only FS shouldn't break the lookup, just the caching
    _cik_map = mapping
    return mapping


def _is_key_form(form: str) -> bool:
    return any(form == f or form.startswith(f + "/") for f in KEY_FORMS)


def _describe_items(items: str) -> list[str]:
    if not items:
        return []
    out = []
    for code in (c.strip() for c in items.split(",")):
        if code:
            out.append(ITEM_LABELS.get(code, f"Item {code}"))
    return out


def _row_to_filing(cik: str, block: dict, i: int) -> dict[str, Any]:
    def col(name: str, default=""):
        seq = block.get(name) or []
        return seq[i] if i < len(seq) else default

    form = str(col("form"))
    accession = str(col("accessionNumber"))
    acc_plain = accession.replace("-", "")
    cik_plain = str(int(cik))          # EDGAR archive paths drop leading zeros
    doc = str(col("primaryDocument"))
    base_form = next((f for f in KEY_FORMS if form.startswith(f)), form)

    return {
        "form": form,
        "form_label": FORM_LABELS.get(base_form, base_form),
        "is_amendment": "/A" in form,
        "filed": str(col("filingDate")),
        "period": str(col("reportDate")) or None,
        "description": str(col("primaryDocDescription")) or None,
        "items": _describe_items(str(col("items"))),
        "size_bytes": col("size", None),
        # The document itself, and the index page listing every exhibit.
        "url": f"{EDGAR_ARCHIVE}/{cik_plain}/{acc_plain}/{doc}" if doc else None,
        "index_url": f"{EDGAR_ARCHIVE}/{cik_plain}/{acc_plain}/{accession}-index.htm",
        "accession": accession,
    }


def get_sec_filings(ticker: str, limit: int = 25) -> dict[str, Any]:
    """Recent key filings for one ticker, newest first.

    Only the `recent` block of the submissions API is read — it holds roughly the
    last 1,000 filings, which for any active company covers several years of
    10-Ks and 8-Ks. Following the paginated older files would multiply the
    request count for history nobody scrolls to on a dashboard.
    """
    sym = (ticker or "").strip().upper()
    if not sym:
        raise ValueError("Ticker is required.")
    limit = max(1, min(int(limit), 100))

    key = (sym, limit)
    hit = _cache.get(key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL_S:
        return {**hit[1], "cached": True}

    cik = _load_cik_map().get(sym)
    if not cik:
        raise ValueError(f"No SEC CIK on file for {sym}. Non-US listings have no EDGAR filings.")

    data = _get(SEC_SUBMISSIONS_URL.format(cik=cik))
    if not data:
        raise ValueError(f"SEC has no submissions record for {sym} (CIK {cik}).")

    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    filings = [
        _row_to_filing(cik, recent, i)
        for i, form in enumerate(forms)
        if _is_key_form(str(form))
    ]
    filings.sort(key=lambda f: f["filed"], reverse=True)

    counts: dict[str, int] = {}
    for f in filings:
        base = next((k for k in KEY_FORMS if f["form"].startswith(k)), f["form"])
        counts[base] = counts.get(base, 0) + 1

    result = {
        "ticker": sym,
        "cik": cik,
        "company": data.get("name") or sym,
        "exchange": (data.get("exchanges") or [None])[0],
        "sic_description": data.get("sicDescription"),
        "fiscal_year_end": data.get("fiscalYearEnd"),
        # Every key filing SEC still lists in its recent block, before truncation.
        "total_available": len(filings),
        "counts_by_form": counts,
        "forms_included": list(KEY_FORMS),
        "filings": filings[:limit],
        "source": "SEC EDGAR",
        "company_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=exclude&count=40",
        "cached": False,
    }
    _cache[key] = (time.time(), {k: v for k, v in result.items() if k != "cached"})
    if len(_cache) > 64:
        _cache.pop(next(iter(_cache)))
    return result


def get_sec_filings_status() -> dict[str, Any]:
    """Whether this will work at all, without making a request."""
    try:
        ua = _user_agent()
        return {"ready": True, "user_agent": ua, "forms": list(KEY_FORMS),
                "cik_map_cached": _CIK_MAP_CACHE.exists()}
    except SecConfigError as e:
        return {"ready": False, "error": str(e), "forms": list(KEY_FORMS),
                "cik_map_cached": _CIK_MAP_CACHE.exists()}
