from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from backend.insights_service import get_chart_data


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


_SYNC_STATE: dict[str, dict[str, Any]] = {}


def get_events(ticker: str) -> dict:
    """
    Best-effort upcoming ticker-specific events.

    Tries yfinance's calendar (earnings/dividend dates). Falls back to an empty list if
    no reliable events are available for the ticker.
    """
    sym = ticker.upper().strip()
    events: list[dict[str, Any]] = []

    try:
        import pandas as pd  # type: ignore
        import yfinance as yf  # type: ignore

        t = yf.Ticker(sym)

        def _to_iso(v: Any) -> str | None:
            if v is None:
                return None
            try:
                if isinstance(v, (list, tuple)) and v:
                    v = v[0]
                if isinstance(v, pd.Timestamp):
                    return v.date().isoformat()
                if isinstance(v, (int, float)) and v > 0:
                    return datetime.fromtimestamp(float(v), tz=timezone.utc).date().isoformat()
                if hasattr(v, "to_pydatetime"):
                    return v.to_pydatetime().date().isoformat()  # type: ignore[attr-defined]
                s = str(v).strip()
                if not s or s.lower() in {"nan", "nat", "none"}:
                    return None
                dt = pd.to_datetime(s, errors="coerce", utc=True)
                if pd.isna(dt):
                    return None
                return dt.date().isoformat()
            except Exception:
                return None

        def _canonical_type(ev_type: str) -> str:
            s = ev_type.lower()
            if "earning" in s:
                return "Earnings Date"
            if "ex-dividend" in s:
                return "Ex-Dividend Date"
            if "dividend" in s:
                return "Dividend Date"
            return ev_type

        def _add(ev_type: str, date_iso: str | None, details: str, source: str, risk: str) -> None:
            if not date_iso:
                return
            ev_type = _canonical_type(ev_type)
            events.append(
                {
                    "id": _uid("event"),
                    "ticker": sym,
                    "event_type": ev_type,
                    "event_date": date_iso,
                    "risk_level": risk,
                    "details": details,
                    "source": source,
                }
            )

        try:
            cal = t.calendar
            if cal is not None and getattr(cal, "empty", True) is False:
                series = cal.iloc[:, 0] if getattr(cal, "shape", (0, 0))[1] >= 1 else cal.squeeze()
                cal_map = {str(k): v for k, v in (series.to_dict().items() if hasattr(series, "to_dict") else [])}
                _add(
                    "Earnings Date",
                    _to_iso(cal_map.get("Earnings Date")),
                    "Expected earnings announcement window.",
                    "Yahoo Finance Calendar",
                    "high",
                )
                _add(
                    "Ex-Dividend Date",
                    _to_iso(cal_map.get("Ex-Dividend Date")),
                    "Ex-dividend date (if applicable).",
                    "Yahoo Finance Calendar",
                    "medium",
                )
                _add(
                    "Dividend Date",
                    _to_iso(cal_map.get("Dividend Date")),
                    "Dividend payment date (if applicable).",
                    "Yahoo Finance Calendar",
                    "low",
                )
        except Exception:
            pass

        try:
            edf = t.get_earnings_dates(limit=12)
            if edf is not None and getattr(edf, "empty", True) is False:
                for idx, row in edf.iterrows():
                    event_date = _to_iso(idx)
                    if not event_date:
                        continue
                    est = row.get("EPS Estimate")
                    rep = row.get("Reported EPS")
                    surprise = row.get("Surprise(%)")
                    parts = []
                    if est is not None and str(est).lower() not in {"nan", "none"}:
                        parts.append(f"EPS est: {est}")
                    if rep is not None and str(rep).lower() not in {"nan", "none"}:
                        parts.append(f"reported EPS: {rep}")
                    if surprise is not None and str(surprise).lower() not in {"nan", "none"}:
                        parts.append(f"surprise: {surprise}%")
                    details = "Quarterly earnings event."
                    if parts:
                        details += " " + ", ".join(parts) + "."
                    _add(
                        "Earnings Release",
                        event_date,
                        details,
                        "Yahoo Earnings Dates",
                        "high",
                    )
        except Exception:
            pass

        try:
            info = t.info or {}
            _add(
                "Earnings Date",
                _to_iso(info.get("earningsTimestamp")),
                "Earnings timestamp from quote metadata.",
                "Yahoo Quote Metadata",
                "high",
            )
            _add(
                "Earnings Start",
                _to_iso(info.get("earningsTimestampStart")),
                "",
                "Yahoo Quote Metadata",
                "high",
            )
            _add(
                "Earnings End",
                _to_iso(info.get("earningsTimestampEnd")),
                "",
                "Yahoo Quote Metadata",
                "medium",
            )
            _add(
                "Ex-Dividend Date",
                _to_iso(info.get("exDividendDate")),
                "Ex-dividend date from quote metadata.",
                "Yahoo Quote Metadata",
                "medium",
            )
            _add(
                "Dividend Date",
                _to_iso(info.get("dividendDate")),
                "Dividend payment date from quote metadata.",
                "Yahoo Quote Metadata",
                "low",
            )
        except Exception:
            pass

        today = datetime.now(timezone.utc).date()
        upcoming: list[dict[str, Any]] = []
        for ev in events:
            try:
                d = datetime.fromisoformat(str(ev.get("event_date"))).date()
            except Exception:
                continue
            if d >= today:
                upcoming.append(ev)

        seen: set[tuple[str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for ev in sorted(upcoming, key=lambda x: str(x.get("event_date") or "")):
            key = (str(ev.get("event_type") or ""), str(ev.get("event_date") or ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ev)

        return {"ticker": sym, "events": deduped[:12]}
    except Exception:
        return {"ticker": sym, "events": []}


def run_backtest(ticker: str, lookback_period: str = "1y") -> dict:
    chart = get_chart_data(ticker, period=lookback_period, interval="1d")
    points = chart.get("points", [])
    if len(points) < 3:
        return {"ticker": ticker.upper(), "strategy": "momentum", "error": "insufficient_history"}
    closes = [float(p["close"]) for p in points]
    start_price = closes[0]
    end_price = closes[-1]
    buy_hold_return = (end_price - start_price) / start_price if start_price else 0.0

    strategy_ret = 1.0
    wins = 0
    losses = 0
    for i in range(2, len(closes)):
        signal_long = closes[i - 1] > closes[i - 2]
        daily_ret = (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] else 0.0
        strat_daily = daily_ret if signal_long else 0.0
        if strat_daily > 0:
            wins += 1
        elif strat_daily < 0:
            losses += 1
        strategy_ret *= 1.0 + strat_daily
    strategy_ret -= 1.0
    return {
        "ticker": ticker.upper(),
        "strategy": "simple_momentum",
        "lookback_period": lookback_period,
        "stats": {
            "buy_hold_return": buy_hold_return,
            "strategy_return": strategy_ret,
            "win_days": wins,
            "loss_days": losses,
            "trading_days": len(closes) - 2,
        },
        "generated_at": _now_iso(),
    }


def issue_guest_token() -> dict:
    token = _uid("guest")
    if token not in _SYNC_STATE:
        _SYNC_STATE[token] = {"created_at": _now_iso(), "state": {}}
    return {"token": token, "created_at": _SYNC_STATE[token]["created_at"]}


def sync_push(token: str, state: dict) -> dict:
    _SYNC_STATE[token] = {"updated_at": _now_iso(), "state": state}
    return {"ok": True, "updated_at": _SYNC_STATE[token]["updated_at"]}


def sync_pull(token: str) -> dict:
    item = _SYNC_STATE.get(token)
    if not item:
        return {"state": {}, "missing": True}
    return {"state": item.get("state", {}), "updated_at": item.get("updated_at") or item.get("created_at")}
