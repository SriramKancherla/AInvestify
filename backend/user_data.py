from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import requests

from backend.insights_service import get_chart_data
from backend.notify import send_price_alert_email

_LOG = logging.getLogger("ainvestify.user_data")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _supabase_url() -> str:
    return (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")


def _service_key() -> str:
    return (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()


def _use_supabase() -> bool:
    return bool(_supabase_url() and _service_key())


def _sb_headers() -> dict[str, str]:
    key = _service_key()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _sb_req(method: str, table: str, params: dict[str, str] | None = None, json_body: Any = None, extra_headers: dict[str, str] | None = None) -> requests.Response:
    url = f"{_supabase_url()}/rest/v1/{table}"
    headers = dict(_sb_headers())
    if extra_headers:
        headers.update(extra_headers)
    return requests.request(method, url, headers=headers, params=params, json=json_body, timeout=30)


def _enrich_holdings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_rows = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    for h in rows:
        ticker = str(h.get("ticker") or "").upper()
        qty = float(h.get("quantity") or 0)
        avg = float(h.get("avg_buy_price") or 0)
        hid = str(h.get("id"))
        created_at = str(h.get("created_at") or _now_iso())
        updated_at = str(h.get("updated_at") or created_at)
        try:
            chart = get_chart_data(ticker, period="5d", interval="1d")
            points = chart.get("points", [])
            last_close = float(points[-1]["close"]) if points else avg
        except Exception:
            last_close = avg
        market_value = qty * last_close
        cost_basis = qty * avg
        pnl = market_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100.0) if cost_basis else 0.0
        total_market_value += market_value
        total_cost_basis += cost_basis
        out_rows.append(
            {
                "id": hid,
                "ticker": ticker,
                "quantity": qty,
                "avg_buy_price": avg,
                "last_price": last_close,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
    total_pnl = total_market_value - total_cost_basis
    total_pnl_pct = (total_pnl / total_cost_basis * 100.0) if total_cost_basis else 0.0
    return {
        "holdings": out_rows,
        "summary": {
            "total_market_value": total_market_value,
            "total_cost_basis": total_cost_basis,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
        },
    }


# --- In-memory fallback (dev / tests when Supabase not configured) ---

_mem_portfolio: dict[str, list[dict[str, Any]]] = defaultdict(list)
_mem_watchlists: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"default": []})
_mem_alerts: dict[str, list[dict[str, Any]]] = defaultdict(list)


def _mem_list_portfolio(user_id: str) -> dict[str, Any]:
    rows = []
    for h in _mem_portfolio[user_id]:
        rows.append(
            {
                "id": h["id"],
                "ticker": h["ticker"],
                "quantity": h["quantity"],
                "avg_buy_price": h["avg_buy_price"],
                "created_at": h.get("created_at", _now_iso()),
                "updated_at": h.get("updated_at", _now_iso()),
            }
        )
    return _enrich_holdings(rows)


def _mem_add_holding(user_id: str, ticker: str, quantity: float, avg_buy_price: float) -> dict[str, Any]:
    hid = str(uuid.uuid4())
    now = _now_iso()
    row = {
        "id": hid,
        "ticker": ticker.upper().strip(),
        "quantity": max(0.0, quantity),
        "avg_buy_price": max(0.0, avg_buy_price),
        "created_at": now,
        "updated_at": now,
    }
    _mem_portfolio[user_id].append(row)
    return {"holding": row}


def _mem_delete_holding(user_id: str, holding_id: str) -> dict[str, Any]:
    before = len(_mem_portfolio[user_id])
    _mem_portfolio[user_id] = [h for h in _mem_portfolio[user_id] if str(h["id"]) != holding_id]
    return {"deleted": before != len(_mem_portfolio[user_id])}


def _mem_list_watchlists(user_id: str) -> dict[str, Any]:
    wmap = _mem_watchlists[user_id]
    return {"watchlists": [{"name": k, "tickers": list(v)} for k, v in sorted(wmap.items())]}


def _mem_upsert_watchlist(user_id: str, name: str, tickers: list[str]) -> dict[str, Any]:
    clean_name = name.strip() or "default"
    _mem_watchlists[user_id][clean_name] = sorted({t.upper().strip() for t in tickers if t.strip()})
    return {"name": clean_name, "tickers": _mem_watchlists[user_id][clean_name]}


def _mem_create_alert(
    user_id: str, ticker: str, rule_type: str, threshold: float, channel_email: bool, channel_whatsapp: bool
) -> dict[str, Any]:
    now = _now_iso()
    aid = str(uuid.uuid4())
    row = {
        "id": aid,
        "ticker": ticker.upper().strip(),
        "rule_type": rule_type,
        "threshold": threshold,
        "channel_email": channel_email,
        "channel_whatsapp": channel_whatsapp,
        "enabled": True,
        "created_at": now,
        "updated_at": now,
    }
    _mem_alerts[user_id].append(row)
    return {"alert": _alert_api(row)}


def _alert_api(row: dict[str, Any]) -> dict[str, Any]:
    cw = bool(row.get("channel_whatsapp"))
    return {
        "id": str(row["id"]),
        "ticker": str(row["ticker"]),
        "rule_type": str(row["rule_type"]),
        "threshold": float(row["threshold"]),
        "channel_email": bool(row.get("channel_email", True)),
        "channel_whatsapp": cw,
        "channel_push": cw,
        "enabled": bool(row.get("enabled", True)),
        "created_at": str(row.get("created_at", "")),
        "updated_at": str(row.get("updated_at", "")),
    }


def _mem_list_alerts(user_id: str) -> dict[str, Any]:
    return {"alerts": [_alert_api(a) for a in _mem_alerts[user_id]]}


def _mem_delete_alert(user_id: str, alert_id: str) -> dict[str, Any]:
    before = len(_mem_alerts[user_id])
    _mem_alerts[user_id] = [a for a in _mem_alerts[user_id] if str(a["id"]) != alert_id]
    return {"deleted": before != len(_mem_alerts[user_id])}


def _mem_evaluate(user_id: str, user_email: str | None, only_alert_id: str | None = None) -> dict[str, Any]:
    """One-shot rules: when price satisfies the rule, notify then remove the rule."""
    triggered: list[dict[str, Any]] = []
    for a in list(_mem_alerts[user_id]):
        if not a.get("enabled", True):
            continue
        aid = str(a["id"])
        if only_alert_id and aid != only_alert_id:
            continue
        chart = get_chart_data(str(a["ticker"]), period="5d", interval="1d")
        points = chart.get("points", [])
        if not points:
            continue
        price = float(points[-1]["close"])
        hit = (a["rule_type"] == "price_above" and price >= float(a["threshold"])) or (
            a["rule_type"] == "price_below" and price <= float(a["threshold"])
        )
        if not hit:
            continue
        delivered_email = False
        email_delivery_error: str | None = None
        if a.get("channel_email"):
            if user_email:
                delivered_email, email_delivery_error = send_price_alert_email(
                    user_email, str(a["ticker"]), price, str(a["rule_type"]), float(a["threshold"])
                )
            else:
                email_delivery_error = "no_recipient_email_in_session"
        _mem_delete_alert(user_id, aid)
        triggered.append(
            {
                "alert_id": aid,
                "ticker": str(a["ticker"]),
                "price": price,
                "rule_type": str(a["rule_type"]),
                "threshold": float(a["threshold"]),
                "delivered": {"email": delivered_email, "whatsapp": False},
                "email_delivery_error": email_delivery_error,
                "rule_deleted": True,
                "evaluated_at": _now_iso(),
            }
        )
    return {"triggered_count": len(triggered), "triggered": triggered}


# --- Supabase ---


def _sb_list_portfolio(user_id: str) -> dict[str, Any]:
    r = _sb_req("GET", "portfolio_holdings", params={"user_id": f"eq.{user_id}", "select": "*", "order": "created_at.asc"})
    if r.status_code >= 400:
        _LOG.warning("portfolio list failed: %s %s", r.status_code, r.text[:400])
        raise RuntimeError("Could not load portfolio.")
    rows = r.json()
    if not isinstance(rows, list):
        rows = []
    return _enrich_holdings(rows)


def _sb_add_holding(user_id: str, ticker: str, quantity: float, avg_buy_price: float) -> dict[str, Any]:
    body = {
        "user_id": user_id,
        "ticker": ticker.upper().strip(),
        "quantity": max(0.0, quantity),
        "avg_buy_price": max(0.0, avg_buy_price),
        "updated_at": _now_iso(),
    }
    r = _sb_req("POST", "portfolio_holdings", json_body=[body], extra_headers={"Prefer": "return=representation"})
    if r.status_code >= 400:
        _LOG.warning("portfolio add failed: %s %s", r.status_code, r.text[:400])
        raise RuntimeError("Could not add holding.")
    data = r.json()
    row = data[0] if isinstance(data, list) and data else body | {"id": ""}
    return {"holding": row}


def _sb_delete_holding(user_id: str, holding_id: str) -> dict[str, Any]:
    r = _sb_req(
        "DELETE",
        "portfolio_holdings",
        params={"id": f"eq.{holding_id}", "user_id": f"eq.{user_id}"},
    )
    if r.status_code >= 400:
        return {"deleted": False}
    return {"deleted": True}


def _sb_list_watchlists(user_id: str) -> dict[str, Any]:
    r = _sb_req("GET", "watchlists", params={"user_id": f"eq.{user_id}", "select": "name,tickers,updated_at"})
    if r.status_code >= 400:
        raise RuntimeError("Could not load watchlists.")
    rows = r.json()
    if not isinstance(rows, list):
        rows = []
    if not rows:
        return {"watchlists": [{"name": "default", "tickers": []}]}
    return {"watchlists": [{"name": str(x["name"]), "tickers": list(x.get("tickers") or [])} for x in rows]}


def _sb_upsert_watchlist(user_id: str, name: str, tickers: list[str]) -> dict[str, Any]:
    clean_name = name.strip() or "default"
    clean_tickers = sorted({t.upper().strip() for t in tickers if t.strip()})
    body = {"user_id": user_id, "name": clean_name, "tickers": clean_tickers, "updated_at": _now_iso()}
    r = _sb_req(
        "POST",
        "watchlists",
        params={"on_conflict": "user_id,name"},
        json_body=[body],
        extra_headers={"Prefer": "resolution=merge-duplicates,return=representation"},
    )
    if r.status_code >= 400:
        _LOG.warning("watchlist upsert failed: %s %s", r.status_code, r.text[:400])
        raise RuntimeError("Could not save watchlist.")
    data = r.json()
    row = data[0] if isinstance(data, list) and data else body
    return {"name": str(row.get("name", clean_name)), "tickers": list(row.get("tickers", clean_tickers))}


def _sb_fetch_alert_rows(user_id: str) -> list[dict[str, Any]]:
    r = _sb_req("GET", "alert_rules", params={"user_id": f"eq.{user_id}", "select": "*", "order": "created_at.asc"})
    if r.status_code >= 400:
        raise RuntimeError("Could not load alerts.")
    rows = r.json()
    return rows if isinstance(rows, list) else []


def _sb_list_alerts(user_id: str) -> dict[str, Any]:
    rows = _sb_fetch_alert_rows(user_id)
    return {"alerts": [_alert_api(dict(x)) for x in rows]}


def _sb_create_alert(
    user_id: str, ticker: str, rule_type: str, threshold: float, channel_email: bool, channel_whatsapp: bool
) -> dict[str, Any]:
    body = {
        "user_id": user_id,
        "ticker": ticker.upper().strip(),
        "rule_type": rule_type,
        "threshold": threshold,
        "channel_email": channel_email,
        "channel_whatsapp": channel_whatsapp,
        "enabled": True,
        "updated_at": _now_iso(),
    }
    r = _sb_req("POST", "alert_rules", json_body=[body], extra_headers={"Prefer": "return=representation"})
    if r.status_code >= 400:
        raise RuntimeError("Could not create alert.")
    data = r.json()
    row = data[0] if isinstance(data, list) and data else body
    return {"alert": _alert_api(dict(row))}


def _sb_delete_alert(user_id: str, alert_id: str) -> dict[str, Any]:
    r = _sb_req("DELETE", "alert_rules", params={"id": f"eq.{alert_id}", "user_id": f"eq.{user_id}"})
    return {"deleted": r.status_code < 400}


def _sb_delete_all_alerts(user_id: str) -> dict[str, Any]:
    r = _sb_req("DELETE", "alert_rules", params={"user_id": f"eq.{user_id}"})
    if r.status_code >= 400:
        _LOG.warning("delete all alerts failed: %s %s", r.status_code, r.text[:400])
        return {"ok": False}
    return {"ok": True}


def _sb_evaluate(user_id: str, user_email: str | None, only_alert_id: str | None = None) -> dict[str, Any]:
    rows = _sb_fetch_alert_rows(user_id)
    triggered: list[dict[str, Any]] = []
    for raw in rows:
        if not raw.get("enabled", True):
            continue
        aid = str(raw.get("id") or "")
        if not aid:
            continue
        if only_alert_id and aid != only_alert_id:
            continue
        chart = get_chart_data(str(raw["ticker"]), period="5d", interval="1d")
        points = chart.get("points", [])
        if not points:
            continue
        price = float(points[-1]["close"])
        hit = (raw["rule_type"] == "price_above" and price >= float(raw["threshold"])) or (
            raw["rule_type"] == "price_below" and price <= float(raw["threshold"])
        )
        if not hit:
            continue
        delivered_email = False
        delivered_whatsapp = False
        email_delivery_error: str | None = None
        if raw.get("channel_email"):
            if user_email:
                delivered_email, email_delivery_error = send_price_alert_email(
                    user_email, str(raw["ticker"]), price, str(raw["rule_type"]), float(raw["threshold"])
                )
            else:
                email_delivery_error = "no_recipient_email_in_session"
        _sb_delete_alert(user_id, aid)
        triggered.append(
            {
                "alert_id": aid,
                "ticker": str(raw["ticker"]),
                "price": price,
                "rule_type": str(raw["rule_type"]),
                "threshold": float(raw["threshold"]),
                "delivered": {"email": delivered_email, "whatsapp": delivered_whatsapp},
                "email_delivery_error": email_delivery_error,
                "rule_deleted": True,
                "evaluated_at": _now_iso(),
            }
        )
    return {"triggered_count": len(triggered), "triggered": triggered}


# --- Public API ---


def list_portfolio(user_id: str) -> dict[str, Any]:
    if _use_supabase():
        return _sb_list_portfolio(user_id)
    return _mem_list_portfolio(user_id)


def add_portfolio_holding(user_id: str, ticker: str, quantity: float, avg_buy_price: float) -> dict[str, Any]:
    if _use_supabase():
        return _sb_add_holding(user_id, ticker, quantity, avg_buy_price)
    return _mem_add_holding(user_id, ticker, quantity, avg_buy_price)


def delete_portfolio_holding(user_id: str, holding_id: str) -> dict[str, Any]:
    if _use_supabase():
        return _sb_delete_holding(user_id, holding_id)
    return _mem_delete_holding(user_id, holding_id)


def list_watchlists(user_id: str) -> dict[str, Any]:
    if _use_supabase():
        return _sb_list_watchlists(user_id)
    return _mem_list_watchlists(user_id)


def upsert_watchlist(user_id: str, name: str, tickers: list[str]) -> dict[str, Any]:
    if _use_supabase():
        return _sb_upsert_watchlist(user_id, name, tickers)
    return _mem_upsert_watchlist(user_id, name, tickers)


def create_alert(
    user_id: str, ticker: str, rule_type: str, threshold: float, channel_email: bool, channel_whatsapp: bool
) -> dict[str, Any]:
    if _use_supabase():
        return _sb_create_alert(user_id, ticker, rule_type, threshold, channel_email, channel_whatsapp)
    return _mem_create_alert(user_id, ticker, rule_type, threshold, channel_email, channel_whatsapp)


def list_alerts(user_id: str) -> dict[str, Any]:
    if _use_supabase():
        return _sb_list_alerts(user_id)
    return _mem_list_alerts(user_id)


def delete_alert(user_id: str, alert_id: str) -> dict[str, Any]:
    if _use_supabase():
        return _sb_delete_alert(user_id, alert_id)
    return _mem_delete_alert(user_id, alert_id)


def delete_all_alerts(user_id: str) -> dict[str, Any]:
    """Remove all pending one-shot rules for this user."""
    if _use_supabase():
        return _sb_delete_all_alerts(user_id)
    n = len(_mem_alerts[user_id])
    _mem_alerts[user_id].clear()
    return {"ok": True, "cleared_count": n}


def evaluate_alerts(user_id: str, user_email: str | None, only_alert_id: str | None = None) -> dict[str, Any]:
    if _use_supabase():
        return _sb_evaluate(user_id, user_email, only_alert_id)
    return _mem_evaluate(user_id, user_email, only_alert_id)
