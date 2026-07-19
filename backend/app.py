from __future__ import annotations

from typing import Any, Literal
import os
import logging
import json
import requests
from pathlib import Path
import time
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

from backend.insights_service import (
    build_explainability,
    compute_insights,
    get_chart_data,
    get_fundamentals_data,
    get_news_data,
    get_ticker_options,
    get_chatbot_reply,
)
from backend.report_pdf import build_report_pdf_bytes, report_attachment_filename
from backend.feature_services import (
    get_events,
    issue_guest_token,
    run_backtest,
    sync_pull,
    sync_push,
)
from backend.supabase_jwt import verify_access_token
from backend.user_data import (
    add_portfolio_holding,
    delete_portfolio_holding,
    list_portfolio,
    list_watchlists,
    upsert_watchlist,
)
from stock_insights import DATA_DIR, MODEL_DIR, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

_FRONTEND_DIR = (os.getenv("FRONTEND_DIR") or "frontend-2").strip() or "frontend-2"
_FRONTEND_ROOT = REPO_ROOT / _FRONTEND_DIR
_FRONTEND_DIST = _FRONTEND_ROOT / "dist"
_FRONTEND_WEB_ROOT = _FRONTEND_DIST if (_FRONTEND_DIST / "index.html").exists() else _FRONTEND_ROOT

# Browsers often cache index.html; stale HTML keeps loading old hashed JS after deploys.
_SPA_INDEX_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}


def _spa_index_response() -> FileResponse | None:
    path = _FRONTEND_WEB_ROOT / "index.html"
    if not path.exists():
        return None
    return FileResponse(path, headers=dict(_SPA_INDEX_CACHE_HEADERS))


def _parse_cors_allow_origins() -> list[str]:
    raw = (os.getenv("CORS_ALLOW_ORIGINS") or "").strip()
    if not raw:
        return [
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ]
    return [o.strip() for o in raw.split(",") if o.strip()]


def _parse_csv_env_list(key: str, default: list[str]) -> list[str]:
    raw = (os.getenv(key) or "").strip()
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_production() -> bool:
    return (os.getenv("APP_ENV") or "development").strip().lower() == "production"


def _env_flag(key: str, default: bool = False) -> bool:
    raw = (os.getenv(key) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = (os.getenv(key) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = {}
_LOG = logging.getLogger("ainvestify.api")
_INSIGHTS_RESP_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_INSIGHTS_INFLIGHT: dict[str, tuple[float, Any]] = {}


def _client_ip(req: Request) -> str:
    if req.client and req.client.host:
        return req.client.host
    return "unknown"


def _check_rate_limit(req: Request, route_key: str, limit: int, window_seconds: int) -> None:
    now = time.time()
    key = f"{route_key}:{_client_ip(req)}"
    bucket = _RATE_LIMIT_BUCKETS.setdefault(key, deque())
    cutoff = now - window_seconds
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "code": "rate_limited",
                    "message": f"Too many requests. Limit is {limit} per {window_seconds} seconds.",
                }
            },
        )
    bucket.append(now)


def _insights_debug_enabled() -> bool:
    return _env_flag("DEBUG_INSIGHTS_CALLS", default=False)


def _log_insights_request(request: Request, req: "InsightsRequest", stage: str) -> None:
    if not _insights_debug_enabled():
        return
    referer = (request.headers.get("referer") or "").strip()
    origin = (request.headers.get("origin") or "").strip()
    ua = (request.headers.get("user-agent") or "").strip()
    _LOG.warning(
        "[insights:%s] ip=%s input=%s referer=%s origin=%s ua=%s",
        stage,
        _client_ip(request),
        req.input.strip().upper()[:24],
        referer[:140],
        origin[:140],
        ua[:140],
    )


def _insights_cache_key(req: Request, body: "InsightsRequest") -> str:
    return "|".join(
        [
            _client_ip(req),
            body.input.strip().upper(),
            body.news_source,
            str(body.max_news),
            str(body.top_items),
            f"{body.sentiment_weight:.6f}",
            "1" if body.train_missing else "0",
        ]
    )


def _insights_cache_ttl_seconds() -> int:
    return max(1, _env_int("INSIGHTS_CACHE_TTL_SECONDS", 8))


def _insights_cached_get(cache_key: str) -> dict[str, Any] | None:
    item = _INSIGHTS_RESP_CACHE.get(cache_key)
    if not item:
        return None
    expires_at, data = item
    if expires_at < time.time():
        _INSIGHTS_RESP_CACHE.pop(cache_key, None)
        return None
    return data


def _insights_cached_set(cache_key: str, data: dict[str, Any]) -> dict[str, Any]:
    _INSIGHTS_RESP_CACHE[cache_key] = (time.time() + _insights_cache_ttl_seconds(), data)
    return data


def _std_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


def _auth_bearer_token(request: Request) -> str:
    auth = (request.headers.get("authorization") or "").strip()
    if not auth.lower().startswith("bearer "):
        raise _std_error(401, "unauthorized", "Missing bearer token.")
    return auth.split(" ", 1)[1].strip()


def _require_auth_claims(request: Request) -> dict[str, Any]:
    try:
        return verify_access_token(_auth_bearer_token(request))
    except ValueError as e:
        raise _std_error(401, "unauthorized", str(e) or "Invalid or expired token.") from e


def _require_auth_user(request: Request) -> str:
    claims = _require_auth_claims(request)
    sub = str(claims.get("sub") or "").strip()
    if not sub:
        raise _std_error(401, "unauthorized", "Token subject missing.")
    return sub


def _startup_readiness_checks() -> None:
    required_paths: list[Path] = [
        DATA_DIR / "financials_cleaned.csv",
        DATA_DIR / "stock_tweets.csv",
        DATA_DIR / "stock_yfinance_data.csv",
        MODEL_DIR / "fundamentals_stock_model.joblib",
        MODEL_DIR / "rfr_stockfundamentalsscorer.pkl",
        MODEL_DIR / "stock_score_regression.pkl",
        MODEL_DIR / "sentiment_logreg.pkl",
        MODEL_DIR / "tfidf_vectorizer.pkl",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise RuntimeError("Startup readiness check failed. Missing files: " + ", ".join(missing))


class InsightsRequest(BaseModel):
    input: str = Field(..., description="Stock symbol (e.g., TSLA) or company name (e.g., Tesla, Inc.)")
    news_source: Literal["auto", "dataset", "live"] = "auto"
    max_news: int = Field(10, ge=1, le=50)
    top_items: int = Field(3, ge=1, le=10)
    sentiment_weight: float = Field(0.3, ge=0.0, le=1.0)
    train_missing: bool = False


class ChatRequest(BaseModel):
    message: str = Field(..., description="User chat message")
    selected_symbol: str | None = Field(default=None, description="Currently selected stock ticker")


class PortfolioHoldingRequest(BaseModel):
    ticker: str
    quantity: float = Field(..., gt=0)
    avg_buy_price: float = Field(..., gt=0)


class WatchlistRequest(BaseModel):
    name: str = "default"
    tickers: list[str] = Field(default_factory=list)


class SyncPushRequest(BaseModel):
    token: str
    state: dict[str, Any] = Field(default_factory=dict)


class UnstickEmailRequest(BaseModel):
    email: str
    password: str


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _startup_readiness_checks()
    using = "dist" if (_FRONTEND_DIST / "index.html").exists() else "source tree (run `npm run build` in frontend-2 for production assets)"
    _LOG.warning("Frontend web root: %s (%s)", _FRONTEND_WEB_ROOT, using)
    yield


app = FastAPI(title="AInvestify Stock Insights API", lifespan=lifespan)

if (_FRONTEND_WEB_ROOT / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_WEB_ROOT / "assets")), name="frontend-assets")

if _env_flag("ENABLE_PROXY_HEADERS", default=_is_production()):
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Development default allows any Host so friends can open http://<your-LAN-IP>:8000.
# In production, set ALLOWED_HOSTS to your real hostname(s); never use "*" there.
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=_parse_csv_env_list(
        "ALLOWED_HOSTS",
        ["*"] if not _is_production() else ["localhost"],
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_allow_origins(),
    allow_credentials=_env_flag("CORS_ALLOW_CREDENTIALS", default=True),
    allow_methods=_parse_csv_env_list(
        "CORS_ALLOW_METHODS",
        ["GET", "POST", "PUT", "DELETE", "OPTIONS"] if _is_production() else ["*"],
    ),
    allow_headers=_parse_csv_env_list(
        "CORS_ALLOW_HEADERS",
        ["Content-Type", "Authorization"] if _is_production() else ["*"],
    ),
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> Response:
    spa = _spa_index_response()
    if spa is not None:
        return spa
    return Response(
        content='{"message":"AInvestify backend is running","docs":"/docs","health":"/health","insights_endpoint":"/api/insights"}',
        media_type="application/json",
    )


@app.get("/portfolio")
def portfolio_page() -> Response:
    spa = _spa_index_response()
    if spa is not None:
        return spa
    return Response(status_code=404)


@app.get("/watchlists")
def watchlists_page() -> Response:
    spa = _spa_index_response()
    if spa is not None:
        return spa
    return Response(status_code=404)


@app.get("/app")
def app_dashboard_page() -> Response:
    spa = _spa_index_response()
    if spa is not None:
        return spa
    return Response(status_code=404)


@app.get("/stock/{symbol}")
def stock_insights_spa(symbol: str) -> Response:
    """SPA entry for /stock/TICKER so refresh and deep links serve the React app."""
    spa = _spa_index_response()
    if spa is not None:
        return spa
    return Response(status_code=404)


@app.api_route("/favicon.ico", methods=["GET", "HEAD"])
def favicon() -> Response:
    # Serve branded favicon; fallback to 204 if missing.
    icon_path = _FRONTEND_WEB_ROOT / "favicon.svg"
    if not icon_path.exists():
        icon_path = REPO_ROOT / "frontend" / "favicon.svg"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/svg+xml")
    return Response(status_code=204)


@app.api_route("/favicon.svg", methods=["GET", "HEAD"])
def favicon_svg() -> Response:
    icon_path = _FRONTEND_WEB_ROOT / "favicon.svg"
    if not icon_path.exists():
        icon_path = REPO_ROOT / "frontend" / "favicon.svg"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/svg+xml")
    return Response(status_code=404)


@app.post("/api/insights")
def insights(req: InsightsRequest, request: Request) -> dict:
    _log_insights_request(request, req, "recv")
    cache_key = _insights_cache_key(request, req)
    cached = _insights_cached_get(cache_key)
    if cached is not None:
        _log_insights_request(request, req, "cache_hit")
        return cached
    inflight_item = _INSIGHTS_INFLIGHT.get(cache_key)
    if inflight_item and inflight_item[0] > time.time():
        try:
            _log_insights_request(request, req, "coalesce")
            return inflight_item[1]
        except Exception:
            _INSIGHTS_INFLIGHT.pop(cache_key, None)

    # Rate-limit only work that hits the model / live data path (not cached repeats).
    try:
        _check_rate_limit(
            request,
            "insights",
            _env_int("RATE_LIMIT_INSIGHTS_COUNT", 60),
            _env_int("RATE_LIMIT_INSIGHTS_WINDOW_SECONDS", 60),
        )
    except HTTPException:
        _log_insights_request(request, req, "429")
        raise

    try:
        response = compute_insights(
            req.input,
            train_missing=req.train_missing,
            top_items=req.top_items,
            sentiment_weight=req.sentiment_weight,
            news_source=req.news_source,
            max_news=req.max_news,
        )
        _INSIGHTS_INFLIGHT[cache_key] = (time.time() + 3, response)
        _log_insights_request(request, req, "ok")
        return _insights_cached_set(cache_key, response)
    except ValueError as e:
        raise _std_error(400, "bad_request", str(e))
    except Exception as e:
        raise _std_error(500, "insights_error", str(e))
    finally:
        _INSIGHTS_INFLIGHT.pop(cache_key, None)


@app.get("/chart/{ticker}")
def chart(ticker: str, period: str = "6mo", interval: str = "1d") -> dict:
    try:
        return get_chart_data(ticker=ticker, period=period, interval=interval)
    except ValueError as e:
        raise _std_error(404, "not_found", str(e))
    except Exception as e:
        raise _std_error(500, "chart_error", str(e))


@app.get("/fundamentals/{ticker}")
def fundamentals(ticker: str) -> dict:
    try:
        return get_fundamentals_data(ticker=ticker)
    except ValueError as e:
        raise _std_error(404, "not_found", str(e))
    except Exception as e:
        raise _std_error(500, "fundamentals_error", str(e))


@app.get("/news/{stock_name}")
def news(stock_name: str, max_articles: int = 10) -> dict:
    try:
        return get_news_data(stock_name=stock_name, max_articles=max_articles)
    except ValueError as e:
        raise _std_error(400, "bad_request", str(e))
    except Exception as e:
        raise _std_error(500, "news_error", str(e))


@app.get("/tickers")
def tickers() -> dict:
    try:
        return get_ticker_options()
    except Exception as e:
        raise _std_error(500, "tickers_error", str(e))


@app.post("/chatbot")
def chatbot(req: ChatRequest, request: Request) -> dict:
    _check_rate_limit(
        request,
        "chatbot",
        _env_int("RATE_LIMIT_CHATBOT_COUNT", 30),
        _env_int("RATE_LIMIT_CHATBOT_WINDOW_SECONDS", 60),
    )
    try:
        return get_chatbot_reply(req.message, selected_symbol=req.selected_symbol)
    except Exception as e:
        raise _std_error(500, "chatbot_error", str(e))


@app.get("/api/insights/explain/{ticker}")
def insights_explain(ticker: str) -> dict:
    try:
        insights = compute_insights(ticker, train_missing=False, top_items=3, sentiment_weight=0.3, news_source="auto", max_news=10)
        return {
            "ticker": ticker.upper(),
            "explainability": build_explainability(insights),
            "final_label": insights.get("final", {}).get("label"),
        }
    except Exception as e:
        raise _std_error(500, "explainability_error", str(e))


@app.get("/api/portfolio")
def portfolio_list(request: Request) -> dict:
    uid = _require_auth_user(request)
    try:
        return list_portfolio(uid)
    except Exception as e:
        raise _std_error(500, "portfolio_error", str(e))


@app.post("/api/portfolio")
def portfolio_add(req: PortfolioHoldingRequest, request: Request) -> dict:
    uid = _require_auth_user(request)
    try:
        return add_portfolio_holding(uid, req.ticker, req.quantity, req.avg_buy_price)
    except Exception as e:
        raise _std_error(400, "portfolio_add_error", str(e))


@app.delete("/api/portfolio/{holding_id}")
def portfolio_delete(holding_id: str, request: Request) -> dict:
    uid = _require_auth_user(request)
    return delete_portfolio_holding(uid, holding_id)


@app.get("/api/watchlists")
def watchlists(request: Request) -> dict:
    uid = _require_auth_user(request)
    return list_watchlists(uid)


@app.post("/api/watchlists")
def watchlists_upsert(req: WatchlistRequest, request: Request) -> dict:
    uid = _require_auth_user(request)
    return upsert_watchlist(uid, req.name, req.tickers)


@app.get("/api/events/{ticker}")
def events(ticker: str) -> dict:
    return get_events(ticker)


@app.get("/api/backtest/{ticker}")
def backtest(ticker: str, period: str = "1y") -> dict:
    try:
        return run_backtest(ticker, lookback_period=period)
    except Exception as e:
        raise _std_error(500, "backtest_error", str(e))


@app.post("/api/report/export")
def report_export(payload: dict, request: Request) -> Response:
    _require_auth_user(request)
    try:
        pdf_bytes = build_report_pdf_bytes(payload)
        fname = report_attachment_filename(payload)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )
    except Exception as e:
        raise _std_error(500, "report_export_error", str(e))


def _supabase_admin_headers() -> dict[str, str]:
    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    service_key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not service_key:
        raise _std_error(400, "supabase_admin_not_configured", "SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY are required.")
    return {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
    }


def _supabase_url() -> str:
    return (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")


def _supabase_find_user_by_email(email: str) -> dict[str, Any] | None:
    headers = _supabase_admin_headers()
    resp = requests.get(f"{_supabase_url()}/auth/v1/admin/users?page=1&per_page=1000", headers=headers, timeout=20)
    if resp.status_code >= 400:
        raise _std_error(500, "supabase_admin_error", f"Failed user lookup ({resp.status_code}).")
    data = resp.json()
    users = data.get("users") if isinstance(data, dict) else data
    if not isinstance(users, list):
        return None
    e = email.strip().lower()
    for u in users:
        if isinstance(u, dict) and str(u.get("email") or "").strip().lower() == e:
            return u
    return None


def _supabase_anon_key() -> str:
    return (os.getenv("SUPABASE_ANON_KEY") or "").strip()


def _password_grant_outcome(email: str, password: str) -> str:
    """Classify password grant: ok | email_not_confirmed | wrong_password | no_anon | error."""
    anon = _supabase_anon_key()
    if not anon:
        return "no_anon"
    resp = requests.post(
        f"{_supabase_url()}/auth/v1/token?grant_type=password",
        headers={
            "apikey": anon,
            "Authorization": f"Bearer {anon}",
            "Content-Type": "application/json",
        },
        json={"email": email, "password": password},
        timeout=20,
    )
    if resp.status_code == 200:
        return "ok"
    try:
        data = resp.json()
    except Exception:
        return "error"
    err_code = str(data.get("error_code") or "").lower()
    msg = str(data.get("msg") or data.get("error_description") or data.get("message") or "").lower()
    blob = f"{err_code} {msg} {data}".lower()
    if "email_not_confirmed" in blob or "email not confirmed" in msg:
        return "email_not_confirmed"
    if "invalid_grant" in blob or "invalid login credentials" in msg or err_code in {"invalid_credentials"}:
        return "wrong_password"
    return "error"


def _supabase_admin_confirm_user_email(user_id: str) -> None:
    headers = _supabase_admin_headers()
    resp = requests.put(
        f"{_supabase_url()}/auth/v1/admin/users/{user_id}",
        headers=headers,
        json={"email_confirm": True},
        timeout=20,
    )
    if resp.status_code >= 400:
        _LOG.warning("admin email_confirm failed: %s %s", resp.status_code, resp.text[:400])
        raise _std_error(500, "confirm_failed", "Could not confirm email. Try again.")


@app.post("/api/auth/unstick-email-confirm")
def unstick_email_confirm(req: UnstickEmailRequest, request: Request) -> dict:
    """For accounts created while 'confirm email' was on: verify password, then mark email confirmed (admin)."""
    _check_rate_limit(
        request,
        "unstick_email_confirm",
        _env_int("RATE_LIMIT_UNSTICK_EMAIL_COUNT", 8),
        _env_int("RATE_LIMIT_UNSTICK_EMAIL_WINDOW_SECONDS", 3600),
    )
    email = (req.email or "").strip().lower()
    password = (req.password or "").strip()
    if "@" not in email or "." not in email:
        raise _std_error(400, "invalid_email", "Please enter a valid email.")
    out = _password_grant_outcome(email, password)
    if out == "no_anon":
        raise _std_error(
            500,
            "server_misconfigured",
            "Set SUPABASE_ANON_KEY on the server to the project anon/publishable key (same as VITE_SUPABASE_ANON_KEY).",
        )
    if out == "wrong_password":
        raise _std_error(401, "invalid_credentials", "Invalid email or password.")
    if out == "ok":
        return {"ok": True, "already_confirmed": True}
    if out != "email_not_confirmed":
        raise _std_error(400, "auth_error", "Could not verify account. Try again.")
    user = _supabase_find_user_by_email(email)
    if not user:
        raise _std_error(400, "user_not_found", "No account found for this email.")
    uid = str(user.get("id") or "").strip()
    if not uid:
        raise _std_error(500, "user_lookup_error", "Could not resolve user id.")
    _supabase_admin_confirm_user_email(uid)
    return {"ok": True, "fixed": True}


@app.post("/api/auth/guest")
def auth_guest() -> dict:
    return issue_guest_token()


@app.post("/api/sync/push")
def sync_state_push(req: SyncPushRequest, request: Request) -> dict:
    _require_auth_user(request)
    return sync_push(req.token, req.state)


@app.get("/api/sync/pull/{token}")
def sync_state_pull(token: str, request: Request) -> dict:
    _require_auth_user(request)
    return sync_pull(token)


@app.get("/chatbot/status")
def chatbot_status() -> dict:
    key_present = bool((os.getenv("GEMINI_API_KEY") or "").strip())
    return {
        "gemini_configured": key_present,
        "provider_when_configured": "gemini",
        "provider_fallback": "local",
    }

