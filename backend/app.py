from __future__ import annotations

from typing import Literal
import os
from pathlib import Path
import time
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.insights_service import (
    compute_insights,
    get_chart_data,
    get_fundamentals_data,
    get_news_data,
    get_ticker_options,
    get_chatbot_reply,
)
from stock_insights import DATA_DIR, MODEL_DIR, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")


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
        MODEL_DIR / "keras_stockfundamentalsscorer.h5",
        MODEL_DIR / "keras_X_scaler.pkl",
        MODEL_DIR / "keras_Y_scaler.pkl",
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _startup_readiness_checks()
    yield


app = FastAPI(title="AInvestify Stock Insights API", lifespan=lifespan)

if _env_flag("ENABLE_PROXY_HEADERS", default=_is_production()):
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=_parse_csv_env_list(
        "ALLOWED_HOSTS",
        ["127.0.0.1", "localhost", "*.trycloudflare.com"]
        if not _is_production()
        else ["localhost"],
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_allow_origins(),
    allow_credentials=_env_flag("CORS_ALLOW_CREDENTIALS", default=True),
    allow_methods=_parse_csv_env_list(
        "CORS_ALLOW_METHODS",
        ["GET", "POST", "OPTIONS"] if _is_production() else ["*"],
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
    frontend_index = REPO_ROOT / "frontend" / "index.html"
    if frontend_index.exists():
        return FileResponse(frontend_index)
    return Response(
        content='{"message":"AInvestify backend is running","docs":"/docs","health":"/health","insights_endpoint":"/api/insights"}',
        media_type="application/json",
    )


@app.api_route("/favicon.ico", methods=["GET", "HEAD"])
def favicon() -> Response:
    # Serve branded favicon; fallback to 204 if missing.
    icon_path = REPO_ROOT / "frontend" / "favicon.svg"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/svg+xml")
    return Response(status_code=204)


@app.api_route("/favicon.svg", methods=["GET", "HEAD"])
def favicon_svg() -> Response:
    icon_path = REPO_ROOT / "frontend" / "favicon.svg"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/svg+xml")
    return Response(status_code=404)


@app.post("/api/insights")
def insights(req: InsightsRequest, request: Request) -> dict:
    _check_rate_limit(
        request,
        "insights",
        _env_int("RATE_LIMIT_INSIGHTS_COUNT", 60),
        _env_int("RATE_LIMIT_INSIGHTS_WINDOW_SECONDS", 60),
    )
    try:
        return compute_insights(
            req.input,
            train_missing=req.train_missing,
            top_items=req.top_items,
            sentiment_weight=req.sentiment_weight,
            news_source=req.news_source,
            max_news=req.max_news,
        )
    except ValueError as e:
        raise _std_error(400, "bad_request", str(e))
    except Exception as e:
        raise _std_error(500, "insights_error", str(e))


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


@app.get("/chatbot/status")
def chatbot_status() -> dict:
    key_present = bool((os.getenv("GEMINI_API_KEY") or "").strip())
    return {
        "gemini_configured": key_present,
        "provider_when_configured": "gemini",
        "provider_fallback": "local",
    }

