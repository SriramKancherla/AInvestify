from __future__ import annotations

from typing import Literal
import re
import os
from difflib import SequenceMatcher
import time

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from Model_gens.webscrapernews import scrape_google_news_rss
from backend.chat_style_service import rewrite_chat_reply
from backend.shared_utils import cached_call, request_with_retry
from stock_insights import DATA_DIR, MODEL_DIR, get_insights


NewsSource = Literal["auto", "dataset", "live"]
_SENTIMENT_CACHE: dict | None = None
_TICKER_UNIVERSE_CACHE: set[str] | None = None
_GEMINI_MODELS = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-2.0-flash"]


def _env_int(key: str, default: int) -> int:
    raw = (os.getenv(key) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _request_with_retry(method: str, url: str, *, json: dict | None = None, timeout: int = 12):
    retries = max(0, _env_int("HTTP_RETRY_COUNT", 2))
    backoff_ms = max(0, _env_int("HTTP_RETRY_BACKOFF_MS", 300))
    return request_with_retry(method, url, json=json, timeout=timeout, retries=retries, backoff_ms=backoff_ms)


def _load_sentiment_assets() -> dict:
    global _SENTIMENT_CACHE
    if _SENTIMENT_CACHE is None:
        _SENTIMENT_CACHE = {
            "model": joblib.load(MODEL_DIR / "sentiment_logreg.pkl"),
            "vectorizer": joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl"),
            "vader": SentimentIntensityAnalyzer(),
        }
    return _SENTIMENT_CACHE


def _load_ticker_universe() -> set[str]:
    global _TICKER_UNIVERSE_CACHE
    if _TICKER_UNIVERSE_CACHE is None:
        fin = pd.read_csv(DATA_DIR / "financials_cleaned.csv")
        tw = pd.read_csv(DATA_DIR / "stock_tweets.csv", sep="\t", engine="python", on_bad_lines="skip")
        fin_set = set(fin["Symbol"].astype(str).str.upper().str.strip().tolist())
        tw_set = set(tw["Stock Name"].astype(str).str.upper().str.strip().tolist())
        _TICKER_UNIVERSE_CACHE = {t for t in (fin_set | tw_set) if t and t != "NAN"}
    return _TICKER_UNIVERSE_CACHE


def _infer_symbol_from_message(message: str, selected_symbol: str | None) -> tuple[str | None, str]:
    """
    Infer symbol using:
    1) explicit ticker tokens in message
    2) company-name match from financials dataset
    3) selected symbol fallback
    """
    text = (message or "").strip()
    q = text.lower()
    ticker_universe = _load_ticker_universe()

    # 1) Explicit ticker token in message wins.
    tokens = re.findall(r"\b[A-Z]{1,5}\b", text.upper())
    explicit = [t for t in tokens if t in ticker_universe]
    if explicit:
        return explicit[0], "message_ticker"

    # 2) Try company name lookup from financials.
    try:
        fin = pd.read_csv(DATA_DIR / "financials_cleaned.csv")
        best_symbol = None
        best_len = -1
        for _, row in fin.iterrows():
            name = str(row.get("Name", "")).strip().lower()
            symbol = str(row.get("Symbol", "")).strip().upper()
            if not name or not symbol:
                continue
            if name in q and len(name) > best_len:
                best_symbol = symbol
                best_len = len(name)
        if best_symbol:
            return best_symbol, "message_company_name"

        # 2b) Partial company-name match (e.g., "apple" -> "Apple Inc.").
        query_words = [w for w in re.findall(r"[a-z]{3,}", q) if w not in {"stock", "summary", "about", "analysis", "sentiment", "fundamentals", "news"}]
        partial_best_symbol = None
        partial_best_score = -1.0
        for _, row in fin.iterrows():
            name = str(row.get("Name", "")).strip().lower()
            symbol = str(row.get("Symbol", "")).strip().upper()
            if not name or not symbol:
                continue
            name_words = set(re.findall(r"[a-z]{3,}", name))
            if not name_words:
                continue

            overlap = sum(1 for w in query_words if w in name_words)
            if overlap > 0:
                # Prefer higher overlap, then closer whole-string similarity.
                sim = SequenceMatcher(None, q, name).ratio()
                score = overlap + sim
                if score > partial_best_score:
                    partial_best_score = score
                    partial_best_symbol = symbol

        if partial_best_symbol is not None:
            return partial_best_symbol, "message_company_partial"
    except Exception:
        pass

    # 3) Selected dropdown stock fallback.
    sel = (selected_symbol or "").strip().upper()
    if sel and sel in ticker_universe:
        return sel, "selected_symbol"

    return None, "none"


def compute_insights(
    input_value: str,
    *,
    train_missing: bool = False,
    top_items: int = 3,
    sentiment_weight: float = 0.3,
    news_source: NewsSource = "auto",
    max_news: int = 10,
) -> dict:
    """
    Thin service wrapper around `stock_insights.get_insights()`.

    Kept separate so the FastAPI layer stays small.
    """
    cache_key = f"insights:{input_value}:{train_missing}:{top_items}:{sentiment_weight}:{news_source}:{max_news}"
    ttl_seconds = max(5, _env_int("INSIGHTS_CACHE_TTL_SECONDS", 45))
    return cached_call(
        cache_key,
        ttl_seconds,
        lambda: get_insights(
            input_value,
            train_missing=train_missing,
            top_items=top_items,
            sentiment_weight=sentiment_weight,
            news_source=news_source,
            max_news=max_news,
        ),
    )


def build_explainability(insights: dict) -> dict:
    final_score = insights.get("final", {}).get("score")
    f_score = insights.get("fundamentals", {}).get("score")
    s_score = insights.get("sentiment", {}).get("score")
    sentiment_weight = float(insights.get("meta", {}).get("sentiment_weight", 0.3) or 0.3)
    fundamentals_weight = 1.0 - sentiment_weight
    drivers = [
        {
            "name": "Fundamentals Model",
            "score": f_score,
            "weight": round(fundamentals_weight, 3),
            "contribution": None if f_score is None else round(float(f_score) * fundamentals_weight, 4),
        },
        {
            "name": "News Sentiment Model",
            "score": s_score,
            "weight": round(sentiment_weight, 3),
            "contribution": None if s_score is None else round(float(s_score) * sentiment_weight, 4),
        },
    ]
    confidence = insights.get("final", {}).get("confidence")
    return {
        "final_score": final_score,
        "confidence": confidence,
        "drivers": drivers,
        "summary": "Final score is a weighted blend of fundamentals and sentiment signals.",
    }


def get_ticker_options() -> dict:
    df = pd.read_csv(DATA_DIR / "financials_cleaned.csv")
    options = []
    for _, row in df.iterrows():
        symbol = str(row.get("Symbol", "")).upper().strip()
        name = str(row.get("Name", "")).strip()
        if not symbol:
            continue
        options.append({"symbol": symbol, "name": name, "label": f"{symbol} - {name}"})
    options = sorted(options, key=lambda x: x["symbol"])
    return {"count": len(options), "tickers": options}


def get_chart_data(ticker: str, period: str = "6mo", interval: str = "1d") -> dict:
    ticker_up = ticker.upper()
    chart_points = []
    source = "live_yfinance"
    error = None

    try:
        t = yf.Ticker(ticker_up)
        hist = t.history(period=period, interval=interval, auto_adjust=False)
        if hist.empty:
            raise ValueError("empty history from yfinance")

        for idx, row in hist.iterrows():
            chart_points.append(
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                }
            )
    except Exception as e:
        # Fallback to local historical dataset.
        source = "dataset_stock_yfinance_data"
        error = str(e)
        ds = pd.read_csv(DATA_DIR / "stock_yfinance_data.csv")
        sub = ds.loc[ds["Stock Name"].astype(str).str.upper() == ticker_up].copy()
        if not sub.empty:
            sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce")
            sub = sub.dropna(subset=["Date"]).sort_values("Date")
            for _, row in sub.iterrows():
                chart_points.append(
                    {
                        "date": row["Date"].strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]),
                    }
                )
        else:
            # Last-resort synthetic fallback for symbols with no yfinance/dataset history.
            # Keeps UI usable for delisted or sparse symbols from fundamentals universe.
            fin = pd.read_csv(DATA_DIR / "financials_cleaned.csv")
            fin_row = fin.loc[fin["Symbol"].astype(str).str.upper() == ticker_up]
            if fin_row.empty:
                raise ValueError(f"No chart data found for ticker '{ticker_up}'.")
            base_price_raw = fin_row.iloc[0].get("Price")
            try:
                base_price = float(base_price_raw)
            except Exception:
                raise ValueError(f"No chart data found for ticker '{ticker_up}'.")

            source = "fallback_synthetic_from_fundamentals"
            today = pd.Timestamp.utcnow().normalize()
            for i in range(30):
                d = (today - pd.Timedelta(days=(29 - i))).strftime("%Y-%m-%d")
                chart_points.append(
                    {
                        "date": d,
                        "open": base_price,
                        "high": base_price,
                        "low": base_price,
                        "close": base_price,
                        "volume": 0,
                    }
                )

    return {
        "ticker": ticker_up,
        "period": period,
        "interval": interval,
        "source": source,
        "source_error": error,
        "points": chart_points,
    }


def get_fundamentals_data(ticker: str) -> dict:
    # Reuse the unified scoring path, forcing live sentiment off.
    insights = compute_insights(
        ticker,
        train_missing=False,
        top_items=3,
        sentiment_weight=0.3,
        news_source="dataset",
        max_news=10,
    )

    info = {}
    info_error = None
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        info_error = str(e)

    # Dataset fallback to avoid empty fundamentals when live provider misses fields.
    dataset_row = None
    try:
        df = pd.read_csv(DATA_DIR / "financials_cleaned.csv")
        mask = df["Symbol"].astype(str).str.upper().str.strip() == ticker.upper()
        if mask.any():
            dataset_row = df.loc[mask].iloc[0]
    except Exception:
        dataset_row = None

    def _pick(primary, dataset_key: str):
        if primary is not None:
            return primary
        if dataset_row is None:
            return None
        try:
            v = dataset_row.get(dataset_key)
            if pd.isna(v):
                return None
            return float(v)
        except Exception:
            return None

    return {
        "ticker": ticker.upper(),
        "fundamentals_score": insights["fundamentals"]["score"],
        "fundamentals_source": insights["fundamentals"].get("source"),
        "fundamentals_meta": insights["fundamentals"].get("meta"),
        "model_scores": insights["fundamentals"]["model_scores"],
        "metrics": {
            "market_cap": _pick(info.get("marketCap"), "Market_Cap"),
            "current_price": _pick(info.get("currentPrice") or info.get("regularMarketPrice"), "Price"),
            "pe_ratio": _pick(info.get("trailingPE"), "Price/Earnings"),
            "pb_ratio": _pick(info.get("priceToBook"), "Price/Book"),
            "price_to_sales": _pick(info.get("priceToSalesTrailing12Months"), "Price/Sales"),
            "book_value": _pick(info.get("bookValue"), "Book_Value"),
            "ebitda": _pick(info.get("ebitda"), "EBITDA"),
            "dividend_yield": _pick(info.get("dividendYield"), "Dividend_Yield"),
            "fifty_two_week_high": _pick(info.get("fiftyTwoWeekHigh"), "52w_high"),
            "fifty_two_week_low": _pick(info.get("fiftyTwoWeekLow"), "52w_low"),
        },
        "metrics_source_error": info_error,
    }


def get_news_data(stock_name: str, max_articles: int = 10) -> dict:
    assets = _load_sentiment_assets()
    model = assets["model"]
    vectorizer = assets["vectorizer"]
    vader = assets["vader"]

    news_items = scrape_google_news_rss(stock_name, max_articles=max_articles)
    if not news_items:
        return {
            "stock_name": stock_name,
            "article_count": 0,
            "overall_sentiment_score": None,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "articles": [],
        }

    texts = [str(item.get("text_for_model", "")) for item in news_items]
    X = vectorizer.transform(texts)
    proba = model.predict_proba(X)
    if 1 in model.classes_:
        pos_idx = list(model.classes_).index(1)
    else:
        pos_idx = 1 if proba.shape[1] > 1 else 0
    lr_pos_probs = proba[:, pos_idx]

    out_articles = []
    combined_scores: list[float] = []
    pos = neg = neu = 0

    for i, item in enumerate(news_items):
        title = str(item.get("title", "")).strip()
        text_for_model = str(item.get("text_for_model", "")).strip()

        lr_score = float(lr_pos_probs[i])  # 0..1
        vader_compound = float(vader.polarity_scores(text_for_model)["compound"])  # -1..1
        vader_01 = (vader_compound + 1.0) / 2.0

        # Blend classifier confidence with lexicon polarity.
        combined = 0.7 * lr_score + 0.3 * vader_01
        combined_scores.append(combined)

        if combined >= 0.6:
            label = "positive"
            pos += 1
        elif combined <= 0.4:
            label = "negative"
            neg += 1
        else:
            label = "neutral"
            neu += 1

        out_articles.append(
            {
                "headline": title,
                "link": item.get("link"),
                "sentiment_label": label,
                "sentiment_score": round(combined, 4),
                "lr_score": round(lr_score, 4),
                "vader_compound": round(vader_compound, 4),
                "quality": {
                    "credibility_score": round(0.55 + (0.35 if "reuters" in title.lower() or "bloomberg" in title.lower() else 0.0), 3),
                    "impact_tag": "high" if abs(vader_compound) > 0.55 else "medium" if abs(vader_compound) > 0.25 else "low",
                },
            }
        )

    return {
        "stock_name": stock_name,
        "article_count": len(out_articles),
        "overall_sentiment_score": float(np.mean(combined_scores)),
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count": neu,
        "articles": out_articles,
    }


def _gemini_reply_with_context(user_message: str, symbol: str, insights: dict) -> tuple[str | None, str | None]:
    """
    Generate a conversational response using Gemini, grounded in current insights.
    Returns None if API key is missing or if call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None, "missing_api_key"

    prompt = (
        "You are AInvestify's stock assistant. "
        "Use only the provided stock context and avoid financial guarantees. "
        "Be concise (4-7 lines), practical, and easy to understand.\n\n"
        f"Selected symbol: {symbol}\n"
        f"Current insights JSON: {insights}\n\n"
        f"User question: {user_message}\n\n"
        "Respond with: 1) short answer, 2) what fundamentals imply, "
        "3) what sentiment implies, 4) one caution."
    )

    timeout_seconds = _env_int("GEMINI_TIMEOUT_SECONDS", 12)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 300,
        },
    }

    last_error: str | None = None
    for model_name in _GEMINI_MODELS:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            f"?key={api_key}"
        )
        try:
            resp = _request_with_retry(
                "POST",
                endpoint,
                json=payload,
                timeout=timeout_seconds,
            )
            if resp.status_code >= 400:
                body = resp.text[:300]
                last_error = f"http_{resp.status_code}:{body}"
                continue
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                last_error = "no_candidates"
                continue
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                last_error = "no_parts"
                continue
            text = parts[0].get("text", "").strip()
            if not text:
                last_error = "empty_text"
                continue
            return text, None
        except Exception as e:
            last_error = str(e)
            continue

    return None, last_error or "unknown_gemini_error"


def get_chatbot_reply(message: str, selected_symbol: str | None = None) -> dict:
    text = (message or "").strip()
    if not text:
        return {"reply": "Please type a question about a stock.", "context_symbol": selected_symbol}

    q = text.lower()
    symbol, symbol_source = _infer_symbol_from_message(text, selected_symbol)

    wants_summary = any(k in q for k in ["summary", "conclusion", "overall", "analyze", "analysis"])
    wants_fund = any(k in q for k in ["fundamental", "fundamentals", "balance", "valuation", "pe", "p/e"])
    wants_sent = any(k in q for k in ["sentiment", "news", "bullish", "bearish", "headlines"])
    wants_compare = any(k in q for k in ["compare", "versus", "vs"])

    def _finalize(payload: dict, raw_reply: str, *, provider: str, fallback_reason: str | None = None) -> dict:
        rewritten, rewrite_provider = rewrite_chat_reply(
            user_message=text,
            base_reply=raw_reply,
            context_symbol=payload.get("context_symbol"),
        )
        payload["reply"] = rewritten
        payload["provider"] = provider
        payload["style_rewrite_provider"] = rewrite_provider
        if fallback_reason:
            payload["fallback_reason"] = fallback_reason
        return payload

    if wants_compare:
        base = {
            "context_symbol": symbol,
            "context_symbol_source": symbol_source,
        }
        return _finalize(
            base,
            "Use Compare mode in the UI, pick two stocks, then ask me to interpret the difference in fundamentals and sentiment.",
            provider="local",
        )

    if symbol is None:
        base = {
            "context_symbol": None,
            "context_symbol_source": symbol_source,
        }
        return _finalize(
            base,
            "Tell me a ticker symbol (for example: TSLA or AAPL), or select one in the dropdown first.",
            provider="local",
        )

    try:
        insights = compute_insights(
            symbol,
            news_source="auto",
            sentiment_weight=0.3,
            top_items=3,
            max_news=10,
            train_missing=False,
        )
    except Exception as e:
        return {"reply": f"I could not fetch insights right now: {e}", "context_symbol": symbol}

    # Prefer Gemini if key exists; fallback to deterministic response if unavailable.
    ai_reply, gemini_error = _gemini_reply_with_context(text, symbol, insights)
    if ai_reply:
        base = {
            "context_symbol": symbol,
            "context_symbol_source": symbol_source,
        }
        return _finalize(base, ai_reply, provider="gemini")

    final_score = insights["final"]["score"]
    final_label = insights["final"]["label"]
    fund = insights["fundamentals"]["score"]
    sent = insights["sentiment"]["score"]

    if wants_fund:
        if fund is None:
            reply = f"For {symbol}, fundamentals are currently unavailable."
        else:
            reply = f"For {symbol}, fundamentals score is {fund:.3f}, which supports a {('strong' if fund >= 0.65 else 'moderate' if fund >= 0.45 else 'weak')} fundamentals view."
        base = {
            "context_symbol": symbol,
            "context_symbol_source": symbol_source,
        }
        return _finalize(base, reply, provider="local", fallback_reason=gemini_error)

    if wants_sent:
        if sent is None:
            reply = f"For {symbol}, sentiment is currently unavailable."
        else:
            tone = "bullish" if sent >= 0.6 else "neutral" if sent >= 0.4 else "bearish"
            reply = f"For {symbol}, sentiment score is {sent:.3f} ({tone})."
        base = {
            "context_symbol": symbol,
            "context_symbol_source": symbol_source,
        }
        return _finalize(base, reply, provider="local", fallback_reason=gemini_error)

    if wants_summary or True:
        fund_text = "N/A" if fund is None else f"{fund:.3f}"
        sent_text = "N/A" if sent is None else f"{sent:.3f}"
        reply = (
            f"{symbol} summary: final score {final_score:.3f} ({final_label}), "
            f"fundamentals {fund_text}, sentiment {sent_text}. "
            "Use Compare mode to check this against another stock."
        )
        base = {
            "context_symbol": symbol,
            "context_symbol_source": symbol_source,
        }
        return _finalize(base, reply, provider="local", fallback_reason=gemini_error)

