from __future__ import annotations

import os
import time
from typing import Optional

import requests

_GEMINI_MODELS = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-2.0-flash"]


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


def _request_with_retry(method: str, url: str, *, json: dict | None = None, timeout: int = 12):
    retries = max(0, _env_int("HTTP_RETRY_COUNT", 2))
    backoff_ms = max(0, _env_int("HTTP_RETRY_BACKOFF_MS", 300))
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return requests.request(method=method, url=url, json=json, timeout=timeout)
        except Exception as e:
            last_error = e
            if attempt < retries and backoff_ms > 0:
                time.sleep((backoff_ms / 1000.0) * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("request_failed_unknown")


def rewrite_chat_reply(
    *,
    user_message: str,
    base_reply: str,
    context_symbol: Optional[str],
) -> tuple[str, str]:
    """
    Style-only rewrite stage:
    - preserves facts from base_reply
    - improves tone and flow
    Returns (reply, provider_tag)
    """
    if not _env_flag("CHAT_STYLE_REWRITE", default=False):
        return base_reply, "off"

    provider = (os.getenv("CHAT_STYLE_PROVIDER") or "gemini").strip().lower()
    if provider not in {"gemini"}:
        return base_reply, f"unsupported:{provider}"

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return base_reply, "missing_api_key"

    timeout_seconds = max(4, _env_int("GEMINI_TIMEOUT_SECONDS", 12))
    prompt = (
        "You are a style rewriter for a stock assistant.\n"
        "Rewrite the assistant response to sound natural and conversational.\n"
        "Do not change facts, numbers, tickers, labels, or recommendations.\n"
        "Keep it concise (3-6 lines), clear, and friendly.\n"
        "No markdown bullets unless needed.\n\n"
        f"Selected symbol: {context_symbol or 'N/A'}\n"
        f"User message: {user_message}\n"
        f"Base assistant reply (facts source of truth): {base_reply}\n\n"
        "Return only the rewritten reply."
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 220},
    }

    for model_name in _GEMINI_MODELS:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            f"?key={api_key}"
        )
        try:
            resp = _request_with_retry("POST", endpoint, json=payload, timeout=timeout_seconds)
            if resp.status_code >= 400:
                continue
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                continue
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                continue
            text = str(parts[0].get("text", "")).strip()
            if text:
                return text, f"gemini_style:{model_name}"
        except Exception:
            continue

    return base_reply, "rewrite_failed"
