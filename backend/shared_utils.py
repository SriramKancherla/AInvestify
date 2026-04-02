from __future__ import annotations

import time
from threading import Lock
from typing import Any, Callable

import requests


_CACHE_LOCK = Lock()
_TTL_CACHE: dict[str, tuple[float, Any]] = {}


def ttl_get(key: str) -> Any | None:
    with _CACHE_LOCK:
        item = _TTL_CACHE.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at < time.time():
            _TTL_CACHE.pop(key, None)
            return None
        return value


def ttl_set(key: str, value: Any, ttl_seconds: int) -> Any:
    with _CACHE_LOCK:
        _TTL_CACHE[key] = (time.time() + max(1, ttl_seconds), value)
    return value


def cached_call(key: str, ttl_seconds: int, fn: Callable[[], Any]) -> Any:
    cached = ttl_get(key)
    if cached is not None:
        return cached
    return ttl_set(key, fn(), ttl_seconds)


def request_with_retry(
    method: str,
    url: str,
    *,
    json: dict | None = None,
    timeout: int = 12,
    retries: int = 2,
    backoff_ms: int = 300,
):
    last_error: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            return requests.request(method=method, url=url, json=json, timeout=timeout)
        except Exception as e:
            last_error = e
            if attempt < retries and backoff_ms > 0:
                time.sleep((backoff_ms / 1000.0) * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("request_failed_unknown")
