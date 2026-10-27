from __future__ import annotations

from fastapi.testclient import TestClient

import backend.app as app_module


def _make_client(monkeypatch) -> TestClient:
    # Avoid filesystem-dependent startup checks in unit tests.
    monkeypatch.setattr(app_module, "_startup_readiness_checks", lambda: None)
    return TestClient(app_module.app, base_url="http://localhost")


def _auth_headers() -> dict[str, str]:
    # Minimal unsigned JWT-like token. Backend only parses claims in Phase 1.
    token = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJ0ZXN0LXVzZXIiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjo0MTAyNDQ0ODAwfQ."
    return {"Authorization": f"Bearer {token}"}


def test_insights_success_shape(monkeypatch) -> None:
    client = _make_client(monkeypatch)

    monkeypatch.setattr(
        app_module,
        "compute_insights",
        lambda *args, **kwargs: {
            "final": {"score": 0.61, "label": "Mildly Bullish"},
            "fundamentals": {"score": 0.58, "source": "dataset", "meta": {"source": "dataset"}},
            "sentiment": {"score": 0.67, "source": "live_news", "item_count": 5},
            "meta": {"news_source": "auto"},
            "resolved": {"fundamentals": {"symbol": "AAPL"}, "sentiment": {"symbol": None}},
            "input": "AAPL",
        },
    )

    resp = client.post(
        "/api/insights",
        json={
            "input": "AAPL",
            "news_source": "auto",
            "max_news": 10,
            "top_items": 3,
            "sentiment_weight": 0.3,
            "train_missing": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "final" in body
    assert "fundamentals" in body
    assert "sentiment" in body


def test_chart_unavailable_returns_404_but_insights_can_succeed(monkeypatch) -> None:
    client = _make_client(monkeypatch)

    monkeypatch.setattr(
        app_module,
        "compute_insights",
        lambda *args, **kwargs: {
            "final": {"score": 0.55, "label": "Mixed / Uncertain"},
            "fundamentals": {"score": 0.54},
            "sentiment": {"score": 0.58},
            "meta": {},
            "resolved": {"fundamentals": {"symbol": "ALGN"}, "sentiment": {"symbol": None}},
            "input": "ALGN",
        },
    )
    monkeypatch.setattr(
        app_module,
        "get_chart_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("No chart data found")),
    )

    insights_resp = client.post("/api/insights", json={"input": "ALGN"})
    chart_resp = client.get("/chart/ALGN")

    assert insights_resp.status_code == 200
    assert chart_resp.status_code == 404
    chart_body = chart_resp.json()
    assert chart_body["detail"]["error"]["code"] == "not_found"


def test_chatbot_local_fallback_on_quota_is_returned(monkeypatch) -> None:
    client = _make_client(monkeypatch)

    monkeypatch.setattr(
        app_module,
        "get_chatbot_reply",
        lambda *args, **kwargs: {
            "reply": "Using local fallback.",
            "provider": "local",
            "fallback_reason": "http_429:quota exceeded",
            "context_symbol": "AAPL",
        },
    )

    resp = client.post("/chatbot", json={"message": "summary about apple", "selected_symbol": "AAPL"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "local"
    assert "http_429" in body["fallback_reason"]


def test_rate_limit_429_on_insights(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    app_module._RATE_LIMIT_BUCKETS.clear()
    app_module._INSIGHTS_RESP_CACHE.clear()
    app_module._INSIGHTS_INFLIGHT.clear()

    monkeypatch.setenv("RATE_LIMIT_INSIGHTS_COUNT", "1")
    monkeypatch.setenv("RATE_LIMIT_INSIGHTS_WINDOW_SECONDS", "60")
    monkeypatch.setattr(
        app_module,
        "compute_insights",
        lambda *args, **kwargs: {
            "input": (args[0] if args else kwargs.get("input_value", "")),
            "final": {"score": 0.5, "label": "Mixed / Uncertain"},
        },
    )

    first = client.post("/api/insights", json={"input": "AAPL"})
    # Same ticker again is served from cache and does not consume another rate-limit slot.
    cached_repeat = client.post("/api/insights", json={"input": "AAPL"})
    second = client.post("/api/insights", json={"input": "MSFT"})

    assert first.status_code == 200
    assert cached_repeat.status_code == 200
    assert second.status_code == 429
    detail = second.json()["detail"]["error"]
    assert detail["code"] == "rate_limited"


def test_portfolio_open_without_auth(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    monkeypatch.setattr(app_module, "list_portfolio", lambda _uid: {"holdings": [], "summary": {}})
    resp = client.get("/api/portfolio")
    assert resp.status_code == 200
    assert "holdings" in resp.json()


def test_portfolio_with_auth_header_still_ok(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    monkeypatch.setattr(app_module, "list_portfolio", lambda _uid: {"holdings": [], "summary": {}})
    resp = client.get("/api/portfolio", headers=_auth_headers())
    assert resp.status_code == 200
    assert "holdings" in resp.json()


def test_sync_push_open_without_auth(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    monkeypatch.setattr(
        app_module, "sync_push", lambda token, state: {"ok": True, "token": token}
    )
    resp = client.post("/api/sync/push", json={"token": "sync-token", "state": {}})
    assert resp.status_code == 200
    assert resp.json().get("ok") is True


def test_sync_pull_open_without_auth(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    monkeypatch.setattr(app_module, "sync_pull", lambda token: {"token": token, "state": {}})
    resp = client.get("/api/sync/pull/sync-token")
    assert resp.status_code == 200
    assert resp.json().get("token") == "sync-token"


def test_sync_push_with_auth_header(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    monkeypatch.setattr(
        app_module, "sync_push", lambda token, state: {"ok": True, "token": token}
    )
    resp = client.post(
        "/api/sync/push",
        json={"token": "sync-token", "state": {"watchlists": []}},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_sync_pull_with_auth_header(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    monkeypatch.setattr(
        app_module, "sync_pull", lambda token: {"token": token, "state": {}}
    )
    resp = client.get("/api/sync/pull/sync-token", headers=_auth_headers())
    assert resp.status_code == 200
    assert resp.json()["token"] == "sync-token"


def test_built_frontend_serves_models_page_and_logos(monkeypatch, tmp_path) -> None:
    dist = tmp_path / "dist"
    (dist / "logos").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>AInvestify</title>", encoding="utf-8")
    (dist / "logos" / "AAPL.png").write_bytes(b"png")
    monkeypatch.setattr(app_module, "_FRONTEND_DIST", dist)
    monkeypatch.setattr(app_module, "_FRONTEND_WEB_ROOT", dist)
    client = _make_client(monkeypatch)

    models = client.get("/models")
    logo = client.get("/logos/AAPL.png")
    missing_api = client.get("/api/not-a-route")

    assert models.status_code == 200
    assert "AInvestify" in models.text
    assert logo.status_code == 200
    assert logo.content == b"png"
    assert missing_api.status_code == 404
    assert b"<html" not in missing_api.content.lower()
    assert app_module._built_frontend_file("../requirements.txt") is None


def test_forecast_status_endpoint(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    resp = client.get("/api/forecast/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "chart_forecast_diffusion"
    assert "horizons_days" in body


def test_forecast_chart_unavailable(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    monkeypatch.setattr(
        app_module,
        "get_chart_forecast_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Chart forecast not ready")),
    )
    resp = client.get("/api/forecast/chart/AAPL?horizon_days=63")
    assert resp.status_code == 503


def test_startup_readiness_failure(monkeypatch) -> None:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def failing_lifespan(_app):
        raise RuntimeError("missing files for startup")
        yield

    failing_app = app_module.FastAPI(lifespan=failing_lifespan)
    try:
        with TestClient(failing_app):
            assert False, "Expected startup readiness to fail"
    except RuntimeError as e:
        assert "missing files for startup" in str(e)
