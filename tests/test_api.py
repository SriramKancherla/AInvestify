from __future__ import annotations

from fastapi.testclient import TestClient

import backend.app as app_module


def _make_client(monkeypatch) -> TestClient:
    # Avoid filesystem-dependent startup checks in unit tests.
    monkeypatch.setattr(app_module, "_startup_readiness_checks", lambda: None)
    return TestClient(app_module.app, base_url="http://localhost")


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

    monkeypatch.setenv("RATE_LIMIT_INSIGHTS_COUNT", "1")
    monkeypatch.setenv("RATE_LIMIT_INSIGHTS_WINDOW_SECONDS", "60")
    monkeypatch.setattr(
        app_module,
        "compute_insights",
        lambda *args, **kwargs: {"input": "AAPL", "final": {"score": 0.5, "label": "Mixed / Uncertain"}},
    )

    first = client.post("/api/insights", json={"input": "AAPL"})
    second = client.post("/api/insights", json={"input": "AAPL"})

    assert first.status_code == 200
    assert second.status_code == 429
    detail = second.json()["detail"]["error"]
    assert detail["code"] == "rate_limited"


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
