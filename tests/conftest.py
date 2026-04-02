from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _auth_jwt_insecure_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real Supabase ES256 verification requires live JWKS; tests use unsigned fixtures."""
    monkeypatch.setenv("AUTH_JWT_INSECURE_SKIP_VERIFY", "1")
