from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any

import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWTError

_LOG = logging.getLogger("ainvestify.supabase_jwt")

_jwks_client: PyJWKClient | None = None


def _env_flag(key: str, default: bool = False) -> bool:
    raw = (os.getenv(key) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _supabase_url() -> str:
    return (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")


def _issuer() -> str:
    custom = (os.getenv("SUPABASE_JWT_ISSUER") or "").strip().rstrip("/")
    if custom:
        return custom
    return f"{_supabase_url()}/auth/v1"


def _insecure_parse_payload(token: str) -> dict[str, Any]:
    """Parse JWT payload without cryptography (tests / emergency only)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid bearer token.")
    pad = "=" * (-len(parts[1]) % 4)
    raw = base64.urlsafe_b64decode(parts[1] + pad)
    return json.loads(raw.decode("utf-8"))


def verify_access_token(token: str) -> dict[str, Any]:
    """
    Validate Supabase access token (ES256 + JWKS). Set AUTH_JWT_INSECURE_SKIP_VERIFY=1 only in tests.
    """
    if _env_flag("AUTH_JWT_INSECURE_SKIP_VERIFY"):
        payload = _insecure_parse_payload(token)
        sub = str(payload.get("sub") or "").strip()
        if not sub:
            raise ValueError("Token subject missing.")
        exp = payload.get("exp")
        if isinstance(exp, (int, float)) and time.time() >= float(exp):
            raise ValueError("Token expired.")
        return payload

    url = _supabase_url()
    if not url:
        raise ValueError("SUPABASE_URL is not configured.")

    global _jwks_client
    if _jwks_client is None:
        jwks_url = f"{url}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(jwks_url, cache_keys=True)

    signing_key = _jwks_client.get_signing_key_from_jwt(token)
    try:
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience="authenticated",
            issuer=_issuer(),
        )
    except PyJWTError as e:
        raise ValueError("Invalid or expired token.") from e
    return dict(payload)
