"""Rate limiting against the real Redis."""

import uuid

import pytest
from fastapi.testclient import TestClient

from rag_platform.container import Container
from rag_platform.core.config import Settings
from rag_platform.main import create_app

pytestmark = pytest.mark.integration


def test_login_brute_force_is_throttled_per_client(app_settings: Settings) -> None:
    settings = app_settings.model_copy(
        update={"rate_limit_enabled": True, "rate_limit_auth_per_minute": 5}
    )
    container = Container.from_settings(settings)
    with TestClient(create_app(container=container)) as client:
        # All TestClient requests come from the address "testclient"; start from a clean slate.
        redis = client.app.state.container.redis  # type: ignore[attr-defined]
        client.portal.call(_clear, redis)  # type: ignore[union-attr]
        body = {"email": f"nobody-{uuid.uuid4().hex[:6]}@example.com", "password": "wrong-pass"}
        responses = [client.post("/auth/login", json=body) for _ in range(7)]

    statuses = [r.status_code for r in responses]
    # A run can straddle a minute boundary, giving a fresh budget; at most two windows' worth
    # of attempts reach the password check.
    assert statuses.count(401) <= 10
    assert 429 in statuses
    limited = next(r for r in responses if r.status_code == 429)
    assert int(limited.headers["retry-after"]) > 0
    assert limited.json()["error"]["code"] == "rate_limited"


async def _clear(redis: object) -> None:
    keys = [k async for k in redis.scan_iter("ratelimit:auth:testclient:*")]  # type: ignore[attr-defined]
    if keys:
        await redis.delete(*keys)  # type: ignore[attr-defined]
