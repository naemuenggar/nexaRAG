import pytest
from fastapi.testclient import TestClient

from rag_platform.container import Container
from rag_platform.core.config import Settings
from rag_platform.infrastructure.health import check_all
from rag_platform.main import create_app

pytestmark = pytest.mark.integration


async def test_every_dependency_is_reachable(live_settings: Settings) -> None:
    container = Container.from_settings(live_settings)
    try:
        results = await check_all(container.health_probes, timeout_seconds=5)
    finally:
        await container.aclose()
    failures = {r.name: r.error for r in results if not r.healthy}
    assert failures == {}


def test_readiness_endpoint_against_live_services(live_settings: Settings) -> None:
    with TestClient(create_app(live_settings)) as client:
        response = client.get("/health/ready")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ready"


async def test_redis_round_trip(live_settings: Settings) -> None:
    container = Container.from_settings(live_settings)
    try:
        await container.redis.set("integration:ping", "pong", ex=10)
        assert await container.redis.get("integration:ping") == "pong"
    finally:
        await container.redis.delete("integration:ping")
        await container.aclose()
