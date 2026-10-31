import asyncio

import pytest
from fastapi.testclient import TestClient

from rag_platform import __version__
from rag_platform.infrastructure.health import check_all
from tests.conftest import healthy_probe


async def failing_probe() -> None:
    raise ConnectionRefusedError("connect to secret-host:5432 password=hunter2 refused")


async def hanging_probe() -> None:
    await asyncio.sleep(10)


def test_liveness(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_readiness_all_healthy(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert set(body["components"]) == {"postgres", "redis", "qdrant"}
    assert all(c["healthy"] for c in body["components"].values())


@pytest.mark.parametrize("probes", [{"postgres": failing_probe, "redis": healthy_probe}])
def test_readiness_reports_failure_without_leaking_details(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["components"]["postgres"] == {
        "healthy": False,
        "latency_ms": body["components"]["postgres"]["latency_ms"],
        "error": "ConnectionRefusedError",
    }
    assert body["components"]["redis"]["healthy"] is True
    assert "hunter2" not in response.text
    assert "secret-host" not in response.text


@pytest.mark.parametrize("probes", [{"postgres": failing_probe}])
def test_liveness_stays_up_when_dependencies_fail(client: TestClient) -> None:
    assert client.get("/health/live").status_code == 200


async def test_check_all_times_out_slow_probes() -> None:
    results = await check_all({"slow": hanging_probe, "fast": healthy_probe}, timeout_seconds=0.05)
    by_name = {r.name: r for r in results}
    assert by_name["slow"].healthy is False
    assert by_name["slow"].error == "timeout"
    assert by_name["fast"].healthy is True
