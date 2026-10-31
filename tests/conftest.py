from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_platform.container import Container
from rag_platform.core.config import Environment, Settings
from rag_platform.infrastructure.health import HealthProbe
from rag_platform.main import create_app

TEST_SETTINGS: dict[str, Any] = {
    "environment": Environment.TEST,
    "log_json": False,
    "database_url": "postgresql+asyncpg://user:pass@db.invalid:5432/rag",
    "redis_url": "redis://redis.invalid:6379/0",
    "qdrant_url": "http://qdrant.invalid:6333",
    "jwt_secret": "x" * 40,
    "health_check_timeout_seconds": 0.2,
    "embedding_warm_up": False,
    "reranker_warm_up": False,
}


def make_settings(**overrides: Any) -> Settings:
    # _env_file=None keeps a developer's local .env from leaking into unit tests.
    return Settings(_env_file=None, **{**TEST_SETTINGS, **overrides})


async def healthy_probe() -> None:
    return None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(storage_dir=tmp_path / "uploads")


@pytest.fixture
def probes() -> dict[str, HealthProbe]:
    return {"postgres": healthy_probe, "redis": healthy_probe, "qdrant": healthy_probe}


@pytest.fixture
def app(settings: Settings, probes: dict[str, HealthProbe]) -> FastAPI:
    # Real clients are constructed (they connect lazily, so nothing is dialled);
    # the health probes are swapped for fakes.
    container = Container.from_settings(settings)
    container.health_probes = probes
    return create_app(container=container)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
