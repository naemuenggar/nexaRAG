"""Integration tests run against the real services from ``docker compose up -d``.

Connection settings come from the environment / .env, exactly as the app reads them. Tests that
write data use a throwaway database, so a developer's local data is never touched.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from rag_platform.core.config import Settings
from rag_platform.db.session import create_session_factory
from rag_platform.infrastructure.storage import LocalFileStorage
from rag_platform.ingestion.worker import IngestionWorker
from rag_platform.main import create_app

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def live_settings() -> Settings:
    return Settings()


@asynccontextmanager
async def scratch_database(settings: Settings) -> AsyncIterator[str]:
    base_url = make_url(str(settings.database_url))
    name = f"rag_test_{uuid.uuid4().hex[:12]}"
    admin = create_async_engine(base_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        yield base_url.set(database=name).render_as_string(hide_password=False)
    finally:
        async with admin.connect() as conn:
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        await admin.dispose()


def alembic_config(database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["database_url"] = database_url
    return config


@pytest.fixture(scope="session")
async def migrated_database_url(live_settings: Settings) -> AsyncIterator[str]:
    async with scratch_database(live_settings) as url:
        # alembic's env.py runs its own event loop, so call it from a worker thread.
        await asyncio.to_thread(command.upgrade, alembic_config(url), "head")
        yield url


@pytest.fixture
def app_settings(live_settings: Settings, migrated_database_url: str, tmp_path: Path) -> Settings:
    return live_settings.model_copy(
        update={
            "database_url": migrated_database_url,
            "storage_dir": tmp_path / "uploads",
            "max_upload_size_mb": 2,
        }
    )


@pytest.fixture
def api(app_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(app_settings), raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
async def worker_engine(app_settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(str(app_settings.database_url))
    yield engine
    await engine.dispose()


@pytest.fixture
def worker(app_settings: Settings, worker_engine: AsyncEngine) -> IngestionWorker:
    return IngestionWorker(
        create_session_factory(worker_engine),
        LocalFileStorage(app_settings.storage_dir),
        app_settings,
    )
