"""Migrations are tested on a throwaway database so a developer's data is never touched."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Connection, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

import rag_platform.db.models  # noqa: F401
from rag_platform.core.config import Settings
from rag_platform.db.base import Base

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
async def scratch_database_url(live_settings: Settings) -> AsyncIterator[str]:
    base_url = make_url(str(live_settings.database_url))
    name = f"rag_migration_test_{uuid.uuid4().hex[:12]}"
    admin = create_async_engine(base_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        yield base_url.set(database=name).render_as_string(hide_password=False)
    finally:
        async with admin.connect() as conn:
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        await admin.dispose()


def _alembic_config(database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["database_url"] = database_url
    return config


def _schema_diff(connection: Connection) -> list[object]:
    context = MigrationContext.configure(connection, opts={"compare_type": True})
    return list(compare_metadata(context, Base.metadata))


def _table_names(connection: Connection) -> set[str]:
    return set(inspect(connection).get_table_names())


async def test_upgrade_matches_models_and_downgrade_is_clean(scratch_database_url: str) -> None:
    config = _alembic_config(scratch_database_url)
    engine = create_async_engine(scratch_database_url)
    try:
        # alembic's env.py runs its own event loop, so call it from a worker thread.
        await asyncio.to_thread(command.upgrade, config, "head")
        async with engine.connect() as conn:
            assert await conn.run_sync(_schema_diff) == []
            tables = await conn.run_sync(_table_names)
        assert {"users", "documents", "ingestion_jobs", "alembic_version"} <= tables

        await asyncio.to_thread(command.downgrade, config, "base")
        async with engine.connect() as conn:
            assert await conn.run_sync(_table_names) == {"alembic_version"}
            enum_count = await conn.scalar(
                text(
                    "SELECT count(*) FROM pg_type WHERE typname IN ('document_status','job_status')"
                )
            )
        assert enum_count == 0
    finally:
        await engine.dispose()
