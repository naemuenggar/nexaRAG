import asyncio

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Connection, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

import rag_platform.db.models  # noqa: F401
from rag_platform.core.config import Settings
from rag_platform.db.base import Base
from tests.integration.conftest import alembic_config, scratch_database

pytestmark = pytest.mark.integration


def _schema_diff(connection: Connection) -> list[object]:
    context = MigrationContext.configure(connection, opts={"compare_type": True})
    return list(compare_metadata(context, Base.metadata))


def _table_names(connection: Connection) -> set[str]:
    return set(inspect(connection).get_table_names())


async def test_upgrade_matches_models_and_downgrade_is_clean(live_settings: Settings) -> None:
    async with scratch_database(live_settings) as url:
        config = alembic_config(url)
        engine = create_async_engine(url)
        try:
            await asyncio.to_thread(command.upgrade, config, "head")
            async with engine.connect() as conn:
                assert await conn.run_sync(_schema_diff) == []
                tables = await conn.run_sync(_table_names)
            assert {"users", "documents", "document_pages", "ingestion_jobs"} <= tables

            await asyncio.to_thread(command.downgrade, config, "base")
            async with engine.connect() as conn:
                assert await conn.run_sync(_table_names) == {"alembic_version"}
                enum_count = await conn.scalar(
                    text(
                        "SELECT count(*) FROM pg_type"
                        " WHERE typname IN ('document_status', 'job_status')"
                    )
                )
            assert enum_count == 0
        finally:
            await engine.dispose()
