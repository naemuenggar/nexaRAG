"""Throwaway PostgreSQL databases with migrations applied, for evaluation runs and tests."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def alembic_config(database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["database_url"] = database_url
    return config


@asynccontextmanager
async def scratch_database(
    base_database_url: str, *, prefix: str = "rag_scratch", migrate: bool = True
) -> AsyncIterator[str]:
    """Create a new database next to ``base_database_url``, yield its URL, then drop it."""
    base_url = make_url(base_database_url)
    name = f"{prefix}_{uuid.uuid4().hex[:12]}"
    admin = create_async_engine(base_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = base_url.set(database=name).render_as_string(hide_password=False)
    try:
        if migrate:
            # Alembic's env.py runs its own event loop, so it has to run in another thread.
            await asyncio.to_thread(command.upgrade, alembic_config(url), "head")
        yield url
    finally:
        async with admin.connect() as conn:
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        await admin.dispose()
