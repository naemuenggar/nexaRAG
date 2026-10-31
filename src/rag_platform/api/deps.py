"""FastAPI dependencies that expose the container's resources to route handlers."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.container import Container
from rag_platform.core.config import Settings


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_settings_dep(container: ContainerDep) -> Settings:
    return container.settings


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


async def get_db_session(container: ContainerDep) -> AsyncIterator[AsyncSession]:
    """One session per request; commits on success, rolls back on any exception."""
    async with container.session_factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
