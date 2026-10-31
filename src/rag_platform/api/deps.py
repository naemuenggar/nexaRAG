"""FastAPI dependencies that expose the container's resources to route handlers."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.auth.tokens import decode_access_token
from rag_platform.container import Container
from rag_platform.core.config import Settings
from rag_platform.core.errors import AuthenticationError
from rag_platform.db.models import User
from rag_platform.repositories.users import UserRepository
from rag_platform.retrieval.dense import DenseRetriever
from rag_platform.services.ask import AskService
from rag_platform.services.auth import AuthService
from rag_platform.services.conversations import ConversationService
from rag_platform.services.documents import DocumentService
from rag_platform.services.search import SearchService


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

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    session: DbSessionDep,
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError("Not authenticated")
    user_id = decode_access_token(credentials.credentials, settings)
    user = await UserRepository(session).get(user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("Invalid or expired token")
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]


def get_auth_service(session: DbSessionDep, settings: SettingsDep) -> AuthService:
    return AuthService(session, settings)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


def get_document_service(session: DbSessionDep, container: ContainerDep) -> DocumentService:
    return DocumentService(session, container.storage, container.vector_index, container.settings)


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


def get_search_service(session: DbSessionDep, container: ContainerDep) -> SearchService:
    dense = DenseRetriever(
        container.qdrant, container.settings.qdrant_collection, container.embedder
    )
    return SearchService(
        session, dense, container.bm25_cache, container.reranker, container.settings
    )


SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]


def get_ask_service(search: SearchServiceDep, container: ContainerDep) -> AskService:
    return AskService(search, container.llm, container.settings)


AskServiceDep = Annotated[AskService, Depends(get_ask_service)]


def get_conversation_service(
    session: DbSessionDep, ask: AskServiceDep, settings: SettingsDep
) -> ConversationService:
    return ConversationService(session, ask, settings)


ConversationServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]
