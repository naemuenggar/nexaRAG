"""Application factory. Serve with ``uvicorn --factory rag_platform.main:app_factory``."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_platform import __version__
from rag_platform.api.body_limit import BodySizeLimitMiddleware
from rag_platform.api.middleware import RequestContextMiddleware
from rag_platform.api.routes import ask, auth, conversations, documents, health, search
from rag_platform.container import Container
from rag_platform.core.config import Settings, get_settings
from rag_platform.core.errors import register_error_handlers
from rag_platform.core.logging import configure_logging, get_logger
from rag_platform.embeddings.base import Embedder
from rag_platform.reranking.base import Reranker

# Room for multipart boundaries and part headers on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings/container; production uses the environment."""
    settings = settings or (container.settings if container else get_settings())
    configure_logging(settings.log_level, json=settings.log_json)
    logger = get_logger("rag_platform")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or Container.from_settings(settings)
        await app.state.container.storage.ensure_ready()
        warm_up = asyncio.create_task(_warm_up(app.state.container, settings, logger))
        logger.info("startup", environment=settings.environment.value, version=__version__)
        try:
            yield
        finally:
            warm_up.cancel()
            await app.state.container.aclose()
            logger.info("shutdown")

    app = FastAPI(
        title="Enterprise RAG Platform",
        version=__version__,
        lifespan=lifespan,
        # Interactive docs are useful locally but expand the attack surface in production.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )
    # Registered before RequestContextMiddleware so it runs inside it (413s get a request id).
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body_bytes=settings.max_upload_size_bytes + MULTIPART_OVERHEAD_BYTES,
    )
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    app.include_router(conversations.router)
    return app


async def _warm_up(
    container: Container, settings: Settings, logger: structlog.stdlib.BoundLogger
) -> None:
    """Load models in the background so the first search isn't slow; never fails startup."""
    models: list[Embedder | Reranker] = []
    if settings.embedding_warm_up:
        models.append(container.embedder)
    if settings.reranker_warm_up:
        models.append(container.reranker)
    for model in models:
        try:
            await model.warm_up()
            logger.info("model_ready", model=model.model_name)
        except Exception:
            # Requests will retry the load and report the error themselves.
            logger.exception("model_warm_up_failed", model=model.model_name)


def app_factory() -> FastAPI:
    return create_app()
