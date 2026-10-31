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
from rag_platform.api.routes import auth, documents, health, search
from rag_platform.container import Container
from rag_platform.core.config import Settings, get_settings
from rag_platform.core.errors import register_error_handlers
from rag_platform.core.logging import configure_logging, get_logger

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
        warm_up: asyncio.Task[None] | None = None
        if settings.embedding_warm_up:
            warm_up = asyncio.create_task(_warm_up(app.state.container, logger))
        logger.info("startup", environment=settings.environment.value, version=__version__)
        try:
            yield
        finally:
            if warm_up is not None:
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
    return app


async def _warm_up(container: Container, logger: structlog.stdlib.BoundLogger) -> None:
    try:
        await container.embedder.warm_up()
        logger.info("embedder_ready", model=container.embedder.model_name)
    except Exception:
        # Searches will retry the load and report the error; startup must not fail on it.
        logger.exception("embedder_warm_up_failed")


def app_factory() -> FastAPI:
    return create_app()
