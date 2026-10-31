"""Application factory. Serve with ``uvicorn --factory rag_platform.main:app_factory``."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_platform import __version__
from rag_platform.api.middleware import RequestContextMiddleware
from rag_platform.api.routes import health
from rag_platform.container import Container
from rag_platform.core.config import Settings, get_settings
from rag_platform.core.errors import register_error_handlers
from rag_platform.core.logging import configure_logging, get_logger


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings/container; production uses the environment."""
    settings = settings or (container.settings if container else get_settings())
    configure_logging(settings.log_level, json=settings.log_json)
    logger = get_logger("rag_platform")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or Container.from_settings(settings)
        logger.info("startup", environment=settings.environment.value, version=__version__)
        try:
            yield
        finally:
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
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)

    app.include_router(health.router)
    return app


def app_factory() -> FastAPI:
    return create_app()
