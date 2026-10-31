"""Composition root: builds long-lived clients once per process and hands them to the API.

Route handlers never construct infrastructure themselves; they receive it through FastAPI
dependencies that read from this container, which keeps them testable with fakes.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from rag_platform.core.config import Settings
from rag_platform.db.session import create_engine, create_session_factory
from rag_platform.infrastructure.health import (
    HealthProbe,
    postgres_probe,
    qdrant_probe,
    redis_probe,
)
from rag_platform.infrastructure.qdrant import create_qdrant_client, create_vector_index
from rag_platform.infrastructure.storage import LocalFileStorage
from rag_platform.infrastructure.vector_index import QdrantVectorIndex


@dataclass(slots=True)
class Container:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis: Redis
    qdrant: AsyncQdrantClient
    storage: LocalFileStorage
    vector_index: QdrantVectorIndex
    health_probes: Mapping[str, HealthProbe] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Container":
        engine = create_engine(settings)
        redis = Redis.from_url(str(settings.redis_url), decode_responses=True)
        qdrant = create_qdrant_client(settings)
        return cls(
            settings=settings,
            engine=engine,
            session_factory=create_session_factory(engine),
            redis=redis,
            qdrant=qdrant,
            storage=LocalFileStorage(settings.storage_dir),
            vector_index=create_vector_index(qdrant, settings),
            health_probes={
                "postgres": postgres_probe(engine),
                "redis": redis_probe(redis),
                "qdrant": qdrant_probe(qdrant),
            },
        )

    async def aclose(self) -> None:
        await self.qdrant.close()
        await self.redis.aclose()
        await self.engine.dispose()
