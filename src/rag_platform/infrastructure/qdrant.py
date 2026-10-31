from qdrant_client import AsyncQdrantClient

from rag_platform.core.config import Settings
from rag_platform.infrastructure.vector_index import QdrantVectorIndex


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    api_key = settings.qdrant_api_key
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=api_key.get_secret_value() if api_key is not None else None,
        # Compatibility is not checked at construction time; /health/ready covers reachability.
        check_compatibility=False,
    )


def create_vector_index(client: AsyncQdrantClient, settings: Settings) -> QdrantVectorIndex:
    return QdrantVectorIndex(
        client, settings.qdrant_collection, batch_size=settings.qdrant_upsert_batch_size
    )
