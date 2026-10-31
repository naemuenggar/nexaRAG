import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from rag_platform.embeddings.base import Embedder
from rag_platform.retrieval.types import ScoredChunk, SearchScope


def scope_filter(scope: SearchScope) -> qm.Filter:
    """Only the resolved, readable documents; there is no unrestricted search."""
    return qm.Filter(
        must=[
            qm.FieldCondition(
                key="document_id",
                match=qm.MatchAny(any=sorted(str(d) for d in scope.document_ids)),
            )
        ]
    )


class DenseRetriever:
    def __init__(self, client: AsyncQdrantClient, collection: str, embedder: Embedder) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder

    async def search(self, query: str, scope: SearchScope, top_k: int) -> list[ScoredChunk]:
        if not scope.document_ids:
            return []
        vector = await self._embedder.embed_query(query)
        response = await self._client.query_points(
            self._collection,
            query=vector,
            query_filter=scope_filter(scope),
            limit=top_k,
            with_payload=["document_id"],
        )
        return [
            ScoredChunk(
                chunk_id=_uuid(point.id),
                document_id=_uuid((point.payload or {})["document_id"]),
                score=float(point.score),
            )
            for point in response.points
        ]


def _uuid(value: object) -> uuid.UUID:
    return uuid.UUID(str(value))
