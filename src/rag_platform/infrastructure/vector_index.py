"""Chunk vectors in Qdrant.

Every point carries ``owner_id`` and ``document_id`` in its payload, with keyword indexes on
both, so searches can always be restricted to the documents a user may read.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from rag_platform.embeddings.base import Vector


@dataclass(frozen=True, slots=True)
class ChunkPoint:
    chunk_id: uuid.UUID
    owner_id: uuid.UUID
    document_id: uuid.UUID
    ordinal: int
    page_start: int
    page_end: int
    vector: Vector


class VectorIndexError(Exception):
    pass


class QdrantVectorIndex:
    def __init__(
        self, client: AsyncQdrantClient, collection: str, *, batch_size: int = 128
    ) -> None:
        self._client = client
        self._collection = collection
        self._batch_size = batch_size

    @property
    def collection(self) -> str:
        return self._collection

    async def ensure_collection(self, dimension: int) -> None:
        """Create the collection if needed; refuse to run against one built for another model."""
        if await self._client.collection_exists(self._collection):
            info = await self._client.get_collection(self._collection)
            vectors = info.config.params.vectors
            existing = vectors.size if isinstance(vectors, qm.VectorParams) else None
            if existing != dimension:
                raise VectorIndexError(
                    f"Collection {self._collection!r} has vector size {existing}, but the "
                    f"embedding model produces {dimension}. Use a new collection name and "
                    "re-index after changing the embedding model."
                )
        else:
            await self._client.create_collection(
                self._collection,
                vectors_config=qm.VectorParams(size=dimension, distance=qm.Distance.COSINE),
            )
        for field in ("owner_id", "document_id"):
            await self._client.create_payload_index(
                self._collection, field, field_schema=qm.PayloadSchemaType.KEYWORD, wait=True
            )

    async def upsert(self, points: Sequence[ChunkPoint]) -> None:
        for start in range(0, len(points), self._batch_size):
            batch = points[start : start + self._batch_size]
            await self._client.upsert(
                self._collection,
                points=[
                    qm.PointStruct(
                        id=str(p.chunk_id),
                        vector=p.vector,
                        payload={
                            "owner_id": str(p.owner_id),
                            "document_id": str(p.document_id),
                            "ordinal": p.ordinal,
                            "page_start": p.page_start,
                            "page_end": p.page_end,
                        },
                    )
                    for p in batch
                ],
                wait=True,
            )

    async def delete_document(self, document_id: uuid.UUID) -> None:
        await self._client.delete(
            self._collection,
            points_selector=qm.FilterSelector(filter=_document_filter(document_id)),
            wait=True,
        )

    async def count_document(self, document_id: uuid.UUID) -> int:
        result = await self._client.count(
            self._collection, count_filter=_document_filter(document_id), exact=True
        )
        return int(result.count)


def _document_filter(document_id: uuid.UUID) -> qm.Filter:
    return qm.Filter(
        must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=str(document_id)))]
    )
