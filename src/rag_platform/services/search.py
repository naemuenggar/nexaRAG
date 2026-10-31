import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.core.errors import NotFoundError
from rag_platform.repositories.chunks import ChunkRecord, ChunkRepository
from rag_platform.retrieval.dense import DenseRetriever
from rag_platform.retrieval.sparse import BM25IndexCache, SparseRetriever
from rag_platform.retrieval.types import RetrievalMode, ScoredChunk, SearchScope

logger = structlog.stdlib.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SearchHit:
    rank: int  # 1-based
    score: float
    chunk: ChunkRecord


@dataclass(frozen=True, slots=True)
class SearchResponse:
    mode: RetrievalMode
    hits: list[SearchHit]
    latency_ms: float


class SearchService:
    def __init__(
        self, session: AsyncSession, dense: DenseRetriever, bm25_cache: BM25IndexCache
    ) -> None:
        self._chunks = ChunkRepository(session)
        self._dense = dense
        self._sparse = SparseRetriever(self._chunks, bm25_cache)

    async def resolve_scope(
        self, owner_id: uuid.UUID, document_ids: Sequence[uuid.UUID] | None
    ) -> SearchScope:
        if document_ids is None:
            return SearchScope(owner_id)
        requested = set(document_ids)
        accessible = await self._chunks.owned_ready_document_ids(owner_id, list(requested))
        if accessible != requested:
            # Same message for "not yours" and "doesn't exist"; also covers "not ready yet".
            raise NotFoundError("Document not found or not ready for search")
        return SearchScope(owner_id, frozenset(accessible))

    async def search(
        self,
        owner_id: uuid.UUID,
        query: str,
        *,
        mode: RetrievalMode,
        top_k: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> SearchResponse:
        started = time.perf_counter()
        scope = await self.resolve_scope(owner_id, document_ids)
        scored = await self.retrieve(query, scope, mode=mode, top_k=top_k)
        hits = await self.hydrate(owner_id, scored)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("search", mode=mode.value, top_k=top_k, hits=len(hits), latency_ms=latency_ms)
        return SearchResponse(mode, hits, latency_ms)

    async def retrieve(
        self, query: str, scope: SearchScope, *, mode: RetrievalMode, top_k: int
    ) -> list[ScoredChunk]:
        if mode is RetrievalMode.DENSE:
            return await self._dense.search(query, scope, top_k)
        return await self._sparse.search(query, scope, top_k)

    async def hydrate(self, owner_id: uuid.UUID, scored: Sequence[ScoredChunk]) -> list[SearchHit]:
        """Attach text and citation metadata. Points without a live chunk row are dropped."""
        records = await self._chunks.get_many(owner_id, [s.chunk_id for s in scored])
        hits: list[SearchHit] = []
        for item in scored:
            record = records.get(item.chunk_id)
            if record is not None:
                hits.append(SearchHit(rank=len(hits) + 1, score=item.score, chunk=record))
        return hits
