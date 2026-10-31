"""Search use case: scope → retrieve (dense, BM25 or both) → fuse → optional rerank → hydrate."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.core.config import Settings
from rag_platform.core.errors import NotFoundError
from rag_platform.observability.metrics import SEARCH_LATENCY
from rag_platform.observability.tracing import tracer
from rag_platform.repositories.chunks import ChunkRecord, ChunkRepository
from rag_platform.reranking.base import Reranker
from rag_platform.retrieval.dense import DenseRetriever
from rag_platform.retrieval.fusion import (
    FusedChunk,
    FusionMethod,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from rag_platform.retrieval.sparse import BM25IndexCache, SparseRetriever
from rag_platform.retrieval.types import RetrievalMode, ScoredChunk, SearchScope

logger = structlog.stdlib.get_logger(__name__)
_tracer = tracer(__name__)


@dataclass(frozen=True, slots=True)
class DocumentFilter:
    """Metadata conditions on documents; all given conditions must hold."""

    document_ids: Sequence[uuid.UUID] | None = None
    filename_contains: str | None = None
    uploaded_after: datetime | None = None
    uploaded_before: datetime | None = None

    @property
    def has_metadata(self) -> bool:
        return any(
            v is not None
            for v in (self.filename_contains, self.uploaded_after, self.uploaded_before)
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    rank: int  # 1-based
    score: float  # the score the ranking is based on (rerank > fused > single retriever)
    chunk: ChunkRecord
    scores: dict[str, float] = field(default_factory=dict)  # every stage's score, by name


@dataclass(frozen=True, slots=True)
class SearchResponse:
    mode: RetrievalMode
    hits: list[SearchHit]
    latency_ms: float


@dataclass(frozen=True, slots=True)
class _Candidate:
    chunk_id: uuid.UUID
    score: float
    scores: dict[str, float]


class SearchService:
    def __init__(
        self,
        session: AsyncSession,
        dense: DenseRetriever,
        bm25_cache: BM25IndexCache,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self._chunks = ChunkRepository(session)
        self._dense = dense
        self._sparse = SparseRetriever(self._chunks, bm25_cache)
        self._reranker = reranker
        self._settings = settings

    async def resolve_scope(
        self, user_id: uuid.UUID, document_filter: DocumentFilter | None = None
    ) -> SearchScope:
        flt = document_filter or DocumentFilter()
        if flt.document_ids is not None:
            requested = set(flt.document_ids)
            readable = await self._chunks.searchable_document_ids(
                user_id, restrict_to=list(requested)
            )
            if readable != requested:
                # Same answer for "not yours", "doesn't exist" and "not ready yet".
                raise NotFoundError("Document not found or not ready for search")
        matching = await self._chunks.searchable_document_ids(
            user_id,
            restrict_to=flt.document_ids,
            filename_contains=flt.filename_contains,
            uploaded_after=flt.uploaded_after,
            uploaded_before=flt.uploaded_before,
        )
        return SearchScope(user_id, frozenset(matching))

    async def search(
        self,
        owner_id: uuid.UUID,
        query: str,
        *,
        mode: RetrievalMode,
        top_k: int,
        document_filter: DocumentFilter | None = None,
    ) -> SearchResponse:
        started = time.perf_counter()
        with _tracer.start_as_current_span("search") as span:
            span.set_attribute("rag.search.mode", mode.value)
            span.set_attribute("rag.search.top_k", top_k)
            scope = await self.resolve_scope(owner_id, document_filter)
            span.set_attribute("rag.search.documents_in_scope", len(scope.document_ids))
            hits = await self.retrieve(query, scope, mode=mode, top_k=top_k)
            span.set_attribute("rag.search.hits", len(hits))
        elapsed = time.perf_counter() - started
        SEARCH_LATENCY.labels(mode.value).observe(elapsed)
        latency_ms = round(elapsed * 1000, 2)
        logger.info("search", mode=mode.value, top_k=top_k, hits=len(hits), latency_ms=latency_ms)
        return SearchResponse(mode, hits, latency_ms)

    async def retrieve(
        self, query: str, scope: SearchScope, *, mode: RetrievalMode, top_k: int
    ) -> list[SearchHit]:
        if mode is RetrievalMode.DENSE:
            dense = await self._dense.search(query, scope, top_k)
            return await self._hydrate(scope.user_id, _single("dense", dense))
        if mode is RetrievalMode.SPARSE:
            sparse = await self._sparse.search(query, scope, top_k)
            return await self._hydrate(scope.user_id, _single("bm25", sparse))

        fused = await self._hybrid(query, scope)
        if mode is RetrievalMode.HYBRID:
            return await self._hydrate(scope.user_id, fused[:top_k])
        return await self._rerank(query, scope, fused, top_k)

    async def _hybrid(self, query: str, scope: SearchScope) -> list[_Candidate]:
        n = self._settings.hybrid_candidates

        async def traced(name: str, coro: Awaitable[list[ScoredChunk]]) -> list[ScoredChunk]:
            with _tracer.start_as_current_span(f"retrieve.{name}") as span:
                result = await coro
                span.set_attribute("rag.retrieve.candidates", len(result))
                return result

        dense, sparse = await asyncio.gather(
            traced("dense", self._dense.search(query, scope, n)),
            traced("bm25", self._sparse.search(query, scope, n)),
        )
        ranked = {"dense": dense, "bm25": sparse}
        fused: list[FusedChunk]
        if self._settings.fusion_method is FusionMethod.RRF:
            fused = reciprocal_rank_fusion(ranked, k=self._settings.rrf_k)
        else:
            weight = self._settings.fusion_dense_weight
            fused = weighted_fusion(ranked, weights={"dense": weight, "bm25": 1 - weight})
        return [_Candidate(f.chunk_id, f.score, {**f.sources, "fused": f.score}) for f in fused]

    async def _rerank(
        self, query: str, scope: SearchScope, fused: list[_Candidate], top_k: int
    ) -> list[SearchHit]:
        pool = await self._hydrate(scope.user_id, fused[: self._settings.rerank_candidates])
        if not pool:
            return []
        with _tracer.start_as_current_span("rerank") as span:
            span.set_attribute("rag.rerank.model", self._reranker.model_name)
            span.set_attribute("rag.rerank.candidates", len(pool))
            scores = await self._reranker.score(query, [h.chunk.text for h in pool])
        # Stable sort: equal rerank scores keep their fused order.
        ordered = sorted(zip(pool, scores, strict=True), key=lambda pair: -pair[1])[:top_k]
        return [
            SearchHit(rank=i, score=s, chunk=h.chunk, scores={**h.scores, "rerank": s})
            for i, (h, s) in enumerate(ordered, start=1)
        ]

    async def _hydrate(
        self, owner_id: uuid.UUID, candidates: Sequence[_Candidate]
    ) -> list[SearchHit]:
        """Attach text and citation metadata. Points without a live chunk row are dropped."""
        records = await self._chunks.get_many(owner_id, [c.chunk_id for c in candidates])
        hits: list[SearchHit] = []
        for candidate in candidates:
            record = records.get(candidate.chunk_id)
            if record is not None:
                hits.append(
                    SearchHit(len(hits) + 1, candidate.score, record, dict(candidate.scores))
                )
        return hits


def _single(name: str, scored: Sequence[ScoredChunk]) -> list[_Candidate]:
    return [_Candidate(s.chunk_id, s.score, {name: s.score}) for s in scored]
