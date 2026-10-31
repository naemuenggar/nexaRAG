"""BM25 retrieval over the caller's own chunks.

Each owner gets a separate index, so term statistics never mix users' documents. Indexes are
built from PostgreSQL on first use and cached in-process; a cheap fingerprint query (document and
chunk counts, last update) tells when a cached index is stale.
"""

import asyncio
import uuid
from collections import OrderedDict

from rag_platform.repositories.chunks import ChunkRepository, CorpusFingerprint
from rag_platform.retrieval.bm25 import BM25Document, BM25Index
from rag_platform.retrieval.tokenize import tokenize
from rag_platform.retrieval.types import ScoredChunk, SearchScope


def build_index(
    rows: list[tuple[uuid.UUID, uuid.UUID, str]], *, k1: float = 1.5, b: float = 0.75
) -> BM25Index:
    return BM25Index(
        (BM25Document(cid, did, tokenize(text)) for cid, did, text in rows), k1=k1, b=b
    )


class BM25IndexCache:
    def __init__(self, max_owners: int = 64, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._max_owners = max_owners
        self._k1 = k1
        self._b = b
        self._entries: OrderedDict[uuid.UUID, tuple[CorpusFingerprint, BM25Index]] = OrderedDict()
        self._locks: dict[uuid.UUID, asyncio.Lock] = {}
        self.builds = 0  # observable in tests and logs

    async def get(self, owner_id: uuid.UUID, chunks: ChunkRepository) -> BM25Index:
        fingerprint = await chunks.fingerprint(owner_id)
        cached = self._entries.get(owner_id)
        if cached and cached[0] == fingerprint:
            self._entries.move_to_end(owner_id)
            return cached[1]
        lock = self._locks.setdefault(owner_id, asyncio.Lock())
        async with lock:
            cached = self._entries.get(owner_id)
            if cached and cached[0] == fingerprint:
                return cached[1]
            rows = list(await chunks.corpus(owner_id))
            index = await asyncio.to_thread(build_index, rows, k1=self._k1, b=self._b)
            self.builds += 1
            self._entries[owner_id] = (fingerprint, index)
            self._entries.move_to_end(owner_id)
            while len(self._entries) > self._max_owners:
                evicted, _ = self._entries.popitem(last=False)
                self._locks.pop(evicted, None)
            return index


class SparseRetriever:
    def __init__(self, chunks: ChunkRepository, cache: BM25IndexCache) -> None:
        self._chunks = chunks
        self._cache = cache

    async def search(self, query: str, scope: SearchScope, top_k: int) -> list[ScoredChunk]:
        if scope.document_ids is not None and not scope.document_ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        index = await self._cache.get(scope.owner_id, self._chunks)
        allowed = set(scope.document_ids) if scope.document_ids is not None else None
        hits = index.search(tokens, top_k, allowed_documents=allowed)
        return [ScoredChunk(h.chunk_id, h.document_id, h.score) for h in hits]
