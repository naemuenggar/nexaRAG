import uuid
from dataclasses import dataclass
from enum import StrEnum


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass(frozen=True, slots=True)
class SearchScope:
    """Who is searching and exactly which documents they may search.

    ``document_ids`` is resolved from the database (owned or shared, ready, matching filters)
    before retrieval; retrievers never widen it.
    """

    user_id: uuid.UUID
    document_ids: frozenset[uuid.UUID]


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    score: float
