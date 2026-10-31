import uuid
from dataclasses import dataclass
from enum import StrEnum


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"


@dataclass(frozen=True, slots=True)
class SearchScope:
    """Who is searching and, optionally, which of their documents to search.

    ``document_ids`` must already be restricted to documents the owner may read.
    """

    owner_id: uuid.UUID
    document_ids: frozenset[uuid.UUID] | None = None


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    score: float
