"""Combine ranked lists from several retrievers into one.

* RRF (reciprocal rank fusion): score = sum of 1 / (k + rank). Uses only ranks, so it needs no
  calibration between BM25 scores (unbounded) and cosine similarities (-1..1).
* Weighted: each list's scores are min-max normalised to 0..1, then combined as
  ``weight * dense + (1 - weight) * sparse``. Sensitive to score distributions, but lets one
  retriever dominate when the evaluation says it should.
"""

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from rag_platform.retrieval.types import ScoredChunk


class FusionMethod(StrEnum):
    RRF = "rrf"
    WEIGHTED = "weighted"


@dataclass(frozen=True, slots=True)
class FusedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    score: float
    # Original score from each retriever that returned this chunk, keyed by retriever name.
    sources: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked: Mapping[str, Sequence[ScoredChunk]], *, k: int = 60
) -> list[FusedChunk]:
    scores: dict[uuid.UUID, float] = {}
    meta: dict[uuid.UUID, tuple[uuid.UUID, dict[str, float]]] = {}
    for name, hits in ranked.items():
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            meta.setdefault(hit.chunk_id, (hit.document_id, {}))[1][name] = hit.score
    return _sorted(scores, meta)


def weighted_fusion(
    ranked: Mapping[str, Sequence[ScoredChunk]], *, weights: Mapping[str, float]
) -> list[FusedChunk]:
    scores: dict[uuid.UUID, float] = {}
    meta: dict[uuid.UUID, tuple[uuid.UUID, dict[str, float]]] = {}
    for name, hits in ranked.items():
        if not hits:
            continue
        values = [h.score for h in hits]
        low, high = min(values), max(values)
        span = high - low
        for hit in hits:
            # A list whose scores are all equal carries no ordering signal; count each as 1.
            normalised = (hit.score - low) / span if span > 0 else 1.0
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weights[name] * normalised
            meta.setdefault(hit.chunk_id, (hit.document_id, {}))[1][name] = hit.score
    return _sorted(scores, meta)


def _sorted(
    scores: dict[uuid.UUID, float], meta: dict[uuid.UUID, tuple[uuid.UUID, dict[str, float]]]
) -> list[FusedChunk]:
    # Ties broken by chunk id so the output never depends on dict ordering.
    ordered = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
    return [FusedChunk(cid, meta[cid][0], score, meta[cid][1]) for cid, score in ordered]
