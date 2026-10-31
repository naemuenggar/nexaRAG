"""Okapi BM25 over an in-memory corpus.

score(q, d) = sum over terms t in q of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))      (Lucene's variant: never negative)
"""

import heapq
import math
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BM25Document:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    tokens: Sequence[str]


@dataclass(frozen=True, slots=True)
class BM25Hit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    score: float


class BM25Index:
    def __init__(self, documents: Iterable[BM25Document], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._ids: list[uuid.UUID] = []
        self._document_ids: list[uuid.UUID] = []
        self._lengths: list[int] = []
        # term -> list of (doc index, term frequency): only documents containing the term are
        # visited at query time.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for index, doc in enumerate(documents):
            self._ids.append(doc.chunk_id)
            self._document_ids.append(doc.document_id)
            self._lengths.append(len(doc.tokens))
            for term, freq in Counter(doc.tokens).items():
                self._postings.setdefault(term, []).append((index, freq))
        self._avgdl = (sum(self._lengths) / len(self._lengths)) if self._lengths else 1.0
        self._avgdl = self._avgdl or 1.0

    def __len__(self) -> int:
        return len(self._ids)

    def idf(self, term: str) -> float:
        df = len(self._postings.get(term, ()))
        n = len(self._ids)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query_tokens: Sequence[str],
        top_k: int,
        *,
        allowed_documents: set[uuid.UUID] | None = None,
    ) -> list[BM25Hit]:
        if not self._ids or top_k <= 0:
            return []
        scores: dict[int, float] = {}
        for term in set(query_tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for index, freq in postings:
                if (
                    allowed_documents is not None
                    and self._document_ids[index] not in allowed_documents
                ):
                    continue
                norm = self.k1 * (1 - self.b + self.b * self._lengths[index] / self._avgdl)
                scores[index] = scores.get(index, 0.0) + idf * freq * (self.k1 + 1) / (freq + norm)
        # Ties are broken by insertion order (chunk ordinal), keeping results deterministic.
        best = heapq.nsmallest(top_k, scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            BM25Hit(self._ids[index], self._document_ids[index], score) for index, score in best
        ]
