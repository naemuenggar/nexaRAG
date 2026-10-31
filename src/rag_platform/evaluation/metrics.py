"""Retrieval and citation metrics. Relevance is judged at page level: a retrieved chunk counts as
relevant when it comes from an expected file and its page span covers an expected page.

With R the set of expected (file, page) pairs and the first k results:
* Recall@k  = |expected pages covered by the top k| / |R|
* MRR       = 1 / rank of the first relevant result (0 if none in the list)
* nDCG@k    = DCG@k / IDCG@k with binary gains; a result scores only for expected pages not
              already covered higher up, so duplicates of one page don't inflate the score.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedRef:
    filename: str
    page_start: int
    page_end: int


Expected = set[tuple[str, int]]


def covered(ref: RetrievedRef, expected: Expected) -> set[tuple[str, int]]:
    return {
        (f, p) for f, p in expected if f == ref.filename and ref.page_start <= p <= ref.page_end
    }


def recall_at_k(results: Sequence[RetrievedRef], expected: Expected, k: int) -> float:
    if not expected:
        return 0.0
    found: set[tuple[str, int]] = set()
    for ref in results[:k]:
        found |= covered(ref, expected)
    return len(found) / len(expected)


def reciprocal_rank(results: Sequence[RetrievedRef], expected: Expected) -> float:
    for rank, ref in enumerate(results, start=1):
        if covered(ref, expected):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: Sequence[RetrievedRef], expected: Expected, k: int) -> float:
    if not expected:
        return 0.0
    seen: set[tuple[str, int]] = set()
    dcg = 0.0
    for rank, ref in enumerate(results[:k], start=1):
        new = covered(ref, expected) - seen
        if new:
            dcg += 1.0 / math.log2(rank + 1)
            seen |= new
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(expected), k) + 1))
    return dcg / ideal


def citation_precision(cited: Sequence[RetrievedRef], expected: Expected) -> float | None:
    """Share of citations that point at an expected page. None when nothing was cited."""
    if not cited:
        return None
    return sum(1 for ref in cited if covered(ref, expected)) / len(cited)


def citation_recall(cited: Sequence[RetrievedRef], expected: Expected) -> float:
    found: set[tuple[str, int]] = set()
    for ref in cited:
        found |= covered(ref, expected)
    return len(found) / len(expected) if expected else 0.0


@dataclass(frozen=True, slots=True)
class Summary:
    mean: float
    p50: float
    p95: float
    n: int


def summarise(values: Sequence[float]) -> Summary | None:
    if not values:
        return None
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return Summary(
        mean=statistics.fmean(ordered),
        p50=statistics.median(ordered),
        p95=ordered[p95_index],
        n=len(ordered),
    )
