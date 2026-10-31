import math
import uuid

import pytest

from rag_platform.repositories.chunks import CorpusFingerprint
from rag_platform.retrieval.bm25 import BM25Document, BM25Index
from rag_platform.retrieval.sparse import BM25IndexCache, build_index
from rag_platform.retrieval.tokenize import tokenize

DOC_A, DOC_B = uuid.uuid4(), uuid.uuid4()


def index_of(
    *texts: str, document_id: uuid.UUID = DOC_A, **params: float
) -> tuple[BM25Index, list[uuid.UUID]]:
    ids = [uuid.uuid4() for _ in texts]
    docs = [BM25Document(cid, document_id, tokenize(t)) for cid, t in zip(ids, texts, strict=True)]
    return BM25Index(docs, **params), ids


def test_score_matches_the_formula() -> None:
    index, ids = index_of("apple banana", "banana cherry cherry", "date")
    k1, b = index.k1, index.b
    avgdl = (2 + 3 + 1) / 3
    idf_cherry = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf_cherry * 2 * (k1 + 1) / (2 + k1 * (1 - b + b * 3 / avgdl))
    hits = index.search(["cherry"], 5)
    assert [h.chunk_id for h in hits] == [ids[1]]
    assert hits[0].score == pytest.approx(expected)


def test_rare_terms_weigh_more_than_common_ones() -> None:
    index, _ = index_of("solar panel", "solar roof", "solar farm", "wind turbine")
    assert index.idf("turbine") > index.idf("solar") > 0


def test_term_frequency_saturates() -> None:
    index, ids = index_of(
        "tax " * 1 + "x " * 9, "tax " * 5 + "x " * 5, "tax " * 10, "other words here"
    )
    scores = {h.chunk_id: h.score for h in index.search(["tax"], 5)}
    gain_low = scores[ids[1]] - scores[ids[0]]
    gain_high = scores[ids[2]] - scores[ids[1]]
    assert gain_low > gain_high > 0


def test_longer_documents_are_penalised() -> None:
    index, ids = index_of("merger", "merger " + "filler " * 30, "unrelated")
    hits = index.search(["merger"], 5)
    assert [h.chunk_id for h in hits] == [ids[0], ids[1]]


def test_allowed_documents_filter() -> None:
    a_ids = [uuid.uuid4()]
    b_ids = [uuid.uuid4()]
    index = BM25Index(
        [
            BM25Document(a_ids[0], DOC_A, tokenize("quarterly revenue")),
            BM25Document(b_ids[0], DOC_B, tokenize("quarterly revenue report")),
        ]
    )
    assert [h.chunk_id for h in index.search(["revenue"], 5, allowed_documents={DOC_A})] == a_ids
    assert index.search(["revenue"], 5, allowed_documents=set()) == []


def test_ties_are_broken_by_corpus_order() -> None:
    index, ids = index_of("identical text", "identical text", "identical text")
    assert [h.chunk_id for h in index.search(["identical"], 3)] == ids


def test_top_k_and_empty_cases() -> None:
    index, _ = index_of("a b c alpha", "alpha beta", "alpha gamma")
    assert len(index.search(["alpha"], 2)) == 2
    assert index.search(["missing"], 5) == []
    assert index.search(["alpha"], 0) == []
    assert BM25Index([]).search(["alpha"], 5) == []


def test_tokenize_drops_stopwords_and_keeps_codes() -> None:
    assert tokenize("What is the Q3-2024 Revenue of ACME?") == ["q3-2024", "revenue", "acme"]
    assert tokenize("the of and") == []


class FakeChunks:
    def __init__(self) -> None:
        self.fingerprint_value = CorpusFingerprint(1, 1, None)
        self.rows = [(uuid.uuid4(), DOC_A, "hello world")]

    async def fingerprint(self, owner_id: uuid.UUID) -> CorpusFingerprint:
        return self.fingerprint_value

    async def corpus(self, owner_id: uuid.UUID) -> list[tuple[uuid.UUID, uuid.UUID, str]]:
        return self.rows


async def test_cache_reuses_index_until_corpus_changes() -> None:
    cache, chunks, owner = BM25IndexCache(), FakeChunks(), uuid.uuid4()
    first = await cache.get(owner, chunks)  # type: ignore[arg-type]
    assert await cache.get(owner, chunks) is first  # type: ignore[arg-type]
    assert cache.builds == 1

    chunks.fingerprint_value = CorpusFingerprint(2, 3, None)
    chunks.rows = [*chunks.rows, (uuid.uuid4(), DOC_B, "new text")]
    second = await cache.get(owner, chunks)  # type: ignore[arg-type]
    assert second is not first
    assert len(second) == 2
    assert cache.builds == 2


async def test_cache_evicts_least_recently_used_owner() -> None:
    cache, chunks = BM25IndexCache(max_owners=2), FakeChunks()
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    for owner in (a, b, a, c):  # b is least recently used when c arrives
        await cache.get(owner, chunks)  # type: ignore[arg-type]
    builds = cache.builds
    await cache.get(a, chunks)  # type: ignore[arg-type]
    assert cache.builds == builds
    await cache.get(b, chunks)  # type: ignore[arg-type]
    assert cache.builds == builds + 1


def test_build_index_uses_parameters() -> None:
    index = build_index([(uuid.uuid4(), DOC_A, "text")], k1=0.9, b=0.4)
    assert (index.k1, index.b) == (0.9, 0.4)
