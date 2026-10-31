import uuid

import pytest

from rag_platform.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from rag_platform.retrieval.types import ScoredChunk

DOC = uuid.uuid4()
A, B, C, D = (uuid.UUID(int=i) for i in range(1, 5))


def hits(*pairs: tuple[uuid.UUID, float]) -> list[ScoredChunk]:
    return [ScoredChunk(cid, DOC, score) for cid, score in pairs]


def test_rrf_scores_follow_the_formula() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": hits((A, 0.9), (B, 0.8)), "bm25": hits((B, 12.0), (C, 3.0))}, k=60
    )
    by_id = {f.chunk_id: f.score for f in fused}
    assert by_id[B] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[A] == pytest.approx(1 / 61)
    assert by_id[C] == pytest.approx(1 / 62)
    assert [f.chunk_id for f in fused] == [B, A, C]


def test_rrf_keeps_original_scores_per_source() -> None:
    fused = reciprocal_rank_fusion({"dense": hits((A, 0.9)), "bm25": hits((A, 7.5), (B, 1.0))})
    assert fused[0].sources == {"dense": 0.9, "bm25": 7.5}
    assert fused[1].sources == {"bm25": 1.0}


def test_rrf_ignores_score_scale() -> None:
    small = reciprocal_rank_fusion({"x": hits((A, 0.001), (B, 0.0005))})
    large = reciprocal_rank_fusion({"x": hits((A, 1000.0), (B, 999.0))})
    assert [f.score for f in small] == [f.score for f in large]


def test_weighted_fusion_normalises_each_list() -> None:
    fused = weighted_fusion(
        {"dense": hits((A, 0.9), (B, 0.5)), "bm25": hits((B, 20.0), (C, 10.0))},
        weights={"dense": 0.7, "bm25": 0.3},
    )
    by_id = {f.chunk_id: f.score for f in fused}
    assert by_id[A] == pytest.approx(0.7 * 1.0)
    assert by_id[B] == pytest.approx(0.7 * 0.0 + 0.3 * 1.0)
    assert by_id[C] == pytest.approx(0.3 * 0.0)
    assert [f.chunk_id for f in fused] == [A, B, C]


def test_weighted_fusion_with_flat_scores_counts_each_hit_fully() -> None:
    fused = weighted_fusion({"dense": hits((A, 0.5), (B, 0.5))}, weights={"dense": 1.0})
    assert [f.score for f in fused] == [1.0, 1.0]


def test_weight_zero_disables_a_retriever() -> None:
    fused = weighted_fusion(
        {"dense": hits((A, 0.9), (B, 0.1)), "bm25": hits((B, 5.0), (A, 1.0))},
        weights={"dense": 0.0, "bm25": 1.0},
    )
    assert [f.chunk_id for f in fused] == [B, A]


def test_ties_are_deterministic() -> None:
    fused = reciprocal_rank_fusion({"dense": hits((D, 1.0)), "bm25": hits((C, 1.0))})
    assert [f.chunk_id for f in fused] == sorted([C, D], key=str)


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}) == []
    assert weighted_fusion({"dense": []}, weights={"dense": 1.0}) == []
