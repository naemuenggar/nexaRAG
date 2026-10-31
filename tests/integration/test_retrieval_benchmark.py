"""Retrieval regression on the bundled evaluation set, with the real embedding model and reranker.

Thresholds sit a little below the measured results in evaluation/results/ (see README,
"Evaluation"), so a change that makes retrieval clearly worse fails CI.
"""

import pytest

from rag_platform.evaluation.runner import ALL_MODES, run

pytestmark = pytest.mark.integration

MINIMUMS = {
    #                 Recall@5  MRR   nDCG@5
    "dense": (0.95, 0.85, 0.88),
    "sparse": (0.85, 0.65, 0.70),
    "hybrid": (0.95, 0.75, 0.80),
    "hybrid_rerank": (0.95, 0.88, 0.90),
}


async def test_retrieval_quality_does_not_regress() -> None:
    report = await run("enterprise-v1", modes=ALL_MODES, top_k=5, generate=False)
    failures = {}
    for mode, (recall, mrr, ndcg) in MINIMUMS.items():
        measured = report.retrieval[mode]
        if (
            measured["recall_at_k"] < recall
            or measured["mrr"] < mrr
            or measured["ndcg_at_k"] < ndcg
        ):
            failures[mode] = measured
    assert failures == {}
    # Reranking must not be worse than the fusion it reorders.
    assert report.retrieval["hybrid_rerank"]["mrr"] >= report.retrieval["hybrid"]["mrr"]
