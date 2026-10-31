import json
import math
from pathlib import Path

import pytest

from rag_platform.evaluation.dataset import load_dataset
from rag_platform.evaluation.judge import Judge
from rag_platform.evaluation.metrics import (
    RetrievedRef,
    citation_precision,
    citation_recall,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    summarise,
)
from rag_platform.evaluation.pdf import make_pdf
from rag_platform.evaluation.runner import (
    GenerationRow,
    RetrievalRow,
    summarise_generation,
    summarise_retrieval,
)
from rag_platform.ingestion.pdf import parse_pdf
from tests.fakes import ScriptedLLM

A1, A2, B1 = RetrievedRef("a.pdf", 1, 1), RetrievedRef("a.pdf", 2, 2), RetrievedRef("b.pdf", 1, 1)
EXPECTED = {("a.pdf", 2)}


def test_recall_mrr_ndcg_for_a_hit_at_rank_two() -> None:
    results = [B1, A2, A1]
    assert recall_at_k(results, EXPECTED, 5) == 1.0
    assert recall_at_k(results, EXPECTED, 1) == 0.0
    assert reciprocal_rank(results, EXPECTED) == 0.5
    assert ndcg_at_k(results, EXPECTED, 5) == pytest.approx(1 / math.log2(3))


def test_perfect_and_empty_rankings() -> None:
    assert ndcg_at_k([A2], EXPECTED, 5) == 1.0
    assert reciprocal_rank([], EXPECTED) == 0.0
    assert recall_at_k([B1], EXPECTED, 5) == 0.0


def test_multi_page_relevance_and_duplicates() -> None:
    expected = {("a.pdf", 1), ("a.pdf", 2)}
    spanning = RetrievedRef("a.pdf", 1, 2)
    assert recall_at_k([spanning], expected, 5) == 1.0
    # A duplicate of an already-covered page adds nothing to nDCG.
    assert ndcg_at_k([A1, A1, A2], expected, 5) == pytest.approx(
        (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )
    assert recall_at_k([A1, A1], expected, 5) == 0.5


def test_citation_metrics() -> None:
    assert citation_precision([A2, B1], EXPECTED) == 0.5
    assert citation_precision([], EXPECTED) is None
    assert citation_recall([A2], EXPECTED) == 1.0
    assert citation_recall([B1], EXPECTED) == 0.0


def test_summary_percentiles() -> None:
    summary = summarise([float(v) for v in range(1, 101)])
    assert summary is not None
    assert (summary.mean, summary.p50, summary.p95, summary.n) == (50.5, 50.5, 95.0, 100)
    assert summarise([]) is None


def test_bundled_dataset_is_consistent_and_parseable() -> None:
    dataset = load_dataset("enterprise-v1")
    assert len(dataset.answerable) >= 30
    assert len(dataset.unanswerable) >= 5
    for document in dataset.corpus.documents:
        parsed = parse_pdf(make_pdf(document.pages), max_pages=100)
        assert parsed.page_count == len(document.pages)
        # Every page the questions point at must survive PDF round-tripping with its text.
        assert all(p.text for p in parsed.pages)


def test_dataset_validation_rejects_bad_page_refs(tmp_path: Path) -> None:
    (tmp_path / "corpus.json").write_text(
        json.dumps({"documents": [{"filename": "x.pdf", "pages": ["only page"]}]})
    )
    (tmp_path / "questions.jsonl").write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "?",
                "expected_answer": "a",
                "sources": [{"filename": "x.pdf", "page": 2}],
                "style": "lexical",
            }
        )
    )
    with pytest.raises(ValueError, match=r"p\.2"):
        load_dataset(tmp_path)


async def test_judge_parses_scores_and_counts_tokens() -> None:
    llm = ScriptedLLM(
        '{"claims": 4, "supported": 3}', 'Sure! {"score": 1}', '```json\n{"score": 0.5}\n```'
    )
    scores = await Judge(llm).score("q?", "answer", "reference", ["passage <one>"])
    assert (scores.faithfulness, scores.relevancy, scores.correctness) == (0.75, 1.0, 0.5)
    assert scores.failures == 0
    assert scores.prompt_tokens > 0
    assert "passage &lt;one&gt;" in llm.prompts[0][1].content  # passages are escaped data


async def test_judge_records_failures_instead_of_guessing() -> None:
    scores = await Judge(ScriptedLLM("no json here")).score("q", "a", "r", ["p"])
    assert (scores.faithfulness, scores.relevancy, scores.correctness) == (None, None, None)
    assert scores.failures == 3


async def test_judge_clamps_out_of_range_scores() -> None:
    llm = ScriptedLLM('{"claims": 2, "supported": 5}', '{"score": 7}')
    scores = await Judge(llm).score("q", "a", None, ["p"])
    assert scores.faithfulness == 1.0
    assert scores.relevancy == 1.0
    assert scores.correctness is None  # no reference answer to compare with


def test_summaries_aggregate_rows() -> None:
    rows = [
        RetrievalRow("q1", "lexical", "dense", 1.0, 1.0, 1.0, 10.0, []),
        RetrievalRow("q2", "paraphrase", "dense", 0.0, 0.0, 0.0, 30.0, []),
    ]
    dense = summarise_retrieval(rows, ["dense"])["dense"]
    assert (dense["recall_at_k"], dense["mrr"], dense["recall_at_k_lexical"]) == (0.5, 0.5, 1.0)
    assert dense["latency_ms"]["mean"] == 20.0

    def gen(qid: str, answerable: bool, answered: bool, **kw: object) -> GenerationRow:
        base: dict[str, object] = {
            "style": "lexical" if answerable else "unanswerable",
            "refusal_reason": None if answered else "model_declined",
            "answer": "x",
            "citations": [],
            "citation_precision": None,
            "citation_recall": None,
            "faithfulness": None,
            "relevancy": None,
            "correctness": None,
            "latency_ms": 100.0,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cost_usd": None,
        }
        base.update(kw)
        return GenerationRow(qid, answerable=answerable, answered=answered, **base)  # type: ignore[arg-type]

    summary = summarise_generation(
        [
            gen("a1", True, True, faithfulness=1.0, citation_precision=1.0),
            gen("a2", True, False),
            gen("u1", False, False),
            gen("u2", False, True),
        ]
    )
    assert summary["answer_rate_answerable"] == 0.5
    assert summary["refusal_rate_unanswerable"] == 0.5
    assert summary["faithfulness"] == 1.0
    assert summary["total_cost_usd"] is None  # unknown unless every row has a cost
