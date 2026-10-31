"""Run an evaluation end to end and write a report.

    python -m rag_platform.evaluation.runner --dataset enterprise-v1 [--generate]

Everything runs against the real stack configured in the environment (PostgreSQL, Qdrant, the
configured embedding model, reranker and LLM), isolated in a throwaway database, Qdrant
collection and upload directory that are removed afterwards. Nothing is simulated: if no LLM is
configured, generation metrics are reported as not measured.
"""

import argparse
import asyncio
import json
import platform
import shutil
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from rag_platform import __version__
from rag_platform.container import Container
from rag_platform.core.config import Settings, get_settings
from rag_platform.core.logging import configure_logging
from rag_platform.evaluation.dataset import Dataset, Question, load_dataset
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
from rag_platform.evaluation.scratch import PROJECT_ROOT, scratch_database
from rag_platform.ingestion.worker import IngestionWorker
from rag_platform.repositories.chunks import ChunkRepository
from rag_platform.retrieval.dense import DenseRetriever
from rag_platform.retrieval.types import RetrievalMode
from rag_platform.services.ask import AskService
from rag_platform.services.auth import AuthService
from rag_platform.services.documents import DocumentService
from rag_platform.services.search import SearchService

logger = structlog.stdlib.get_logger(__name__)

ALL_MODES = [m.value for m in RetrievalMode]


@dataclass
class RetrievalRow:
    question_id: str
    style: str
    mode: str
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    latency_ms: float
    top: list[str]


@dataclass
class GenerationRow:
    question_id: str
    style: str
    answerable: bool
    answered: bool
    refusal_reason: str | None
    answer: str
    citations: list[str]
    citation_precision: float | None
    citation_recall: float | None
    faithfulness: float | None
    relevancy: float | None
    correctness: float | None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_failures: int = 0


@dataclass
class Report:
    dataset: str
    started_at: str
    environment: dict[str, Any]
    ingestion: dict[str, Any]
    top_k: int
    retrieval: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] | None = None
    retrieval_rows: list[RetrievalRow] = field(default_factory=list)
    generation_rows: list[GenerationRow] = field(default_factory=list)


def _ref(filename: str, page_start: int, page_end: int) -> RetrievedRef:
    return RetrievedRef(filename, page_start, page_end)


def _expected(question: Question) -> set[tuple[str, int]]:
    return {(s.filename, s.page) for s in question.sources}


def _environment(settings: Settings, llm_model: str | None) -> dict[str, Any]:
    return {
        "platform_version": __version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "processor": platform.processor() or "unknown",
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "reranker_model": settings.reranker_model,
        "chunk_size_words": settings.chunk_size_words,
        "chunk_overlap_words": settings.chunk_overlap_words,
        "chunk_cross_pages": settings.chunk_cross_pages,
        "fusion_method": settings.fusion_method.value,
        "rrf_k": settings.rrf_k,
        "hybrid_candidates": settings.hybrid_candidates,
        "rerank_candidates": settings.rerank_candidates,
        "llm_model": llm_model,
    }


async def _ingest(container: Container, dataset: Dataset) -> tuple[uuid.UUID, dict[str, Any]]:
    settings = container.settings
    async with container.session_factory() as session:
        user = await AuthService(session, settings).register(
            f"eval-{uuid.uuid4().hex[:8]}@example.com", uuid.uuid4().hex + uuid.uuid4().hex
        )
        await session.commit()
        user_id = user.id

    async def one(pages: list[str]) -> AsyncIterator[bytes]:
        yield make_pdf(pages)

    started = time.perf_counter()
    for document in dataset.corpus.documents:
        async with container.session_factory() as session:
            service = DocumentService(session, container.storage, container.vector_index, settings)
            await service.upload(user_id, document.filename, one(document.pages))
    worker = IngestionWorker(
        container.session_factory,
        container.storage,
        container.embedder,
        container.vector_index,
        settings,
    )
    while await worker.run_once():
        pass
    seconds = time.perf_counter() - started

    async with container.session_factory() as session:
        documents, _ = await DocumentService(
            session, container.storage, container.vector_index, settings
        ).list(user_id, limit=100, offset=0)
    statuses = sorted({d.status.value for d in documents})
    if statuses != ["ready"]:
        raise RuntimeError(f"ingestion did not finish cleanly: {statuses}")
    pages = sum(len(d.pages) for d in dataset.corpus.documents)
    return user_id, {"documents": len(documents), "pages": pages, "seconds": round(seconds, 2)}


def _search_service(container: Container, session: Any) -> SearchService:
    dense = DenseRetriever(
        container.qdrant, container.settings.qdrant_collection, container.embedder
    )
    return SearchService(
        session, dense, container.bm25_cache, container.reranker, container.settings
    )


async def _evaluate_retrieval(
    container: Container, user_id: uuid.UUID, dataset: Dataset, modes: list[str], top_k: int
) -> list[RetrievalRow]:
    rows: list[RetrievalRow] = []
    for mode_name in modes:
        mode = RetrievalMode(mode_name)
        async with container.session_factory() as session:
            # Untimed warm-up: loads models and the BM25 index so latency reflects steady state.
            await _search_service(container, session).search(
                user_id, "warm up", mode=mode, top_k=top_k
            )
        for question in dataset.answerable:
            async with container.session_factory() as session:
                started = time.perf_counter()
                response = await _search_service(container, session).search(
                    user_id, question.question, mode=mode, top_k=top_k
                )
                latency_ms = (time.perf_counter() - started) * 1000
            refs = [
                _ref(h.chunk.filename, h.chunk.page_start, h.chunk.page_end) for h in response.hits
            ]
            expected = _expected(question)
            rows.append(
                RetrievalRow(
                    question_id=question.id,
                    style=question.style,
                    mode=mode_name,
                    recall_at_k=recall_at_k(refs, expected, top_k),
                    reciprocal_rank=reciprocal_rank(refs, expected),
                    ndcg_at_k=ndcg_at_k(refs, expected, top_k),
                    latency_ms=round(latency_ms, 2),
                    top=[f"{r.filename}#p{r.page_start}" for r in refs],
                )
            )
    return rows


async def _evaluate_generation(
    container: Container, user_id: uuid.UUID, dataset: Dataset
) -> list[GenerationRow]:
    if container.llm is None:
        raise RuntimeError("generation needs a configured LLM")
    judge = Judge(container.llm)
    rows: list[GenerationRow] = []
    for question in dataset.questions:
        async with container.session_factory() as session:
            ask = AskService(_search_service(container, session), container.llm, container.settings)
            started = time.perf_counter()
            result = await ask.ask(user_id, question.question)
            latency_ms = (time.perf_counter() - started) * 1000
            # The passages the answer cites, for the faithfulness judge.
            cited_text = []
            if result.citations:
                chunks = await ChunkRepository(session).get_many(
                    user_id, [c.chunk_id for c in result.citations]
                )
                cited_text = [
                    chunks[c.chunk_id].text for c in result.citations if c.chunk_id in chunks
                ]
        cited = [_ref(c.filename, c.page_start, c.page_end) for c in result.citations]
        expected = _expected(question)
        scores = (
            await judge.score(
                question.question, result.answer, question.expected_answer, cited_text
            )
            if result.answered
            else None
        )
        usage = result.usage
        rows.append(
            GenerationRow(
                question_id=question.id,
                style=question.style,
                answerable=question.answerable,
                answered=result.answered,
                refusal_reason=result.refusal_reason.value if result.refusal_reason else None,
                answer=result.answer,
                citations=[f"{r.filename}#p{r.page_start}" for r in cited],
                citation_precision=citation_precision(cited, expected)
                if question.answerable
                else None,
                citation_recall=citation_recall(cited, expected) if question.answerable else None,
                faithfulness=scores.faithfulness if scores else None,
                relevancy=scores.relevancy if scores else None,
                correctness=scores.correctness if scores else None,
                latency_ms=round(latency_ms, 2),
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                cost_usd=usage.cost_usd if usage else None,
                judge_prompt_tokens=scores.prompt_tokens if scores else 0,
                judge_completion_tokens=scores.completion_tokens if scores else 0,
                judge_failures=scores.failures if scores else 0,
            )
        )
    return rows


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


def summarise_retrieval(rows: list[RetrievalRow], modes: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in modes:
        mode_rows = [r for r in rows if r.mode == mode]
        latency = summarise([r.latency_ms for r in mode_rows])
        entry: dict[str, Any] = {
            "questions": len(mode_rows),
            "recall_at_k": _mean([r.recall_at_k for r in mode_rows]),
            "mrr": _mean([r.reciprocal_rank for r in mode_rows]),
            "ndcg_at_k": _mean([r.ndcg_at_k for r in mode_rows]),
            "latency_ms": {k: round(v, 2) for k, v in asdict(latency).items() if k != "n"}
            if latency
            else None,
        }
        for style in ("lexical", "paraphrase"):
            styled = [r for r in mode_rows if r.style == style]
            entry[f"recall_at_k_{style}"] = _mean([r.recall_at_k for r in styled])
        summary[mode] = entry
    return summary


def summarise_generation(rows: list[GenerationRow]) -> dict[str, Any]:
    answerable = [r for r in rows if r.answerable]
    unanswerable = [r for r in rows if not r.answerable]
    answered = [r for r in answerable if r.answered]
    latency = summarise([r.latency_ms for r in rows])
    costs = [r.cost_usd for r in rows if r.cost_usd is not None]
    return {
        "questions": len(rows),
        "answer_rate_answerable": round(len(answered) / len(answerable), 4) if answerable else None,
        "refusal_rate_unanswerable": round(
            sum(1 for r in unanswerable if not r.answered) / len(unanswerable), 4
        )
        if unanswerable
        else None,
        "faithfulness": _mean([r.faithfulness for r in answered]),
        "relevancy": _mean([r.relevancy for r in answered]),
        "correctness": _mean([r.correctness for r in answered]),
        "citation_precision": _mean([r.citation_precision for r in answered]),
        "citation_recall": _mean([r.citation_recall for r in answered]),
        "latency_ms": {k: round(v, 2) for k, v in asdict(latency).items() if k != "n"}
        if latency
        else None,
        "avg_prompt_tokens": _mean([float(r.prompt_tokens) for r in rows]),
        "avg_completion_tokens": _mean([float(r.completion_tokens) for r in rows]),
        "total_cost_usd": round(sum(costs), 6) if len(costs) == len(rows) else None,
        "judge_failures": sum(r.judge_failures for r in rows),
    }


def render_markdown(report: Report) -> str:
    env = report.environment
    lines = [
        f"# Evaluation: {report.dataset}",
        "",
        f"Run at {report.started_at} (UTC) on {env['machine']}, Python {env['python']}, "
        f"platform {env['platform_version']}.",
        f"Embeddings `{env['embedding_model']}` on {env['embedding_device']}, reranker "
        f"`{env['reranker_model']}` over {env['rerank_candidates']} candidates, fusion "
        f"{env['fusion_method']} (k={env['rrf_k']}, {env['hybrid_candidates']} candidates per "
        f"retriever), chunks of {env['chunk_size_words']} words with {env['chunk_overlap_words']} "
        "overlap.",
        f"Corpus: {report.ingestion['documents']} documents, {report.ingestion['pages']} pages, "
        f"ingested in {report.ingestion['seconds']} s.",
        "",
        f"## Retrieval (top {report.top_k}, {next(iter(report.retrieval.values()))['questions']} "
        "answerable questions)",
        "",
        "| Mode | Recall@k | MRR | nDCG@k | Recall@k lexical | Recall@k paraphrase "
        "| Latency mean / p50 / p95 (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, m in report.retrieval.items():
        lat = m["latency_ms"]
        lines.append(
            f"| {mode} | {m['recall_at_k']:.3f} | {m['mrr']:.3f} | {m['ndcg_at_k']:.3f} "
            f"| {m['recall_at_k_lexical']:.3f} | {m['recall_at_k_paraphrase']:.3f} "
            f"| {lat['mean']:.1f} / {lat['p50']:.1f} / {lat['p95']:.1f} |"
        )
    lines += ["", "## Generation", ""]
    gen = report.generation
    if gen is None:
        lines.append(
            "Not measured: no LLM was configured (`LLM_PROVIDER=none`). Answer faithfulness, "
            "relevancy, correctness, citation correctness, generation latency and token cost "
            "require a model."
        )
    else:

        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        lat = gen["latency_ms"]
        lines += [
            f"Model `{env['llm_model']}`, retrieval mode `hybrid_rerank`, {gen['questions']} "
            "questions. Judge: the same model (see limitations).",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Answered (answerable questions) | {fmt(gen['answer_rate_answerable'])} |",
            f"| Refused (unanswerable questions) | {fmt(gen['refusal_rate_unanswerable'])} |",
            f"| Faithfulness | {fmt(gen['faithfulness'])} |",
            f"| Relevancy | {fmt(gen['relevancy'])} |",
            f"| Correctness vs reference | {fmt(gen['correctness'])} |",
            f"| Citation precision | {fmt(gen['citation_precision'])} |",
            f"| Citation recall | {fmt(gen['citation_recall'])} |",
            f"| End-to-end latency mean / p50 / p95 (ms) | {lat['mean']:.0f} / {lat['p50']:.0f} "
            f"/ {lat['p95']:.0f} |",
            f"| Avg prompt / completion tokens | {fmt(gen['avg_prompt_tokens'])} / "
            f"{fmt(gen['avg_completion_tokens'])} |",
            f"| Total cost (USD) | {fmt(gen['total_cost_usd'])} |",
            f"| Judge parse failures | {gen['judge_failures']} |",
        ]
    lines += [
        "",
        "Per-question results are in the JSON report next to this file.",
        "",
    ]
    return "\n".join(lines)


async def run(
    dataset_name: str,
    *,
    modes: list[str],
    top_k: int,
    generate: bool,
) -> Report:
    base = get_settings()
    configure_logging("WARNING", json=base.log_json)
    dataset = load_dataset(dataset_name)
    storage_dir = Path(tempfile.mkdtemp(prefix="rag-eval-"))
    started_at = datetime.now(UTC)

    async with scratch_database(str(base.database_url), prefix="rag_eval") as database_url:
        settings = base.model_copy(
            update={
                "database_url": database_url,
                "storage_dir": storage_dir,
                "qdrant_collection": f"eval_{uuid.uuid4().hex[:12]}",
            }
        )
        container = Container.from_settings(settings)
        try:
            await container.storage.ensure_ready()
            await container.embedder.warm_up()
            await container.vector_index.ensure_collection(container.embedder.dimension)
            user_id, ingestion = await _ingest(container, dataset)
            report = Report(
                dataset=dataset.name,
                started_at=started_at.isoformat(timespec="seconds"),
                environment=_environment(settings, container.llm.model if container.llm else None),
                ingestion=ingestion,
                top_k=top_k,
            )
            report.retrieval_rows = await _evaluate_retrieval(
                container, user_id, dataset, modes, top_k
            )
            report.retrieval = summarise_retrieval(report.retrieval_rows, modes)
            if generate:
                if container.llm is None:
                    logger.warning("generation_skipped", reason="no LLM configured")
                else:
                    report.generation_rows = await _evaluate_generation(container, user_id, dataset)
                    report.generation = summarise_generation(report.generation_rows)
        finally:
            try:
                await container.qdrant.delete_collection(settings.qdrant_collection)
            finally:
                await container.aclose()
                shutil.rmtree(storage_dir, ignore_errors=True)

    return report


def write_report(report: Report, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(report.started_at).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{report.dataset}-{stamp}"
    (output_dir / f"{stem}.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown_path = output_dir / f"{stem}.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--dataset", default="enterprise-v1")
    parser.add_argument("--modes", default=",".join(ALL_MODES))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--generate", action="store_true", help="also evaluate answers (needs an LLM)"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - set(ALL_MODES)
    if unknown:
        parser.error(f"unknown modes: {sorted(unknown)}")
    report = asyncio.run(run(args.dataset, modes=modes, top_k=args.top_k, generate=args.generate))
    markdown_path = write_report(report, args.output_dir)
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"Report written to {markdown_path}")


if __name__ == "__main__":
    main()
