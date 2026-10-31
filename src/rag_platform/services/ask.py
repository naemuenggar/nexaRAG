"""Grounded question answering: retrieve → check evidence → generate → verify citations.

The service prefers saying "I don't know" over an unsupported answer. It refuses without calling
the model when retrieval finds nothing (or nothing relevant enough), and after the call when the
model declines or cites no valid source.
"""

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import structlog
from opentelemetry.trace import Status, StatusCode

from rag_platform.core.config import Settings
from rag_platform.core.errors import ServiceUnavailableError, UpstreamError
from rag_platform.generation.citations import parse_answer, select_snippets
from rag_platform.generation.injection import looks_like_injection
from rag_platform.generation.llm import ChatMessage, LLMError, LLMProvider
from rag_platform.generation.prompts import NO_ANSWER, PromptSource, build_messages
from rag_platform.observability.metrics import (
    ASK_OUTCOMES,
    LLM_COST,
    LLM_LATENCY,
    LLM_TOKENS,
    SUSPICIOUS_SOURCES,
)
from rag_platform.observability.tracing import tracer
from rag_platform.retrieval.types import RetrievalMode
from rag_platform.services.search import DocumentFilter, SearchHit, SearchService

logger = structlog.stdlib.get_logger(__name__)
_tracer = tracer(__name__)


class RefusalReason(StrEnum):
    NO_SOURCES = "no_sources"  # retrieval returned nothing
    LOW_RELEVANCE = "low_relevance"  # best passage scored below ask_min_rerank_score
    MODEL_DECLINED = "model_declined"  # the model said the sources don't answer it
    UNSUPPORTED = "unsupported"  # the model answered but cited no valid source


@dataclass(frozen=True, slots=True)
class Citation:
    number: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_start: int
    page_end: int
    snippet: str


@dataclass(frozen=True, slots=True)
class Usage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None


@dataclass(frozen=True, slots=True)
class AskResult:
    answer: str
    answered: bool
    refusal_reason: RefusalReason | None
    citations: list[Citation]
    sources_considered: int
    suspicious_sources: int
    usage: Usage | None
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class AskService:
    def __init__(self, search: SearchService, llm: LLMProvider | None, settings: Settings) -> None:
        self._search = search
        self._llm = llm
        self._settings = settings

    async def ask(
        self,
        owner_id: uuid.UUID,
        question: str,
        *,
        top_k: int | None = None,
        document_filter: DocumentFilter | None = None,
        history: Sequence[ChatMessage] = (),
        retrieval_query: str | None = None,
    ) -> AskResult:
        """``retrieval_query`` defaults to the question; conversations pass a query that also
        carries the previous question, so follow-ups like "and in 2023?" find the right passages."""
        if self._llm is None:
            raise ServiceUnavailableError(
                "Answer generation is not configured on this server", code="llm_not_configured"
            )
        started = time.perf_counter()
        mode = self._settings.ask_retrieval_mode
        retrieved = await self._search.search(
            owner_id,
            retrieval_query or question,
            mode=mode,
            top_k=top_k or self._settings.ask_top_k,
            document_filter=document_filter,
        )
        retrieval_ms = _ms_since(started)
        hits = retrieved.hits

        refusal = self._pre_generation_refusal(hits, mode)
        if refusal is not None:
            return self._refusal(refusal, len(hits), 0, None, retrieval_ms, 0.0, started)

        sources = [
            PromptSource(
                number=i,
                filename=hit.chunk.filename,
                page_start=hit.chunk.page_start,
                page_end=hit.chunk.page_end,
                text=hit.chunk.text,
                suspicious=looks_like_injection(hit.chunk.text),
            )
            for i, hit in enumerate(hits, start=1)
        ]
        suspicious = sum(s.suspicious for s in sources)
        if suspicious:
            SUSPICIOUS_SOURCES.inc(suspicious)
            logger.warning("prompt_injection_suspected", sources=suspicious)

        generation_started = time.perf_counter()
        with _tracer.start_as_current_span("llm.generate") as span:
            # OpenTelemetry GenAI semantic conventions; Langfuse renders these as a generation.
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", self._llm.model)
            span.set_attribute("gen_ai.request.max_tokens", self._settings.llm_max_tokens)
            span.set_attribute("gen_ai.request.temperature", self._settings.llm_temperature)
            span.set_attribute("rag.sources", len(sources))
            try:
                result = await self._llm.complete(
                    build_messages(question, sources, history),
                    max_tokens=self._settings.llm_max_tokens,
                    temperature=self._settings.llm_temperature,
                )
            except LLMError as exc:
                span.set_status(Status(StatusCode.ERROR, "llm_error"))
                logger.error("llm_failed", reason=str(exc))
                raise UpstreamError(
                    "The language model is unavailable, please retry later"
                ) from exc
            span.set_attribute("gen_ai.response.model", result.model)
            span.set_attribute("gen_ai.usage.input_tokens", result.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", result.completion_tokens)
        generation_ms = _ms_since(generation_started)
        LLM_LATENCY.labels(result.model).observe(generation_ms / 1000)
        LLM_TOKENS.labels(result.model, "input").inc(result.prompt_tokens)
        LLM_TOKENS.labels(result.model, "output").inc(result.completion_tokens)
        usage = Usage(
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=self._cost(result.prompt_tokens, result.completion_tokens),
        )

        parsed = parse_answer(result.text, {s.number for s in sources})
        if parsed.invalid:
            logger.warning("invalid_citations_removed", numbers=parsed.invalid)
        if parsed.is_refusal:
            return self._refusal(
                RefusalReason.MODEL_DECLINED,
                len(hits),
                suspicious,
                usage,
                retrieval_ms,
                generation_ms,
                started,
            )
        if not parsed.cited and self._settings.ask_require_citations:
            return self._refusal(
                RefusalReason.UNSUPPORTED,
                len(hits),
                suspicious,
                usage,
                retrieval_ms,
                generation_ms,
                started,
            )

        by_number = dict(enumerate(hits, start=1))
        snippets = select_snippets(
            parsed.text, parsed.cited, {n: by_number[n].chunk.text for n in parsed.cited}
        )
        citations = [
            Citation(
                number=n,
                chunk_id=by_number[n].chunk.chunk_id,
                document_id=by_number[n].chunk.document_id,
                filename=by_number[n].chunk.filename,
                page_start=by_number[n].chunk.page_start,
                page_end=by_number[n].chunk.page_end,
                snippet=snippets[n],
            )
            for n in parsed.cited
        ]
        total_ms = _ms_since(started)
        ASK_OUTCOMES.labels("answered").inc()
        if usage.cost_usd is not None:
            LLM_COST.labels(usage.model).inc(usage.cost_usd)
        logger.info(
            "ask",
            answered=True,
            citations=len(citations),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )
        return AskResult(
            answer=parsed.text,
            answered=True,
            refusal_reason=None,
            citations=citations,
            sources_considered=len(hits),
            suspicious_sources=suspicious,
            usage=usage,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )

    def _pre_generation_refusal(
        self, hits: list[SearchHit], mode: RetrievalMode
    ) -> RefusalReason | None:
        if not hits:
            return RefusalReason.NO_SOURCES
        threshold = self._settings.ask_min_rerank_score
        if (
            threshold is not None
            and mode is RetrievalMode.HYBRID_RERANK
            and hits[0].score < threshold
        ):
            return RefusalReason.LOW_RELEVANCE
        return None

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        price_in = self._settings.llm_price_input_per_1k
        price_out = self._settings.llm_price_output_per_1k
        if price_in is None or price_out is None:
            return None
        return round(prompt_tokens / 1000 * price_in + completion_tokens / 1000 * price_out, 6)

    @staticmethod
    def _refusal(
        reason: RefusalReason,
        considered: int,
        suspicious: int,
        usage: Usage | None,
        retrieval_ms: float,
        generation_ms: float,
        started: float,
    ) -> AskResult:
        total_ms = _ms_since(started)
        ASK_OUTCOMES.labels(reason.value).inc()
        logger.info("ask", answered=False, reason=reason.value, total_ms=total_ms)
        return AskResult(
            answer=NO_ANSWER,
            answered=False,
            refusal_reason=reason,
            citations=[],
            sources_considered=considered,
            suspicious_sources=suspicious,
            usage=usage,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
