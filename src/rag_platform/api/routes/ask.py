from fastapi import APIRouter

from rag_platform.api.deps import AskRateLimit, AskServiceDep, CurrentUserDep
from rag_platform.api.routes.search import to_filter
from rag_platform.api.schemas import AskRequest, AskResponse, CitationOut, LatencyOut, UsageOut

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post(
    "",
    response_model=AskResponse,
    summary="Answer a question from your documents, with citations",
    dependencies=[AskRateLimit],
)
async def ask(body: AskRequest, user: CurrentUserDep, service: AskServiceDep) -> AskResponse:
    result = await service.ask(
        user.id, body.question, top_k=body.top_k, document_filter=to_filter(body.filters)
    )
    return AskResponse(
        answer=result.answer,
        answered=result.answered,
        refusal_reason=result.refusal_reason.value if result.refusal_reason else None,
        citations=[
            CitationOut(
                number=c.number,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                page_start=c.page_start,
                page_end=c.page_end,
                snippet=c.snippet,
            )
            for c in result.citations
        ],
        sources_considered=result.sources_considered,
        suspicious_sources=result.suspicious_sources,
        usage=(
            UsageOut(
                model=result.usage.model,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cost_usd=result.usage.cost_usd,
            )
            if result.usage
            else None
        ),
        latency=LatencyOut(
            retrieval_ms=result.retrieval_ms,
            generation_ms=result.generation_ms,
            total_ms=result.total_ms,
        ),
    )
