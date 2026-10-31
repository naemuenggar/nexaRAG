from fastapi import APIRouter

from rag_platform.api.deps import CurrentUserDep, SearchServiceDep, SettingsDep
from rag_platform.api.schemas import (
    SearchFilters,
    SearchHitOut,
    SearchRequest,
    SearchResponseOut,
)
from rag_platform.core.errors import AppError
from rag_platform.services.search import DocumentFilter

router = APIRouter(prefix="/search", tags=["search"])


def to_filter(filters: SearchFilters | None) -> DocumentFilter | None:
    if filters is None:
        return None
    return DocumentFilter(
        document_ids=filters.document_ids,
        filename_contains=filters.filename_contains,
        uploaded_after=filters.uploaded_after,
        uploaded_before=filters.uploaded_before,
    )


@router.post("", response_model=SearchResponseOut, summary="Retrieve passages from your documents")
async def search(
    body: SearchRequest, user: CurrentUserDep, service: SearchServiceDep, settings: SettingsDep
) -> SearchResponseOut:
    top_k = body.top_k or settings.search_default_top_k
    if top_k > settings.search_max_top_k:
        raise AppError(f"top_k must be at most {settings.search_max_top_k}", code="invalid_top_k")
    result = await service.search(
        user.id, body.query, mode=body.mode, top_k=top_k, document_filter=to_filter(body.filters)
    )
    return SearchResponseOut(
        mode=result.mode,
        latency_ms=result.latency_ms,
        results=[
            SearchHitOut(
                rank=hit.rank,
                score=hit.score,
                chunk_id=hit.chunk.chunk_id,
                document_id=hit.chunk.document_id,
                filename=hit.chunk.filename,
                page_start=hit.chunk.page_start,
                page_end=hit.chunk.page_end,
                text=hit.chunk.text,
                scores=hit.scores,
            )
            for hit in result.hits
        ],
    )
