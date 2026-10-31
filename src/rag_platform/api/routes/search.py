from fastapi import APIRouter

from rag_platform.api.deps import CurrentUserDep, SearchServiceDep, SettingsDep
from rag_platform.api.schemas import SearchHitOut, SearchRequest, SearchResponseOut
from rag_platform.core.errors import AppError

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponseOut, summary="Retrieve passages from your documents")
async def search(
    body: SearchRequest, user: CurrentUserDep, service: SearchServiceDep, settings: SettingsDep
) -> SearchResponseOut:
    top_k = body.top_k or settings.search_default_top_k
    if top_k > settings.search_max_top_k:
        raise AppError(f"top_k must be at most {settings.search_max_top_k}", code="invalid_top_k")
    result = await service.search(
        user.id, body.query, mode=body.mode, top_k=top_k, document_ids=body.document_ids
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
            )
            for hit in result.hits
        ],
    )
