import uuid
from typing import Annotated

from fastapi import APIRouter, File, Query, Response, UploadFile, status

from rag_platform.api.deps import CurrentUserDep, DocumentServiceDep
from rag_platform.api.schemas import DocumentDetail, DocumentList, DocumentOut, JobOut, PageOut
from rag_platform.services.documents import DocumentWithJob, iter_chunks

router = APIRouter(prefix="/documents", tags=["documents"])

Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


def _detail(item: DocumentWithJob) -> DocumentDetail:
    return DocumentDetail(
        **DocumentOut.model_validate(item.document).model_dump(),
        latest_job=JobOut.model_validate(item.job) if item.job else None,
    )


@router.post(
    "",
    response_model=DocumentDetail,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF; parsing happens in the background",
)
async def upload_document(
    user: CurrentUserDep,
    service: DocumentServiceDep,
    file: Annotated[UploadFile, File(description="PDF file")],
) -> DocumentDetail:
    item = await service.upload(user.id, file.filename, iter_chunks(file.file))
    return _detail(item)


@router.get("", response_model=DocumentList)
async def list_documents(
    user: CurrentUserDep, service: DocumentServiceDep, limit: Limit = 20, offset: Offset = 0
) -> DocumentList:
    documents, total = await service.list(user.id, limit=limit, offset=offset)
    return DocumentList(
        items=[DocumentOut.model_validate(d) for d in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: uuid.UUID, user: CurrentUserDep, service: DocumentServiceDep
) -> DocumentDetail:
    return _detail(await service.get(user.id, document_id))


@router.get("/{document_id}/pages", response_model=list[PageOut])
async def list_pages(
    document_id: uuid.UUID,
    user: CurrentUserDep,
    service: DocumentServiceDep,
    limit: Limit = 20,
    offset: Offset = 0,
) -> list[PageOut]:
    pages = await service.pages(user.id, document_id, limit=limit, offset=offset)
    return [PageOut.model_validate(p) for p in pages]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID, user: CurrentUserDep, service: DocumentServiceDep
) -> Response:
    await service.delete(user.id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
