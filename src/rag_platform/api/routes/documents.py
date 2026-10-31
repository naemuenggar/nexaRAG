import uuid
from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, File, Query, Response, UploadFile, status

from rag_platform.api.deps import CurrentUserDep, DocumentServiceDep
from rag_platform.api.schemas import (
    DocumentDetail,
    DocumentList,
    DocumentOut,
    JobOut,
    PageOut,
    ShareOut,
    ShareRequest,
)
from rag_platform.db.models import Document, DocumentShare
from rag_platform.services.documents import DocumentWithJob, iter_chunks

router = APIRouter(prefix="/documents", tags=["documents"])

Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


class Ownership(StrEnum):
    ALL = "all"
    OWNED = "owned"
    SHARED = "shared"


def _out(document: Document, user_id: uuid.UUID) -> DocumentOut:
    return DocumentOut.model_validate(document).model_copy(
        update={"access": "owner" if document.owner_id == user_id else "shared"}
    )


def _detail(item: DocumentWithJob, user_id: uuid.UUID) -> DocumentDetail:
    return DocumentDetail(
        **_out(item.document, user_id).model_dump(),
        latest_job=JobOut.model_validate(item.job) if item.job else None,
    )


def _share(share: DocumentShare, email: str) -> ShareOut:
    return ShareOut(user_id=share.user_id, email=email, created_at=share.created_at)


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
    return _detail(item, user.id)


@router.get("", response_model=DocumentList, summary="Documents you own or that were shared")
async def list_documents(
    user: CurrentUserDep,
    service: DocumentServiceDep,
    ownership: Ownership = Ownership.ALL,
    limit: Limit = 20,
    offset: Offset = 0,
) -> DocumentList:
    owned = {Ownership.ALL: None, Ownership.OWNED: True, Ownership.SHARED: False}[ownership]
    documents, total = await service.list(user.id, owned=owned, limit=limit, offset=offset)
    return DocumentList(
        items=[_out(d, user.id) for d in documents], total=total, limit=limit, offset=offset
    )


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: uuid.UUID, user: CurrentUserDep, service: DocumentServiceDep
) -> DocumentDetail:
    return _detail(await service.get(user.id, document_id), user.id)


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


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Owner only")
async def delete_document(
    document_id: uuid.UUID, user: CurrentUserDep, service: DocumentServiceDep
) -> Response:
    await service.delete(user.id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{document_id}/shares", response_model=list[ShareOut], summary="Owner only")
async def list_shares(
    document_id: uuid.UUID, user: CurrentUserDep, service: DocumentServiceDep
) -> list[ShareOut]:
    return [_share(s, e) for s, e in await service.list_shares(user.id, document_id)]


@router.post(
    "/{document_id}/shares",
    response_model=ShareOut,
    status_code=status.HTTP_201_CREATED,
    summary="Give another user read access (owner only)",
)
async def share_document(
    document_id: uuid.UUID, body: ShareRequest, user: CurrentUserDep, service: DocumentServiceDep
) -> ShareOut:
    share, email = await service.share(user.id, document_id, body.email)
    return _share(share, email)


@router.delete(
    "/{document_id}/shares/{recipient_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke read access (owner only)",
)
async def unshare_document(
    document_id: uuid.UUID,
    recipient_id: uuid.UUID,
    user: CurrentUserDep,
    service: DocumentServiceDep,
) -> Response:
    await service.unshare(user.id, document_id, recipient_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
