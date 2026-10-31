import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import Document, DocumentPage


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_owned(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> Document | None:
        result = await self._session.execute(
            select(Document).where(Document.id == document_id, Document.owner_id == owner_id)
        )
        return result.scalar_one_or_none()

    async def get_owned_by_hash(self, owner_id: uuid.UUID, content_sha256: str) -> Document | None:
        result = await self._session.execute(
            select(Document).where(
                Document.owner_id == owner_id, Document.content_sha256 == content_sha256
            )
        )
        return result.scalar_one_or_none()

    async def list_owned(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[Sequence[Document], int]:
        rows = await self._session.execute(
            select(Document)
            .where(Document.owner_id == owner_id)
            .order_by(Document.created_at.desc(), Document.id)
            .limit(limit)
            .offset(offset)
        )
        total = await self._session.scalar(
            select(func.count()).select_from(Document).where(Document.owner_id == owner_id)
        )
        return rows.scalars().all(), total or 0

    async def add(self, document: Document) -> Document:
        self._session.add(document)
        await self._session.flush()
        return document

    async def delete(self, document: Document) -> None:
        await self._session.delete(document)
        await self._session.flush()

    async def list_pages(
        self, owner_id: uuid.UUID, document_id: uuid.UUID, *, limit: int, offset: int
    ) -> Sequence[DocumentPage]:
        rows = await self._session.execute(
            select(DocumentPage)
            .join(Document, Document.id == DocumentPage.document_id)
            .where(Document.id == document_id, Document.owner_id == owner_id)
            .order_by(DocumentPage.page_number)
            .limit(limit)
            .offset(offset)
        )
        return rows.scalars().all()


class DocumentPageWriter:
    """Used only by the ingestion worker, which acts on a job rather than on behalf of a user."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_pages(self, document_id: uuid.UUID, pages: Sequence[tuple[int, str]]) -> None:
        # Re-running a job must not duplicate pages, so the old set is replaced atomically.
        await self._session.execute(
            delete(DocumentPage).where(DocumentPage.document_id == document_id)
        )
        self._session.add_all(
            DocumentPage(
                document_id=document_id, page_number=number, text=text, char_count=len(text)
            )
            for number, text in pages
        )
        await self._session.flush()
