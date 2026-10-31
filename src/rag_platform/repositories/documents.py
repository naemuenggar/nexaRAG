import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import Chunk, Document, DocumentPage, DocumentShare, User
from rag_platform.repositories.access import readable_by


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

    async def get_readable(self, user_id: uuid.UUID, document_id: uuid.UUID) -> Document | None:
        result = await self._session.execute(
            select(Document).where(Document.id == document_id, readable_by(user_id))
        )
        return result.scalar_one_or_none()

    async def list_readable(
        self, user_id: uuid.UUID, *, owned: bool | None, limit: int, offset: int
    ) -> tuple[Sequence[Document], int]:
        """owned=True: only the user's own; False: only shared with them; None: both."""
        conditions = [readable_by(user_id)]
        if owned is True:
            conditions.append(Document.owner_id == user_id)
        elif owned is False:
            conditions.append(Document.owner_id != user_id)
        rows = await self._session.execute(
            select(Document)
            .where(*conditions)
            .order_by(Document.created_at.desc(), Document.id)
            .limit(limit)
            .offset(offset)
        )
        total = await self._session.scalar(
            select(func.count()).select_from(Document).where(*conditions)
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
        self, user_id: uuid.UUID, document_id: uuid.UUID, *, limit: int, offset: int
    ) -> Sequence[DocumentPage]:
        rows = await self._session.execute(
            select(DocumentPage)
            .join(Document, Document.id == DocumentPage.document_id)
            .where(Document.id == document_id, readable_by(user_id))
            .order_by(DocumentPage.page_number)
            .limit(limit)
            .offset(offset)
        )
        return rows.scalars().all()

    async def list_shares(self, document_id: uuid.UUID) -> Sequence[tuple[DocumentShare, str]]:
        rows = await self._session.execute(
            select(DocumentShare, User.email)
            .join(User, User.id == DocumentShare.user_id)
            .where(DocumentShare.document_id == document_id)
            .order_by(DocumentShare.created_at)
        )
        return [(share, email) for share, email in rows.all()]

    async def add_share(self, document_id: uuid.UUID, user_id: uuid.UUID) -> DocumentShare:
        existing = await self._session.get(DocumentShare, (document_id, user_id))
        if existing is not None:
            return existing  # sharing twice is a no-op
        share = DocumentShare(document_id=document_id, user_id=user_id)
        self._session.add(share)
        await self._session.flush()
        return share

    async def remove_share(self, document_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        result = await self._session.execute(
            delete(DocumentShare).where(
                DocumentShare.document_id == document_id, DocumentShare.user_id == user_id
            )
        )
        return bool(result.rowcount)  # type: ignore[attr-defined]


class IngestionWriter:
    """Used only by the ingestion worker, which acts on a job rather than on behalf of a user.

    Re-running a job must not duplicate anything, so each write replaces the previous set.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_pages(self, document_id: uuid.UUID, pages: Sequence[tuple[int, str]]) -> None:
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

    async def replace_chunks(self, document_id: uuid.UUID, chunks: Sequence[Chunk]) -> None:
        await self._session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        self._session.add_all(chunks)
        await self._session.flush()
