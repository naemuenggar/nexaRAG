"""Chunk reads for retrieval. Every query is restricted to documents the user may read (see
``repositories.access``) and that are fully ingested, so a half-processed, deleted or unshared
document can never show up in results."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import String, cast, func, literal, select
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import Chunk, Document
from rag_platform.repositories.access import searchable_by


@dataclass(frozen=True, slots=True)
class CorpusFingerprint:
    """Changes whenever the user's searchable chunks change: upload, re-ingest, delete, and
    shares granted or revoked (the digest covers the exact set of document ids)."""

    documents_digest: str | None
    chunks: int
    last_updated: datetime | None


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    ordinal: int
    page_start: int
    page_end: int
    text: str


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def fingerprint(self, user_id: uuid.UUID) -> CorpusFingerprint:
        documents = await self._session.execute(
            select(
                func.md5(
                    func.string_agg(
                        cast(Document.id, String), aggregate_order_by(literal(","), Document.id)
                    )
                ),
                func.max(Document.updated_at),
            ).where(*searchable_by(user_id))
        )
        digest, last_updated = documents.one()
        chunk_count = await self._session.scalar(
            select(func.count(Chunk.id))
            .join(Document, Document.id == Chunk.document_id)
            .where(*searchable_by(user_id))
        )
        return CorpusFingerprint(digest, int(chunk_count or 0), last_updated)

    async def corpus(self, user_id: uuid.UUID) -> Sequence[tuple[uuid.UUID, uuid.UUID, str]]:
        """(chunk id, document id, text) for every searchable chunk, in a stable order."""
        rows = await self._session.execute(
            select(Chunk.id, Chunk.document_id, Chunk.text)
            .join(Document, Document.id == Chunk.document_id)
            .where(*searchable_by(user_id))
            .order_by(Chunk.document_id, Chunk.ordinal)
        )
        return [(r[0], r[1], r[2]) for r in rows.all()]

    async def get_many(
        self, user_id: uuid.UUID, chunk_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, ChunkRecord]:
        if not chunk_ids:
            return {}
        rows = await self._session.execute(
            select(Chunk, Document.filename)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(chunk_ids), *searchable_by(user_id))
        )
        return {
            chunk.id: ChunkRecord(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                ordinal=chunk.ordinal,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
            )
            for chunk, filename in rows.all()
        }

    async def searchable_document_ids(
        self,
        user_id: uuid.UUID,
        *,
        restrict_to: Sequence[uuid.UUID] | None = None,
        filename_contains: str | None = None,
        uploaded_after: datetime | None = None,
        uploaded_before: datetime | None = None,
    ) -> set[uuid.UUID]:
        conditions = list(searchable_by(user_id))
        if restrict_to is not None:
            conditions.append(Document.id.in_(restrict_to))
        if filename_contains:
            # Escape LIKE wildcards so the user's text is matched literally.
            escaped = (
                filename_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            conditions.append(Document.filename.ilike(f"%{escaped}%", escape="\\"))
        if uploaded_after is not None:
            conditions.append(Document.created_at >= uploaded_after)
        if uploaded_before is not None:
            conditions.append(Document.created_at < uploaded_before)
        rows = await self._session.execute(select(Document.id).where(*conditions))
        return set(rows.scalars().all())
