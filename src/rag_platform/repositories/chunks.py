"""Chunk reads for retrieval. Every query is scoped to one owner and to documents that are
fully ingested, so a half-processed or deleted document can never show up in results."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import Chunk, Document, DocumentStatus


@dataclass(frozen=True, slots=True)
class CorpusFingerprint:
    """Changes whenever the owner's searchable chunks change (upload, re-ingest, delete)."""

    documents: int
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

    @staticmethod
    def _ready(owner_id: uuid.UUID):  # type: ignore[no-untyped-def]
        return (Document.owner_id == owner_id, Document.status == DocumentStatus.READY)

    async def fingerprint(self, owner_id: uuid.UUID) -> CorpusFingerprint:
        documents = await self._session.execute(
            select(func.count(Document.id), func.max(Document.updated_at)).where(
                *self._ready(owner_id)
            )
        )
        doc_count, last_updated = documents.one()
        chunk_count = await self._session.scalar(
            select(func.count(Chunk.id))
            .join(Document, Document.id == Chunk.document_id)
            .where(*self._ready(owner_id))
        )
        return CorpusFingerprint(int(doc_count), int(chunk_count or 0), last_updated)

    async def corpus(self, owner_id: uuid.UUID) -> Sequence[tuple[uuid.UUID, uuid.UUID, str]]:
        """(chunk id, document id, text) for every searchable chunk, in a stable order."""
        rows = await self._session.execute(
            select(Chunk.id, Chunk.document_id, Chunk.text)
            .join(Document, Document.id == Chunk.document_id)
            .where(*self._ready(owner_id))
            .order_by(Chunk.document_id, Chunk.ordinal)
        )
        return [(r[0], r[1], r[2]) for r in rows.all()]

    async def get_many(
        self, owner_id: uuid.UUID, chunk_ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, ChunkRecord]:
        if not chunk_ids:
            return {}
        rows = await self._session.execute(
            select(Chunk, Document.filename)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(chunk_ids), Chunk.owner_id == owner_id, *self._ready(owner_id))
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

    async def owned_ready_document_ids(
        self, owner_id: uuid.UUID, document_ids: Sequence[uuid.UUID]
    ) -> set[uuid.UUID]:
        rows = await self._session.execute(
            select(Document.id).where(Document.id.in_(document_ids), *self._ready(owner_id))
        )
        return set(rows.scalars().all())
