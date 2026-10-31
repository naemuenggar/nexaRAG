"""Document use cases: upload, list, inspect, delete. Every operation is scoped to one owner."""

import asyncio
import hashlib
import re
import unicodedata
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO

import structlog
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.core.config import Settings
from rag_platform.core.errors import (
    AppError,
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from rag_platform.db.models import Document, DocumentPage, DocumentStatus, IngestionJob
from rag_platform.infrastructure.storage import LocalFileStorage
from rag_platform.infrastructure.vector_index import QdrantVectorIndex
from rag_platform.ingestion.pdf import MAGIC_SEARCH_WINDOW, looks_like_pdf
from rag_platform.repositories.documents import DocumentRepository
from rag_platform.repositories.jobs import JobRepository

logger = structlog.stdlib.get_logger(__name__)

READ_CHUNK_BYTES = 1024 * 1024
_UNSAFE_FILENAME_CHARS = re.compile(r"[\x00-\x1f\x7f<>:\"/\\|?*]")


def sanitise_filename(raw: str | None) -> str:
    """A display name only (never a path): last path component, no control/reserved chars."""
    name = unicodedata.normalize("NFC", raw or "")
    name = PureWindowsPath(PurePosixPath(name).name).name
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    if not name:
        return "document.pdf"
    if len(name) > 255:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: 250 - len(ext)] + dot + ext) if dot and len(ext) <= 10 else name[:255]
    return name


@dataclass(frozen=True, slots=True)
class StagedUpload:
    temp_path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DocumentWithJob:
    document: Document
    job: IngestionJob | None


async def iter_chunks(stream: BinaryIO, chunk_size: int = READ_CHUNK_BYTES) -> AsyncIterator[bytes]:
    while chunk := await asyncio.to_thread(stream.read, chunk_size):
        yield chunk


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        storage: LocalFileStorage,
        vector_index: QdrantVectorIndex,
        settings: Settings,
    ) -> None:
        self._session = session
        self._index = vector_index
        self._documents = DocumentRepository(session)
        self._jobs = JobRepository(session)
        self._storage = storage
        self._settings = settings

    async def stage(self, chunks: AsyncIterator[bytes]) -> StagedUpload:
        """Stream the upload to a temp file, enforcing the size limit and the PDF signature."""
        limit = self._settings.max_upload_size_bytes
        temp_path = self._storage.new_temp_path()
        digest = hashlib.sha256()
        size = 0
        head = b""
        try:
            with await asyncio.to_thread(temp_path.open, "wb") as out:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > limit:
                        raise PayloadTooLargeError(
                            f"File exceeds the {self._settings.max_upload_size_mb} MB limit"
                        )
                    if len(head) < MAGIC_SEARCH_WINDOW:
                        head += chunk[: MAGIC_SEARCH_WINDOW - len(head)]
                    digest.update(chunk)
                    await asyncio.to_thread(out.write, chunk)
            if size == 0:
                raise AppError("The uploaded file is empty", code="empty_file")
            # Content-Type and extension are client-controlled; the file signature is not.
            if not looks_like_pdf(head):
                raise UnsupportedMediaTypeError("Only PDF files are accepted")
        except BaseException:
            await self._storage.discard_temp(temp_path)
            raise
        return StagedUpload(temp_path=temp_path, sha256=digest.hexdigest(), size_bytes=size)

    async def upload(
        self, owner_id: uuid.UUID, filename: str | None, chunks: AsyncIterator[bytes]
    ) -> DocumentWithJob:
        staged = await self.stage(chunks)
        try:
            return await self._create(owner_id, sanitise_filename(filename), staged)
        finally:
            await self._storage.discard_temp(staged.temp_path)  # no-op once moved into place

    async def _create(
        self, owner_id: uuid.UUID, filename: str, staged: StagedUpload
    ) -> DocumentWithJob:
        existing = await self._documents.get_owned_by_hash(owner_id, staged.sha256)
        if existing is not None:
            raise _duplicate(existing.id)

        document_id = uuid.uuid4()
        key = self._storage.document_key(owner_id, document_id)
        document = Document(
            id=document_id,
            owner_id=owner_id,
            filename=filename,
            content_sha256=staged.sha256,
            size_bytes=staged.size_bytes,
            mime_type="application/pdf",
            storage_path=key,
            status=DocumentStatus.PENDING,
        )
        try:
            await self._documents.add(document)
        except IntegrityError as exc:  # the same file uploaded concurrently
            await self._session.rollback()
            raced = await self._documents.get_owned_by_hash(owner_id, staged.sha256)
            raise _duplicate(raced.id if raced else None) from exc
        job = await self._jobs.enqueue(document_id)

        await self._storage.commit_temp(staged.temp_path, key)
        try:
            await self._session.commit()
        except BaseException:
            await self._storage.delete(key)
            raise
        logger.info("document_uploaded", document_id=str(document_id), size_bytes=staged.size_bytes)
        return DocumentWithJob(document, job)

    async def get(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> DocumentWithJob:
        document = await self._get_owned(owner_id, document_id)
        return DocumentWithJob(document, await self._jobs.latest_for_document(document.id))

    async def list(
        self, owner_id: uuid.UUID, *, limit: int, offset: int
    ) -> tuple[Sequence[Document], int]:
        return await self._documents.list_owned(owner_id, limit=limit, offset=offset)

    async def pages(
        self, owner_id: uuid.UUID, document_id: uuid.UUID, *, limit: int, offset: int
    ) -> Sequence[DocumentPage]:
        await self._get_owned(owner_id, document_id)
        return await self._documents.list_pages(owner_id, document_id, limit=limit, offset=offset)

    async def delete(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> None:
        document = await self._get_owned(owner_id, document_id)
        key = document.storage_path
        await self._documents.delete(document)  # pages, chunks and jobs cascade
        await self._session.commit()
        await self._storage.delete(key)
        try:
            await self._index.delete_document(document_id)
        except Exception:
            # The database rows are already gone and retrieval only returns vectors whose chunk
            # row exists, so a leftover vector is unreachable; log it for cleanup.
            logger.exception("vector_delete_failed", document_id=str(document_id))
        logger.info("document_deleted", document_id=str(document_id))

    async def _get_owned(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        document = await self._documents.get_owned(owner_id, document_id)
        if document is None:
            # Same answer whether it doesn't exist or belongs to someone else: no existence leak.
            raise NotFoundError("Document not found")
        return document


def _duplicate(existing_id: uuid.UUID | None) -> ConflictError:
    return ConflictError(
        "This file has already been uploaded",
        code="duplicate_document",
        details={"document_id": str(existing_id)} if existing_id else None,
    )
