"""Background ingestion worker. Run with ``python -m rag_platform.ingestion.worker``.

Pipeline per job: read file → parse pages → clean → chunk → embed → index in Qdrant → store
pages and chunks in PostgreSQL → mark the document ready.

Each job runs in two short transactions (claim, then finish) with the slow work in between, so
no database lock or connection is held while a PDF is processed. Qdrant is written before the
final transaction and always by "delete this document's points, then upsert", so a retry after
any failure converges to the same state.
"""

import asyncio
import contextlib
import signal
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_platform.core.config import Settings, get_settings
from rag_platform.core.logging import configure_logging
from rag_platform.db.models import Chunk, Document, DocumentStatus, IngestionJob, JobStatus
from rag_platform.db.session import create_engine, create_session_factory
from rag_platform.embeddings.base import Embedder
from rag_platform.embeddings.sentence_transformers import create_embedder
from rag_platform.infrastructure.qdrant import create_qdrant_client, create_vector_index
from rag_platform.infrastructure.storage import LocalFileStorage
from rag_platform.infrastructure.vector_index import ChunkPoint, QdrantVectorIndex
from rag_platform.ingestion.chunking import ChunkingConfig, chunk_pages
from rag_platform.ingestion.pdf import ParsedPdf, PdfParseError, parse_pdf
from rag_platform.observability.tracing import configure_tracing, tracer
from rag_platform.repositories.documents import IngestionWriter
from rag_platform.repositories.jobs import JobRepository

logger = structlog.stdlib.get_logger(__name__)
_tracer = tracer(__name__)

_GENERIC_FAILURE = "Processing failed due to an internal error."


@dataclass(frozen=True, slots=True)
class _ClaimedJob:
    job_id: uuid.UUID
    document_id: uuid.UUID
    owner_id: uuid.UUID
    storage_path: str
    attempt: int


@dataclass(frozen=True, slots=True)
class _Processed:
    parsed: ParsedPdf
    chunks: list[Chunk]


def chunking_config(settings: Settings) -> ChunkingConfig:
    return ChunkingConfig(
        size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
        cross_pages=settings.chunk_cross_pages,
    )


class IngestionWorker:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        storage: LocalFileStorage,
        embedder: Embedder,
        vector_index: QdrantVectorIndex,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._embedder = embedder
        self._index = vector_index
        self._settings = settings
        self._chunking = chunking_config(settings)
        self._lease = timedelta(seconds=settings.ingestion_lease_seconds)

    async def run_once(self) -> bool:
        """Process at most one job. Returns False when there was nothing to do."""
        claimed = await self._claim()
        if claimed is None:
            return False
        log = logger.bind(
            job_id=str(claimed.job_id),
            document_id=str(claimed.document_id),
            attempt=claimed.attempt,
        )
        log.info("ingestion_started")
        started = asyncio.get_running_loop().time()

        try:
            with _tracer.start_as_current_span("ingest.document") as span:
                span.set_attribute("rag.document_id", str(claimed.document_id))
                span.set_attribute("rag.attempt", claimed.attempt)
                processed = await self._process(claimed)
                span.set_attribute("rag.pages", processed.parsed.page_count)
                span.set_attribute("rag.chunks", len(processed.chunks))
        except PdfParseError as exc:
            log.info("ingestion_rejected", reason=exc.code)
            await self._finish_failed(claimed, exc.message, retry=False)
        except Exception:
            log.exception("ingestion_error")
            await self._finish_failed(claimed, _GENERIC_FAILURE, retry=True)
        else:
            if await self._finish_succeeded(claimed, processed):
                log.info(
                    "ingestion_succeeded",
                    pages=processed.parsed.page_count,
                    chunks=len(processed.chunks),
                    duration_ms=round((asyncio.get_running_loop().time() - started) * 1000, 2),
                )
            else:
                # Deleted while we worked: undo the indexing so no orphan vectors remain.
                await self._index.delete_document(claimed.document_id)
        return True

    async def run_forever(self, stop: asyncio.Event) -> None:
        logger.info("worker_started")
        while not stop.is_set():
            self._heartbeat()
            try:
                await self._reap_exhausted()
                worked = await self.run_once()
            except Exception:
                # A database outage must not kill the worker; back off and try again.
                logger.exception("worker_loop_error")
                worked = False
            if not worked:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        stop.wait(), timeout=self._settings.ingestion_poll_interval_seconds
                    )
        logger.info("worker_stopped")

    def _heartbeat(self) -> None:
        path = self._settings.worker_heartbeat_file
        if path is not None:
            try:
                path.touch()
            except OSError:
                logger.warning("heartbeat_write_failed", path=str(path))

    async def _process(self, claimed: _ClaimedJob) -> _Processed:
        data = await self._storage.read_bytes(claimed.storage_path)
        parsed = await asyncio.to_thread(parse_pdf, data, max_pages=self._settings.max_pdf_pages)
        pieces = await asyncio.to_thread(chunk_pages, parsed.pages, self._chunking)
        chunks = [
            Chunk(
                id=uuid.uuid4(),
                document_id=claimed.document_id,
                owner_id=claimed.owner_id,
                ordinal=piece.ordinal,
                page_start=piece.page_start,
                page_end=piece.page_end,
                text=piece.text,
                word_count=piece.word_count,
            )
            for piece in pieces
        ]
        vectors = await self._embedder.embed_passages([c.text for c in chunks])
        await self._index.delete_document(claimed.document_id)
        await self._index.upsert(
            [
                ChunkPoint(
                    chunk_id=c.id,
                    owner_id=c.owner_id,
                    document_id=c.document_id,
                    ordinal=c.ordinal,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    vector=v,
                )
                for c, v in zip(chunks, vectors, strict=True)
            ]
        )
        return _Processed(parsed, chunks)

    async def _claim(self) -> _ClaimedJob | None:
        async with self._session_factory() as session, session.begin():
            job = await JobRepository(session).claim_next(
                now=datetime.now(UTC),
                lease=self._lease,
                max_attempts=self._settings.ingestion_max_attempts,
            )
            if job is None:
                return None
            document = await session.get(Document, job.document_id)
            if document is None:  # deleted between upload and processing
                job.status = JobStatus.FAILED
                job.finished_at = datetime.now(UTC)
                return None
            document.status = DocumentStatus.PROCESSING
            document.error_message = None
            return _ClaimedJob(
                job.id, document.id, document.owner_id, document.storage_path, job.attempts
            )

    async def _finish_succeeded(self, claimed: _ClaimedJob, processed: _Processed) -> bool:
        async with self._session_factory() as session, session.begin():
            job, document = await self._load(session, claimed)
            if job is None or document is None:
                return False
            writer = IngestionWriter(session)
            await writer.replace_pages(
                document.id, [(p.number, p.text) for p in processed.parsed.pages]
            )
            await writer.replace_chunks(document.id, processed.chunks)
            document.page_count = processed.parsed.page_count
            document.status = DocumentStatus.READY
            job.status = JobStatus.SUCCEEDED
            job.error_message = None
            job.finished_at = datetime.now(UTC)
            return True

    async def _finish_failed(self, claimed: _ClaimedJob, message: str, *, retry: bool) -> None:
        async with self._session_factory() as session, session.begin():
            job, document = await self._load(session, claimed)
            if job is None or document is None:
                return
            if retry and job.attempts < self._settings.ingestion_max_attempts:
                job.status = JobStatus.QUEUED
                document.status = DocumentStatus.PENDING
            else:
                job.status = JobStatus.FAILED
                job.finished_at = datetime.now(UTC)
                document.status = DocumentStatus.FAILED
                document.error_message = message
            job.error_message = message

    async def _reap_exhausted(self) -> None:
        async with self._session_factory() as session, session.begin():
            jobs = await JobRepository(session).fail_exhausted(
                now=datetime.now(UTC),
                lease=self._lease,
                max_attempts=self._settings.ingestion_max_attempts,
            )
            for job in jobs:
                document = await session.get(Document, job.document_id)
                if document is not None:
                    document.status = DocumentStatus.FAILED
                    document.error_message = _GENERIC_FAILURE
                logger.warning("ingestion_lease_exhausted", job_id=str(job.id))

    @staticmethod
    async def _load(
        session: AsyncSession, claimed: _ClaimedJob
    ) -> tuple[IngestionJob | None, Document | None]:
        job = await session.get(IngestionJob, claimed.job_id, with_for_update=True)
        document = await session.get(Document, claimed.document_id, with_for_update=True)
        return job, document


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    configure_tracing(settings)
    engine = create_engine(settings)
    storage = LocalFileStorage(settings.storage_dir)
    await storage.ensure_ready()
    qdrant = create_qdrant_client(settings)
    index = create_vector_index(qdrant, settings)
    embedder = create_embedder(settings)
    # Load the model (downloading it on first run) and validate the collection before taking jobs.
    await embedder.warm_up()
    await index.ensure_collection(embedder.dimension)
    logger.info("embedder_ready", model=embedder.model_name, dimension=embedder.dimension)

    worker = IngestionWorker(create_session_factory(engine), storage, embedder, index, settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Not available on Windows event loops; Ctrl+C still raises KeyboardInterrupt there.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    try:
        await worker.run_forever(stop)
    finally:
        await qdrant.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
