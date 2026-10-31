"""Background ingestion worker. Run with ``python -m rag_platform.ingestion.worker``.

Each job runs in two short transactions (claim, then finish) with the slow parsing in between,
so no database lock or connection is held while a PDF is being processed.
"""

import asyncio
import contextlib
import signal
import uuid
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_platform.core.config import Settings, get_settings
from rag_platform.core.logging import configure_logging
from rag_platform.db.models import Document, DocumentStatus, IngestionJob, JobStatus
from rag_platform.db.session import create_engine, create_session_factory
from rag_platform.infrastructure.storage import LocalFileStorage
from rag_platform.ingestion.pdf import ParsedPdf, PdfParseError, parse_pdf
from rag_platform.repositories.documents import DocumentPageWriter
from rag_platform.repositories.jobs import JobRepository

logger = structlog.stdlib.get_logger(__name__)

_GENERIC_FAILURE = "Processing failed due to an internal error."


class IngestionWorker:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        storage: LocalFileStorage,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._settings = settings
        self._lease = timedelta(seconds=settings.ingestion_lease_seconds)

    async def run_once(self) -> bool:
        """Process at most one job. Returns False when there was nothing to do."""
        claimed = await self._claim()
        if claimed is None:
            return False
        job_id, document_id, storage_path, attempt = claimed
        log = logger.bind(job_id=str(job_id), document_id=str(document_id), attempt=attempt)
        log.info("ingestion_started")
        started = asyncio.get_running_loop().time()

        try:
            data = await self._storage.read_bytes(storage_path)
            parsed = await asyncio.to_thread(
                parse_pdf, data, max_pages=self._settings.max_pdf_pages
            )
        except PdfParseError as exc:
            log.info("ingestion_rejected", reason=exc.code)
            await self._finish_failed(job_id, document_id, exc.message, retry=False)
        except Exception:
            log.exception("ingestion_error")
            await self._finish_failed(job_id, document_id, _GENERIC_FAILURE, retry=True)
        else:
            await self._finish_succeeded(job_id, document_id, parsed)
            log.info(
                "ingestion_succeeded",
                pages=parsed.page_count,
                chars=parsed.total_chars,
                duration_ms=round((asyncio.get_running_loop().time() - started) * 1000, 2),
            )
        return True

    async def run_forever(self, stop: asyncio.Event) -> None:
        logger.info("worker_started")
        while not stop.is_set():
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

    async def _claim(self) -> tuple[uuid.UUID, uuid.UUID, str, int] | None:
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
            return job.id, document.id, document.storage_path, job.attempts

    async def _finish_succeeded(
        self, job_id: uuid.UUID, document_id: uuid.UUID, parsed: ParsedPdf
    ) -> None:
        async with self._session_factory() as session, session.begin():
            job, document = await self._load(session, job_id, document_id)
            if job is None or document is None:
                return  # document deleted while we were parsing
            await DocumentPageWriter(session).replace_pages(
                document_id, [(p.number, p.text) for p in parsed.pages]
            )
            document.page_count = parsed.page_count
            document.status = DocumentStatus.READY
            job.status = JobStatus.SUCCEEDED
            job.error_message = None
            job.finished_at = datetime.now(UTC)

    async def _finish_failed(
        self, job_id: uuid.UUID, document_id: uuid.UUID, message: str, *, retry: bool
    ) -> None:
        async with self._session_factory() as session, session.begin():
            job, document = await self._load(session, job_id, document_id)
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
        session: AsyncSession, job_id: uuid.UUID, document_id: uuid.UUID
    ) -> tuple[IngestionJob | None, Document | None]:
        job = await session.get(IngestionJob, job_id, with_for_update=True)
        document = await session.get(Document, document_id, with_for_update=True)
        return job, document


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    engine = create_engine(settings)
    storage = LocalFileStorage(settings.storage_dir)
    await storage.ensure_ready()
    worker = IngestionWorker(create_session_factory(engine), storage, settings)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Not available on Windows event loops; Ctrl+C still raises KeyboardInterrupt there.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    try:
        await worker.run_forever(stop)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
