"""PostgreSQL-backed job queue.

Jobs are rows in ``ingestion_jobs``. Enqueueing happens in the same transaction that creates the
document, so a committed upload always has a job. Workers claim jobs with
``FOR UPDATE SKIP LOCKED``, so several workers never take the same job.
"""

import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_platform.db.models import IngestionJob, JobStatus


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def enqueue(self, document_id: uuid.UUID) -> IngestionJob:
        job = IngestionJob(document_id=document_id, status=JobStatus.QUEUED, attempts=0)
        self._session.add(job)
        await self._session.flush()
        return job

    async def latest_for_document(self, document_id: uuid.UUID) -> IngestionJob | None:
        result = await self._session.execute(
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def claim_next(
        self, *, now: datetime, lease: timedelta, max_attempts: int
    ) -> IngestionJob | None:
        """Lock the oldest queued job, or a running job whose lease expired, and mark it running."""
        claimable = or_(
            IngestionJob.status == JobStatus.QUEUED,
            and_(
                IngestionJob.status == JobStatus.RUNNING,
                IngestionJob.started_at < now - lease,
            ),
        )
        result = await self._session.execute(
            select(IngestionJob)
            .where(claimable, IngestionJob.attempts < max_attempts)
            .order_by(IngestionJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = result.scalar_one_or_none()
        if job is None:
            return None
        job.status = JobStatus.RUNNING
        job.attempts += 1
        job.started_at = now
        job.finished_at = None
        await self._session.flush()
        return job

    async def fail_exhausted(
        self, *, now: datetime, lease: timedelta, max_attempts: int
    ) -> Sequence[IngestionJob]:
        """Mark jobs that crashed on every attempt as failed so they stop looking 'running'."""
        result = await self._session.execute(
            select(IngestionJob)
            .where(
                IngestionJob.status == JobStatus.RUNNING,
                IngestionJob.started_at < now - lease,
                IngestionJob.attempts >= max_attempts,
            )
            .with_for_update(skip_locked=True)
        )
        jobs = result.scalars().all()
        for job in jobs:
            job.status = JobStatus.FAILED
            job.finished_at = now
            job.error_message = "Worker did not finish within the lease on any attempt"
        await self._session.flush()
        return jobs
