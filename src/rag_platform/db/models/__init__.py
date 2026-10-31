"""ORM models. Importing this package registers every table on ``Base.metadata``."""

from rag_platform.db.models.document import Document, DocumentStatus
from rag_platform.db.models.ingestion_job import IngestionJob, JobStatus
from rag_platform.db.models.user import User

__all__ = ["Document", "DocumentStatus", "IngestionJob", "JobStatus", "User"]
