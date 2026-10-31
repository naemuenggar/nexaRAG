"""ORM models. Importing this package registers every table on ``Base.metadata``."""

from rag_platform.db.models.chunk import Chunk
from rag_platform.db.models.conversation import Conversation, Message, MessageRole
from rag_platform.db.models.document import Document, DocumentStatus
from rag_platform.db.models.document_page import DocumentPage
from rag_platform.db.models.document_share import DocumentShare
from rag_platform.db.models.ingestion_job import IngestionJob, JobStatus
from rag_platform.db.models.user import User

__all__ = [
    "Chunk",
    "Conversation",
    "Document",
    "DocumentPage",
    "DocumentShare",
    "DocumentStatus",
    "IngestionJob",
    "JobStatus",
    "Message",
    "MessageRole",
    "User",
]
