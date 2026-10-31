"""The single definition of "documents this user may read".

A user may read a document they own or one shared with them. Every read path (document listing,
pages, search, answers) builds its SQL from these conditions, so the rule lives in one place.
"""

import uuid

from sqlalchemy import ColumnElement, or_, select

from rag_platform.db.models import Document, DocumentShare, DocumentStatus


def readable_by(user_id: uuid.UUID) -> ColumnElement[bool]:
    shared = select(DocumentShare.document_id).where(DocumentShare.user_id == user_id)
    return or_(Document.owner_id == user_id, Document.id.in_(shared))


def searchable_by(user_id: uuid.UUID) -> tuple[ColumnElement[bool], ColumnElement[bool]]:
    """Readable and fully ingested."""
    return (readable_by(user_id), Document.status == DocumentStatus.READY)
