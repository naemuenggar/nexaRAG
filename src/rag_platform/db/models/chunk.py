import uuid

from sqlalchemy import ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from rag_platform.db.base import Base, UUIDPrimaryKeyMixin


class Chunk(UUIDPrimaryKeyMixin, Base):
    """A retrievable passage. The id doubles as the Qdrant point id.

    ``owner_id`` is denormalised from the document so retrieval can filter by owner without a
    join, in SQL and in the vector index alike.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_id_ordinal", "document_id", "ordinal", unique=True),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
