"""Add chunks: retrievable passages with page spans, mirrored as Qdrant points.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-31 18:31:17.365594
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chunks",
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name=op.f("fk_chunks_owner_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
    )
    op.create_index(
        "ix_chunks_document_id_ordinal", "chunks", ["document_id", "ordinal"], unique=True
    )
    op.create_index(op.f("ix_chunks_owner_id"), "chunks", ["owner_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_chunks_owner_id"), table_name="chunks")
    op.drop_index("ix_chunks_document_id_ordinal", table_name="chunks")
    op.drop_table("chunks")
