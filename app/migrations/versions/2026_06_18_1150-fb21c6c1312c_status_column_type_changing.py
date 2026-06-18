"""status column type changing

Revision ID: fb21c6c1312c
Revises: b1c2d3e4f5a6
Create Date: 2026-06-18 11:50:53.325981

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "fb21c6c1312c"
down_revision: Union[str, Sequence[str], None] = "b1c2d3e4f5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_ENUM_NAME = "phrasestatusenum"
_ENUM_VALUES = (
    "DRAFT",
    "GENERATING_IN_PROGRESS",
    "GENERATING_DONE",
    "GENERATING_FAILED",
    "TRANSLATING_IN_PROGRESS",
    "TRANSLATING_DONE",
    "TRANSLATING_FAILED",
    "EMBEDDING_IN_PROGRESS",
    "EMBEDDING_DONE",
    "EMBEDDING_FAILED",
    "LOADING_IN_PROGRESS",
    "LOADING_DONE",
    "LOADING_FAILED",
)


def upgrade() -> None:
    """Upgrade schema."""
    phrase_status_enum = sa.Enum(*_ENUM_VALUES, name=_ENUM_NAME)
    phrase_status_enum.create(op.get_bind())

    op.alter_column("phrases", "status", server_default=None)
    op.alter_column(
        "phrases",
        "status",
        existing_type=sa.VARCHAR(length=30),
        type_=phrase_status_enum,
        existing_nullable=False,
        postgresql_using=f"status::{_ENUM_NAME}",
    )
    op.alter_column("phrases", "status", server_default=sa.text("'DRAFT'"))


def downgrade() -> None:
    """Downgrade schema."""
    phrase_status_enum = sa.Enum(*_ENUM_VALUES, name=_ENUM_NAME)

    op.alter_column("phrases", "status", server_default=None)
    op.alter_column(
        "phrases",
        "status",
        existing_type=phrase_status_enum,
        type_=sa.VARCHAR(length=30),
        existing_nullable=False,
        postgresql_using="status::text",
    )
    op.alter_column(
        "phrases", "status", server_default=sa.text("'DRAFT'::character varying")
    )
    phrase_status_enum.drop(op.get_bind())
