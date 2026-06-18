"""status column type changing

Revision ID: a11adac6b5f6
Revises: a1b2c3d4e5f6
Create Date: 2026-06-18 10:10:43.553428

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a11adac6b5f6'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_ENUM_NAME = "phrasestatusenum"
_ENUM_VALUES = (
    "draft",
    "generating_in_progress", "generating_done", "generating_failed",
    "translating_in_progress", "translating_done", "translating_failed",
    "embedding_in_progress", "embedding_done", "embedding_failed",
    "loading_in_progress", "loading_done", "loading_failed",
)


def upgrade() -> None:
    """Upgrade schema."""
    phrase_status_enum = sa.Enum(*_ENUM_VALUES, name=_ENUM_NAME)
    phrase_status_enum.create(op.get_bind())

    op.alter_column('phrases', 'status', server_default=None)
    op.alter_column(
        'phrases', 'status',
        existing_type=sa.VARCHAR(length=30),
        type_=phrase_status_enum,
        existing_nullable=False,
        postgresql_using=f"status::{_ENUM_NAME}"
    )
    op.alter_column('phrases', 'status', server_default=sa.text("'draft'"))


def downgrade() -> None:
    """Downgrade schema."""
    phrase_status_enum = sa.Enum(*_ENUM_VALUES, name=_ENUM_NAME)

    op.alter_column('phrases', 'status', server_default=None)
    op.alter_column(
        'phrases', 'status',
        existing_type=phrase_status_enum,
        type_=sa.VARCHAR(length=30),
        existing_nullable=False,
        postgresql_using="status::text"
    )
    op.alter_column('phrases', 'status', server_default=sa.text("'draft'::character varying"))
    phrase_status_enum.drop(op.get_bind())
