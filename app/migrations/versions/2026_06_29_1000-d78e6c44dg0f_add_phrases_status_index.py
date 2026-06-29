"""add_phrases_status_updated_index

Revision ID: d78e6c44dg0f
Revises: c67d5b33cf9e
Create Date: 2026-06-29 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "d78e6c44dg0f"
down_revision: Union[str, Sequence[str], None] = "c67d5b33cf9e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_phrases_status_updated", "phrases", ["status", "updated_at"])


def downgrade() -> None:
    op.drop_index("ix_phrases_status_updated", table_name="phrases")
