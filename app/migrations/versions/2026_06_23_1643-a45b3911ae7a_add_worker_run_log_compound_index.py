"""add_worker_run_log_compound_index

Revision ID: a45b3911ae7a
Revises: f6e5d4c3b2a1
Create Date: 2026-06-23 16:43:14.458040

"""

from typing import Sequence, Union

from alembic import op

revision: str = "a45b3911ae7a"
down_revision: Union[str, Sequence[str], None] = "f6e5d4c3b2a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_worker_run_log_worker_status",
        "worker_run_log",
        ["worker", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_worker_run_log_worker_status", table_name="worker_run_log")
