"""rename_token_worker_main_to_redis_token_worker

Revision ID: c67d5b33cf9e
Revises: a45b3911ae7a
Create Date: 2026-06-26 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "c67d5b33cf9e"
down_revision: Union[str, Sequence[str], None] = "a45b3911ae7a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "UPDATE worker_run_log SET worker = 'redis_token_worker' WHERE worker = 'main'"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE worker_run_log SET worker = 'main' WHERE worker = 'redis_token_worker'"
    )
