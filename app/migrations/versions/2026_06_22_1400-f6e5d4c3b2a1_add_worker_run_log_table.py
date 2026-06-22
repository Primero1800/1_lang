"""add worker_run_log table

Revision ID: f6e5d4c3b2a1
Revises: e1f2a3b4c5d6
Create Date: 2026-06-22 14:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "f6e5d4c3b2a1"
down_revision: Union[str, None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_ENUM_NAME = "workerstatusenum"
_ENUM_VALUES = ("RUNNING", "DONE", "FAILED")


def upgrade() -> None:
    op.execute(
        "DO $$ BEGIN "
        "CREATE TYPE workerstatusenum AS ENUM ('RUNNING', 'DONE', 'FAILED'); "
        "EXCEPTION WHEN duplicate_object THEN NULL; "
        "END $$"
    )
    op.create_table(
        "worker_run_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False, comment="Уникальный идентификатор"),
        sa.Column("worker", sa.String(50), nullable=False, comment="Имя воркера (w2_generate, token_worker и т.п.)"),
        sa.Column("status", PgEnum(*_ENUM_VALUES, name=_ENUM_NAME, create_type=False), nullable=False, server_default="RUNNING", comment="Статус выполнения"),
        sa.Column("batch_size", sa.Integer(), nullable=True, comment="Количество элементов, взятых в обработку"),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True, comment="Время завершения (NULL пока выполняется)"),
        sa.Column("result", JSONB(), nullable=True, comment="Результат выполнения: счётчики, ошибки и т.п."),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("TIMEZONE('utc', now())"), nullable=False, comment="Время создания записи"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("TIMEZONE('utc', now())"), nullable=False, comment="Время обновления записи"),
        sa.PrimaryKeyConstraint("id"),
        comment="Лог запусков воркеров: старт, финиш, размер батча, результат",
    )
    op.create_index("ix_worker_run_log_worker", "worker_run_log", ["worker"])


def downgrade() -> None:
    op.drop_index("ix_worker_run_log_worker", table_name="worker_run_log")
    op.drop_table("worker_run_log")
    op.execute(f"DROP TYPE IF EXISTS {_ENUM_NAME}")
