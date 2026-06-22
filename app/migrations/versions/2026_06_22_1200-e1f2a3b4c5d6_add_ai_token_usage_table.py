"""add ai_token_usage table

Revision ID: e1f2a3b4c5d6
Revises: d38ab1b2d8ce
Create Date: 2026-06-22 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, Sequence[str], None] = "d38ab1b2d8ce"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ai_token_usage",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
            comment="Уникальный идентификатор",
        ),
        sa.Column(
            "model",
            sa.String(100),
            nullable=False,
            comment="Идентификатор модели (например mistral-large-latest)",
        ),
        sa.Column(
            "date",
            sa.Date(),
            nullable=False,
            comment="Дата агрегации",
        ),
        sa.Column(
            "name",
            sa.String(100),
            server_default=sa.text("'system'"),
            nullable=False,
            comment="Актор (system, user и т.п.)",
        ),
        sa.Column(
            "operation",
            sa.String(100),
            nullable=False,
            comment="Название операции (w2_generate, w3_translate и т.п.)",
        ),
        sa.Column(
            "input_tokens",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
            comment="Суммарные входящие токены за день",
        ),
        sa.Column(
            "output_tokens",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
            comment="Суммарные исходящие токены за день",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("TIMEZONE('utc', now())"),
            nullable=False,
            comment="Время создания записи",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("TIMEZONE('utc', now())"),
            nullable=False,
            comment="Время обновления записи",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model", "date", "name", "operation", name="uq_ai_token_usage"),
        comment="Дневная агрегация токенов по модели, операции и актору",
    )


def downgrade() -> None:
    op.drop_table("ai_token_usage")
