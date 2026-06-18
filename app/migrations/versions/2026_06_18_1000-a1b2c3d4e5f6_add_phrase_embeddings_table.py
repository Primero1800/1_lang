"""add_phrase_embeddings_table

Revision ID: a1b2c3d4e5f6
Revises: df19b82e52c4
Create Date: 2026-06-18 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "df19b82e52c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "phrase_embeddings",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
            comment="Уникальный идентификатор",
        ),
        sa.Column(
            "phrase_id",
            sa.BigInteger(),
            nullable=False,
            comment="FK на оригинальную фразу",
        ),
        sa.Column(
            "embedding",
            postgresql.ARRAY(sa.REAL()),
            nullable=False,
            comment="Вектор эмбеддинга (1024 float32)",
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
        sa.ForeignKeyConstraint(["phrase_id"], ["phrases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("phrase_id"),
        comment="Эмбеддинги фраз, staging перед загрузкой в Qdrant",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("phrase_embeddings")
