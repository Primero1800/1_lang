"""lang type changed

Revision ID: d38ab1b2d8ce
Revises: c2d3e4f5a6b1
Create Date: 2026-06-19 14:01:14.423153

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d38ab1b2d8ce"
down_revision: Union[str, Sequence[str], None] = "c2d3e4f5a6b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_ENUM_NAME = "langenum"
_ENUM_VALUES = (
    "RU",
    "EN",
)


def upgrade() -> None:
    """Upgrade schema."""
    lang_enum = sa.Enum(*_ENUM_VALUES, name=_ENUM_NAME)
    lang_enum.create(op.get_bind(), checkfirst=True)

    op.alter_column("phrases", "lang", server_default=None)
    op.alter_column(
        "phrases",
        "lang",
        existing_type=sa.VARCHAR(length=2),
        type_=lang_enum,
        existing_comment="Язык фразы",
        existing_nullable=False,
        postgresql_using=f"lang::{_ENUM_NAME}",
    )
    op.alter_column("phrases", "lang", server_default=sa.text("'RU'"))


def downgrade() -> None:
    """Downgrade schema."""
    lang_enum = sa.Enum(*_ENUM_VALUES, name=_ENUM_NAME)

    op.alter_column("phrases", "lang", server_default=None)
    op.alter_column(
        "phrases",
        "lang",
        existing_type=lang_enum,
        type_=sa.VARCHAR(length=2),
        existing_comment="Язык фразы",
        existing_nullable=False,
        postgresql_using="lang::text",
    )
    op.alter_column(
        "phrases", "lang", server_default=sa.text("'RU'::character varying")
    )
    lang_enum.drop(op.get_bind())
