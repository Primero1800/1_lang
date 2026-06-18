"""convert_phrases_status_to_enum_and_add_index

Revision ID: b2c3d4e5f6a1
Revises: a1b2c3d4e5f6
Create Date: 2026-06-18 11:00:00.000000

Converts phrases.status from VARCHAR(30) to a proper PostgreSQL ENUM type
and adds a composite index on (status, id) for worker batch queries.

Migration is data-safe: values are copied via a temp column before the
original is dropped, so existing rows are preserved in both directions.

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a1"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_ENUM_NAME = "phrase_status_enum"
_ENUM_VALUES = (
    "draft",
    "generating_in_progress",
    "generating_done",
    "generating_failed",
    "translating_in_progress",
    "translating_done",
    "translating_failed",
    "embedding_in_progress",
    "embedding_done",
    "embedding_failed",
    "loading_in_progress",
    "loading_done",
    "loading_failed",
)


def upgrade() -> None:
    """Upgrade schema."""
    values_sql = ", ".join(f"'{v}'" for v in _ENUM_VALUES)

    # 1. Create PostgreSQL ENUM type
    op.execute(f"CREATE TYPE {_ENUM_NAME} AS ENUM ({values_sql})")

    # 2. Add temp column with ENUM type
    op.execute(f"ALTER TABLE phrases ADD COLUMN status_temp {_ENUM_NAME}")

    # 3. Copy existing values — cast is safe because all current values are valid enum members
    op.execute(f"UPDATE phrases SET status_temp = status::{_ENUM_NAME}")

    # 4. Apply NOT NULL and restore server default
    op.execute("ALTER TABLE phrases ALTER COLUMN status_temp SET NOT NULL")
    op.execute(
        f"ALTER TABLE phrases ALTER COLUMN status_temp SET DEFAULT 'draft'::{_ENUM_NAME}"
    )

    # 5. Preserve column comment
    op.execute("COMMENT ON COLUMN phrases.status_temp IS 'Статус обработки'")

    # 6. Drop old VARCHAR column
    op.drop_column("phrases", "status")

    # 7. Rename temp column to status
    op.execute("ALTER TABLE phrases RENAME COLUMN status_temp TO status")

    # 8. Add composite index for worker batch queries (status IN (...) + ORDER BY id)
    op.create_index("idx_phrases_status_id", "phrases", ["status", "id"])


def downgrade() -> None:
    """Downgrade schema."""
    # 1. Drop composite index
    op.drop_index("idx_phrases_status_id", table_name="phrases")

    # 2. Add temp VARCHAR column
    op.add_column(
        "phrases",
        sa.Column(
            "status_old",
            sa.String(30),
            nullable=True,
            comment="Статус обработки",
        ),
    )

    # 3. Copy values back to string
    op.execute("UPDATE phrases SET status_old = status::text")

    # 4. Apply NOT NULL and restore server default
    op.execute("ALTER TABLE phrases ALTER COLUMN status_old SET NOT NULL")
    op.execute("ALTER TABLE phrases ALTER COLUMN status_old SET DEFAULT 'draft'")

    # 5. Drop ENUM column
    op.drop_column("phrases", "status")

    # 6. Rename back to status
    op.execute("ALTER TABLE phrases RENAME COLUMN status_old TO status")

    # 7. Drop ENUM type
    op.execute(f"DROP TYPE {_ENUM_NAME}")
