"""lang values to uppercase

Revision ID: c2d3e4f5a6b1
Revises: fb21c6c1312c
Create Date: 2026-06-19 13:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "c2d3e4f5a6b1"
down_revision: Union[str, Sequence[str], None] = "fb21c6c1312c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE phrases SET lang = UPPER(lang)")


def downgrade() -> None:
    op.execute("UPDATE phrases SET lang = LOWER(lang)")
