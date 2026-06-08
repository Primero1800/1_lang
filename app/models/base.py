from datetime import datetime
from typing import Annotated

from sqlalchemy import DateTime, text
from sqlalchemy.orm import mapped_column, DeclarativeBase, Mapped

int_pk = Annotated[
    int,
    mapped_column(
        primary_key=True, autoincrement=True, comment="Уникальный идентификатор"
    ),
]

created_dt = Annotated[
    datetime,
    mapped_column(
        DateTime(timezone=True),
        server_default=text("TIMEZONE('utc', now())"),
        comment="Время создания записи",
    ),
]

updated_dt = Annotated[
    datetime,
    mapped_column(
        DateTime(timezone=True),
        server_default=text("TIMEZONE('utc', now())"),
        onupdate=text("TIMEZONE('utc', now())"),
        comment="Время обновления записи",
    ),
]


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models"""

    created_at: Mapped[created_dt]
    updated_at: Mapped[updated_dt]
