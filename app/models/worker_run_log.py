from datetime import datetime

from sqlalchemy import DateTime, Enum as SqlEnum, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import WorkerStatusEnum
from app.models.base import Base, int_pk


class WorkerRunLog(Base):
    """Audit log of individual worker batch executions"""

    __tablename__ = "worker_run_log"
    __table_args__ = {"comment": "Лог запусков воркеров: старт, финиш, размер батча, результат"}

    id: Mapped[int_pk]
    worker: Mapped[str] = mapped_column(
        String(50),
        index=True,
        comment="Имя воркера (w2_generate, token_worker и т.п.)",
    )
    status: Mapped[WorkerStatusEnum] = mapped_column(
        SqlEnum(WorkerStatusEnum),
        default=WorkerStatusEnum.RUNNING,
        comment="Статус выполнения",
    )
    batch_size: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Количество элементов, взятых в обработку",
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Время завершения (NULL пока выполняется)",
    )
    result: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Результат выполнения: счётчики, ошибки и т.п.",
    )
