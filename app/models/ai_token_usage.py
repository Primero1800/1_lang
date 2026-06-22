from datetime import date

from sqlalchemy import BigInteger, Date, String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, int_pk


class AiTokenUsage(Base):
    """Daily aggregated AI token usage per model, operation, and actor"""

    __tablename__ = "ai_token_usage"
    __table_args__ = (
        UniqueConstraint("model", "date", "name", "operation", name="uq_ai_token_usage"),
        {"comment": "Дневная агрегация токенов по модели, операции и актору"},
    )

    id: Mapped[int_pk]
    model: Mapped[str] = mapped_column(
        String(100),
        comment="Идентификатор модели (например mistral-large-latest)",
    )
    date: Mapped[date] = mapped_column(
        Date,
        comment="Дата агрегации",
    )
    name: Mapped[str] = mapped_column(
        String(100),
        default="system",
        server_default=text("'system'"),
        comment="Актор (system, user и т.п.)",
    )
    operation: Mapped[str] = mapped_column(
        String(100),
        comment="Название операции (w2_generate, w3_translate и т.п.)",
    )
    input_tokens: Mapped[int] = mapped_column(
        BigInteger,
        default=0,
        server_default=text("0"),
        comment="Суммарные входящие токены за день",
    )
    output_tokens: Mapped[int] = mapped_column(
        BigInteger,
        default=0,
        server_default=text("0"),
        comment="Суммарные исходящие токены за день",
    )
