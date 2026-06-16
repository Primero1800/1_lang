from sqlalchemy import BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, int_pk


class PhraseData(Base):
    __tablename__ = "phrase_data"
    __table_args__ = {
        "comment": "Варианты фраз по настроению и полу, staging перед загрузкой в Qdrant"
    }

    id: Mapped[int_pk]
    phrase_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("phrases.id", ondelete="CASCADE"),
        unique=True,
        comment="FK на оригинальную фразу",
    )
    variants: Mapped[dict] = mapped_column(
        JSONB,
        comment='Варианты по настроению: {"A": {"male": [...5...], "female": [...5...]}, ...}',
    )

    phrase: Mapped["Phrase"] = relationship(back_populates="phrase_data")  # noqa: F821
