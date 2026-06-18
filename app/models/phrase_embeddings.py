from sqlalchemy import BigInteger, ForeignKey, REAL
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, int_pk


class PhraseEmbedding(Base):
    """Embedding vector for a Phrase, stored as a staging step before Qdrant loading"""

    __tablename__ = "phrase_embeddings"
    __table_args__ = {"comment": "Эмбеддинги фраз, staging перед загрузкой в Qdrant"}

    id: Mapped[int_pk]
    phrase_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("phrases.id", ondelete="CASCADE"),
        unique=True,
        comment="FK на оригинальную фразу",
    )
    embedding: Mapped[list[float]] = mapped_column(
        ARRAY(REAL),
        comment="Вектор эмбеддинга (1024 float32)",
    )

    phrase: Mapped["Phrase"] = relationship(back_populates="phrase_embedding")  # type: ignore[name-defined]  # noqa: F821
