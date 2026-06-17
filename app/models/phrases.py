from sqlalchemy import String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.common.enums import LangEnum, PhraseStatusEnum, TagEnum
from app.models.base import Base, int_pk


class Phrase(Base):
    """Vision-extracted observation phrase with language, tag, pipeline status, and tone variants"""

    __tablename__ = "phrases"
    __table_args__ = (
        UniqueConstraint("original", "lang", name="uq_phrases_original_lang"),
        {"comment": "Оригинальные фразы-наблюдения, полученные от vision-модели"},
    )

    id: Mapped[int_pk]
    original: Mapped[str] = mapped_column(
        String,
        comment="Фраза в lowercase без знаков препинания",
    )
    tag: Mapped[TagEnum] = mapped_column(
        String(20),
        comment="Тег наблюдения",
    )
    lang: Mapped[LangEnum] = mapped_column(
        String(2),
        default=LangEnum.RU,
        server_default=text(f"'{LangEnum.RU.value}'"),
        comment="Язык фразы",
    )
    status: Mapped[PhraseStatusEnum] = mapped_column(
        String(30),
        default=PhraseStatusEnum.DRAFT,
        server_default=text(f"'{PhraseStatusEnum.DRAFT.value}'"),
        comment="Статус обработки",
    )

    phrase_data: Mapped["PhraseData"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="phrase",
        cascade="all, delete-orphan",
    )
