from datetime import date
from typing import Annotated

from fastapi import Query
from pydantic import BaseModel, field_validator

from app.common.enums import LangEnum, MoodEnum


class TagExclusionFilters(BaseModel):
    """Tag exclusion flags for Qdrant search — excluded tags are omitted from results"""

    not_behavior: Annotated[bool, Query(default=False, description="Исключить тег behavior")]
    not_appearance: Annotated[bool, Query(default=False, description="Исключить тег appearance")]
    not_age: Annotated[bool, Query(default=False, description="Исключить тег age")]
    not_mood: Annotated[bool, Query(default=False, description="Исключить тег mood")]
    not_posture: Annotated[bool, Query(default=False, description="Исключить тег posture")]
    not_hairstyle: Annotated[bool, Query(default=False, description="Исключить тег hairstyle")]


class SearchSettings(BaseModel):
    """Search configuration: target language and desired mood tone"""

    lang: Annotated[LangEnum, Query(default=LangEnum.RU, description="Язык результатов")]
    mood: Annotated[MoodEnum, Query(default=MoodEnum.OBJECTIVE, description="Тональность комментария (A–E)")]


class Pagination(BaseModel):
    per_page: Annotated[int, Query(default=25, gt=0, examples=[10, 50], description="Выводить по...")]
    page: Annotated[int, Query(default=1, gt=0, examples=[1, 2, 5], description="Номер страницы...")]


class AITokenFilter(BaseModel):
    """Query filters for AI token usage listing"""

    date_from: Annotated[date | None, Query(default=None, description="Начало диапазона дат")]
    date_to: Annotated[date | None, Query(default=None, description="Конец диапазона дат")]
    model: Annotated[str | None, Query(default=None, description="Фильтр по модели (точное совпадение)", examples=["mistral-small-latest"])]
    name: Annotated[str | None, Query(default=None, description="Фильтр по актору", examples=["system"])]
    operation: Annotated[str | None, Query(default=None, description="Префикс операции: буква + цифра (напр. w2)", examples=["w2", "t1"])]
    exclude_input: Annotated[bool | None, Query(default=None, description="Исключить строки с input_tokens > 0")]
    exclude_output: Annotated[bool | None, Query(default=None, description="Исключить строки с output_tokens > 0")]

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 2 or not v[0].isalpha() or not v[1].isdigit():
            raise ValueError("operation must be 2 chars: first a letter, second a digit (e.g. 'w2')")
        return v.lower()
