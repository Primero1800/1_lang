from pydantic import BaseModel

from app.common.enums import LangEnum, MoodEnum


class TagExclusionFilters(BaseModel):
    """Tag exclusion flags for Qdrant search — excluded tags are omitted from results"""

    not_behavior: bool = False
    not_appearance: bool = False
    not_age: bool = False
    not_mood: bool = False
    not_posture: bool = False
    not_hairstyle: bool = False


class SearchSettings(BaseModel):
    """Search configuration: target language and desired mood tone"""

    lang: LangEnum = LangEnum.RU
    mood: MoodEnum = MoodEnum.OBJECTIVE
