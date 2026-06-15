from enum import Enum


class TagEnum(str, Enum):
    BEHAVIOR = "behavior"
    APPEARANCE = "appearance"
    AGE = "age"
    MOOD = "mood"
    POSTURE = "posture"
    HAIRSTYLE = "hairstyle"


class LangEnum(str, Enum):
    RU = "ru"
    EN = "en"


class PhraseStatusEnum(str, Enum):
    DRAFT = "draft"
    GENERATING_IN_PROGRESS = "generating_in_progress"
    GENERATING_DONE = "generating_done"
    GENERATING_FAILED = "generating_failed"
    TRANSLATING_IN_PROGRESS = "translating_in_progress"
    TRANSLATING_DONE = "translating_done"
    TRANSLATING_FAILED = "translating_failed"
    EMBEDDING_IN_PROGRESS = "embedding_in_progress"
    EMBEDDING_DONE = "embedding_done"
    EMBEDDING_FAILED = "embedding_failed"
    LOADING_IN_PROGRESS = "loading_in_progress"
    LOADING_DONE = "loading_done"
    LOADING_FAILED = "loading_failed"
