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
    IN_PROGRESS = "in_progress"
    READY = "ready"
    FAILED = "failed"
