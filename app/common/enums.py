from enum import Enum


class TagEnum(str, Enum):
    """Observation categories used to tag vision model output"""

    BEHAVIOR = "behavior"
    APPEARANCE = "appearance"
    AGE = "age"
    MOOD = "mood"
    POSTURE = "posture"
    HAIRSTYLE = "hairstyle"


class MoodEnum(str, Enum):
    """Tone mood levels for phrase variant selection — maps to Qdrant payload keys A–E"""

    CYNIC = "A"
    DIRECT = "B"
    OBJECTIVE = "C"
    FRIENDLY = "D"
    JOYFUL = "E"


class LangEnum(str, Enum):
    """Supported phrase languages"""

    RU = "ru"
    EN = "en"


class WorkerStatusEnum(str, Enum):
    """Execution statuses for worker batch runs"""

    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class PhraseStatusEnum(str, Enum):
    """Processing pipeline statuses for phrases"""

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


class WorkerRoleEnum(str, Enum):
    """Pipeline worker roles for W2-W5 dispatch tasks"""

    W2 = "w2"
    W3 = "w3"
    W4 = "w4"
    W5 = "w5"
