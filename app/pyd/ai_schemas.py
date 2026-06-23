from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class PhraseItem(BaseModel):
    """Single phrase with its tag category, as extracted from a vision model response"""

    phrase: str
    tag: str


class VisionOutput(BaseModel):
    """Structured output from the W1 vision chain — flat list of all phrases across all photos"""

    phrases: list[PhraseItem]


class ToneVariants(BaseModel):
    """Phrase variants for a single tone, split by gender"""

    male: list[str]
    female: list[str]


class PhraseVariants(BaseModel):
    """All tone variants for a single phrase, keyed A-E"""

    id: int
    A: ToneVariants | None = None
    B: ToneVariants | None = None
    C: ToneVariants | None = None
    D: ToneVariants | None = None
    E: ToneVariants | None = None


class TranslatedPhrase(PhraseVariants):
    """Translated phrase text and all tone variants for a single item"""

    translated: str


class MistralResponse(BaseModel, Generic[T]):
    """Generic root response schema for Mistral calls returning a results list"""

    results: list[T]
