from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class PhraseItem(BaseModel):
    """Single phrase with its tag category, as extracted from a vision model response"""

    phrase: str = Field(
        description="A 5-6 word lowercase phrase describing the person. Language must match the prompt language."
    )
    tag: str = Field(
        description="One of: behavior, appearance, age, mood, posture, hairstyle"
    )


class VisionOutput(BaseModel):
    """Structured output from the W1 vision chain — flat list of all phrases across all photos"""

    phrases: list[PhraseItem] = Field(
        description=(
            "All extracted phrases across all photos. "
            "For each photo provide exactly 5 unique variants per tag (6 tags × 5 variants = 30 items per photo). "
            "Variants for the same tag must be unique across all photos in the batch."
        )
    )


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
