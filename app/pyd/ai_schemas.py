from pydantic import BaseModel, Field


class PhraseItem(BaseModel):
    """Single two-part observation with its tag category, as extracted from a vision model response"""

    concrete: str = Field(
        description="A concrete 5-6 word observation about the person. Lowercase, no punctuation."
    )
    abstract: str = Field(
        description="A more abstract or figurative 5-6 word take on the same observation. Lowercase, no punctuation."
    )
    tag: str = Field(
        description="One of: behavior, appearance, age, mood, posture, hairstyle"
    )


class VisionOutput(BaseModel):
    """Phrases extracted from a single photo"""

    phrases: list[PhraseItem] = Field(
        description=(
            "Exactly 30 phrases for one photo: 5 unique variants per tag × 6 tags. "
            "Variants for the same tag must be unique across all photos in the batch."
        )
    )


class VisionBatchOutput(BaseModel):
    """Structured output from the W1 vision chain — one VisionOutput per photo"""

    photos: list[VisionOutput] = Field(
        description=(
            "One entry per photo in the batch, each containing exactly 30 phrases. "
            "The list length must equal the number of photos provided."
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


class VariantsResponse(BaseModel):
    """Concrete W2 schema for with_structured_output (avoids generic bracket name issue)"""

    results: list[PhraseVariants]


class TranslationResponse(BaseModel):
    """Concrete W3 schema for with_structured_output (avoids generic bracket name issue)"""

    results: list[TranslatedPhrase]
