from pydantic import BaseModel


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


class W2MistralResponse(BaseModel):
    """Root response schema for the W2 Mistral variant generation call"""

    results: list[PhraseVariants]
