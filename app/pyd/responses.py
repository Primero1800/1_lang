from typing import Any

from pydantic import BaseModel, Field


class HTTPExceptionResponse(BaseModel):
    """Schema for HTTP exception response"""

    detail: str
    headers: dict[str, Any]
    status_code: int


class PhraseVariantsRequest(BaseModel):
    phrase: str
    tag: str
    count: int = Field(default=5, ge=1, le=10)


class PhraseVariantsResponse(BaseModel):
    original: str
    tag: str
    variants: dict[str, dict[str, list[str]]]  # {mood: {gender: [phrases]}}
