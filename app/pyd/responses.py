from typing import Any

from pydantic import BaseModel, Field


class HTTPExceptionResponse(BaseModel):
    """Schema for HTTP exception response"""

    detail: str
    headers: dict[str, Any]
    status_code: int


class PhraseVariantsRequest(BaseModel):
    """Request body for single phrase variant generation"""

    phrase: str
    tag: str
    count: int = Field(default=5, ge=1, le=10)


class PhraseVariantsResponse(BaseModel):
    """Response containing generated phrase variants grouped by mood and gender"""

    original: str
    tag: str
    variants: dict[str, dict[str, list[str]]]  # {mood: {gender: [phrases]}}


class UploadImagesResponse(BaseModel):
    """Response summarising the result of an image upload pipeline batch"""

    phrases_found: int
    inserted: int
    skipped: int
    error: str | None = None


class WorkerBatchResponse(BaseModel):
    """Response summarising the result of a worker batch cycle"""

    processed: int
    failed: int
    skipped: int


class W2GenerateResponse(WorkerBatchResponse):
    """W2 variant generation batch result"""

    error: str | None = None


class W3TranslateResponse(WorkerBatchResponse):
    """W3 translation batch result"""


class W4EmbedResponse(WorkerBatchResponse):
    """W4 embedding batch result"""


class W5LoadResponse(WorkerBatchResponse):
    """W5 Qdrant load batch result"""

    upserted: int = 0
