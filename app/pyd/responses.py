from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from app.common.enums import WorkerStatusEnum


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


class WorkerBatchWithErrorResponse(WorkerBatchResponse):
    """Worker batch result with optional pipeline error field"""

    error: str | None = None


class W2GenerateResponse(WorkerBatchWithErrorResponse):
    """W2 variant generation batch result"""


class W3TranslateResponse(WorkerBatchWithErrorResponse):
    """W3 translation batch result"""


class W4EmbedResponse(WorkerBatchWithErrorResponse):
    """W4 embedding batch result"""


class W5LoadResponse(WorkerBatchResponse):
    """W5 Qdrant load batch result"""

    upserted: int = 0


class BasePaginated(BaseModel):
    """Base schema for paginated list responses"""

    model_config = ConfigDict(from_attributes=True)

    per_page: int
    page: int
    total_count: int


class AITokenUsageItem(BaseModel):
    """Single AI token usage record"""

    model: str
    date: date
    name: str
    operation: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    updated_at: datetime


class PaginatedAiTokenUsageItemList(BasePaginated):
    """Paginated list of AI token usage records"""

    items: list[AITokenUsageItem]


class AiTokenAggregatedItem(BaseModel):
    """Aggregated AI token usage row"""

    model: str | None
    name: str | None
    operation: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int


class WorkerRunLogItem(BaseModel):
    """Single worker run log record"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    worker: str
    status: WorkerStatusEnum
    batch_size: int | None
    finished_at: datetime | None
    result: dict[str, Any] | None
    created_at: datetime


class PaginatedWorkerRunLogList(BasePaginated):
    """Paginated list of worker run log records"""

    items: list[WorkerRunLogItem]
