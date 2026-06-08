from typing import Any

from pydantic import BaseModel


class HTTPExceptionResponse(BaseModel):
    """Schema for HTTP exception response"""

    detail: str
    headers: dict[str, Any]
    status_code: int
