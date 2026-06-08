import re
from typing import Any

from fastapi import status


class BaseCustomException(Exception):
    """Base class for all custom application exceptions"""

    def __init__(
        self,
        detail: str | dict[str, Any],
        status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


class IntegrityDataException(BaseCustomException):
    """Raised on PostgreSQL integrity constraint violations"""

    def __init__(
        self,
        detail: str,
        status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE,
        headers: dict[str, str] | None = None,
    ) -> None:
        detail_text = re.search(r"DETAIL: (.+)", detail or "")
        refined_message = detail_text.group(1).strip() if detail_text else detail
        super().__init__(
            detail=refined_message, status_code=status_code, headers=headers
        )


class DBHealthCheckError(BaseCustomException):
    """Raised when the database health check fails"""


class ConnectionException(BaseCustomException):
    """Raised when an external HTTP request fails"""
