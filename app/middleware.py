import time

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.common.logging import logger


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for request monitoring and unhandled exception handling"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()
        try:
            response = await call_next(request)
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"ERROR: {request.method} {request.url} failed after {process_time:.4f}s",
                exc_info=e,
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "detail": "Internal Server Error"},
            )
        process_time = time.time() - start_time
        if process_time > 1.0:
            logger.warning(
                f"SLOW: {request.method} {request.url} took {process_time:.4f}s"
            )
        return response
