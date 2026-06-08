import logging

from fastapi import APIRouter, status

from app.api.health_check import router as healthcheck_router
from app.common.logging import log_decorator
from app.core.config import settings

router = APIRouter()


@router.get(
    "/version",
    response_model=str,
    status_code=status.HTTP_200_OK,
)
@log_decorator(logging.INFO)
async def get_version() -> str:
    """Retrieve application name and version"""
    return f"{settings.APP_NAME}, v{settings.APP_VERSION}"


router.include_router(healthcheck_router)
