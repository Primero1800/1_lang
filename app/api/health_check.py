import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.common.exceptions import (
    DBHealthCheckError,
    QueueHealthCheckError,
    VectorDBHealthCheckError,
)
from app.common.logging import log_decorator
from app.dependencies.services import get_health_check_service_without_session
from app.pyd.responses import HTTPExceptionResponse
from app.services.health_check_service import HealthCheckService

router = APIRouter(
    prefix="/health_check",
    tags=["Health Check"],
)


@router.get(
    "",
    status_code=200,
    responses={
        503: {"model": HTTPExceptionResponse},
    },
)
@log_decorator(level=logging.INFO)
async def health_check(
    health_check_service: Annotated[
        HealthCheckService, Depends(get_health_check_service_without_session)
    ],
) -> None:
    """Check health of all infrastructure components

    :role:
        public

    :param:
        health_check_service: service that runs DB, vector DB, and queue checks

    :raise:
        HTTPException: 503 if any component is unavailable

    :returns:
        None
    """
    try:
        return await health_check_service.check()
    except (VectorDBHealthCheckError, DBHealthCheckError, QueueHealthCheckError) as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
            headers=exc.headers,
        ) from exc
