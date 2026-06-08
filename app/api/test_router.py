import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from app.common.logging import log_decorator
from app.dependencies.services import (
    get_test_service_without_session,
)
from app.services.base import BaseServiceAbstract

router = APIRouter(
    prefix="/test_routes",
    tags=["Test"],
)


@router.get(
    "",
    status_code=200,
)
@log_decorator(level=logging.INFO)
async def test(
    text: str,
    test_service: Annotated[
        BaseServiceAbstract, Depends(get_test_service_without_session)
    ],
) -> None:
    return await test_service.check(text)
