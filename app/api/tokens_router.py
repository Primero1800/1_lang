import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.common.logging import log_decorator
from app.dependencies.services import get_token_usage_service
from app.pyd.requests import AITokenFilter, Pagination
from app.pyd.responses import PaginatedAiTokenUsageItemList
from app.services.token_usage_service import TokenUsageService

router = APIRouter(
    prefix="/tokens",
    tags=["Tokens"],
)


@router.get(
    "/usage",
    response_model=PaginatedAiTokenUsageItemList,
)
@log_decorator(level=logging.INFO)
async def get_all(
    service: Annotated[TokenUsageService, Depends(get_token_usage_service)],
    filters: Annotated[AITokenFilter, Depends()],
    pagination: Annotated[Pagination, Depends()],
) -> Any:
    """List AI token usage records with optional filters and aggregated totals

    :role:
        admin

    :param:
        service: token usage service
        filters: optional query filters (date range, model, name, operation prefix, exclusions)
        pagination: page and per_page

    :returns:
        result: paginated list of usage records with total count
    """
    return await service.list_usage(filters=filters, pagination=pagination)
