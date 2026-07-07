import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from app.common.logging import log_decorator
from app.dependencies.services import get_worker_run_log_service
from app.pyd.requests import Pagination, WorkerRunLogFilter
from app.pyd.responses import PaginatedWorkerRunLogList
from app.services.worker_run_log_query_service import WorkerRunLogQueryService

router = APIRouter(
    prefix="/workers",
    tags=["Workers"],
)


@router.get(
    "/runs",
    response_model=PaginatedWorkerRunLogList,
)
@log_decorator(level=logging.INFO)
async def get_all(
    service: Annotated[WorkerRunLogQueryService, Depends(get_worker_run_log_service)],
    filters: Annotated[WorkerRunLogFilter, Depends()],
    pagination: Annotated[Pagination, Depends()],
) -> Any:
    """List worker run log records with optional filters and pagination

    :role:
        admin

    :param:
        service: worker run log service
        filters: optional query filters (worker prefix, status, started_at range)
        pagination: page and per_page

    :returns:
        result: paginated list of worker run log records with total count
    """
    return await service.list_runs(filters=filters, pagination=pagination)
