import logging
from typing import Any

from app.common.logging import log_decorator
from app.pyd.requests import Pagination, WorkerRunLogFilter
from app.services.base import BaseService


class WorkerRunLogQueryService(BaseService):
    """Service for reading worker run log records"""

    @log_decorator(level=logging.INFO)
    async def list_runs(
        self,
        filters: WorkerRunLogFilter,
        pagination: Pagination,
    ) -> dict[str, Any]:
        """Return paginated worker run log records matching filters

        :param:
            filters: query filters (worker prefix, status, started_at range)
            pagination: page and per_page values

        :returns:
            result: dict with per_page, page, total_count and items list
        """
        rows, total_count = await self.uow.worker_run_log_repository.list_runs(  # type: ignore[union-attr]
            worker=filters.worker,
            status=filters.status,
            started_from=filters.started_from,
            started_to=filters.started_to,
            per_page=pagination.per_page,
            page=pagination.page,
        )
        items = [
            {
                "id": row.id,
                "worker": row.worker,
                "status": row.status,
                "batch_size": row.batch_size,
                "finished_at": row.finished_at,
                "result": row.result,
                "created_at": row.created_at,
            }
            for row in rows
        ]
        return {
            "per_page": pagination.per_page,
            "page": pagination.page,
            "total_count": total_count,
            "items": items,
        }
