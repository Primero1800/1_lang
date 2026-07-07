from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.common.enums import WorkerStatusEnum
from app.pyd.requests import Pagination, WorkerRunLogFilter
from app.services.worker_run_log_query_service import WorkerRunLogQueryService


def _make_row(
    worker: str = "w2_generate",
    status: WorkerStatusEnum = WorkerStatusEnum.DONE,
    batch_size: int | None = 7,
    result: dict | None = None,
) -> MagicMock:
    row = MagicMock()
    row.id = 1
    row.worker = worker
    row.status = status
    row.batch_size = batch_size
    row.finished_at = datetime(2026, 7, 1, tzinfo=timezone.utc)
    row.result = result
    row.created_at = datetime(2026, 7, 1, tzinfo=timezone.utc)
    return row


@pytest.fixture
def service() -> WorkerRunLogQueryService:
    base_deps = MagicMock()
    uow = MagicMock()
    uow.worker_run_log_repository = AsyncMock()
    return WorkerRunLogQueryService(base_deps=base_deps, uow=uow)


# --- list_runs ---


@pytest.mark.asyncio
async def test_list_runs_returns_paginated_structure(
    service: WorkerRunLogQueryService,
) -> None:
    row = _make_row()
    service.uow.worker_run_log_repository.list_runs = AsyncMock(return_value=([row], 1))

    result = await service.list_runs(
        filters=WorkerRunLogFilter(),
        pagination=Pagination(per_page=10, page=1),
    )

    assert result["per_page"] == 10
    assert result["page"] == 1
    assert result["total_count"] == 1
    assert len(result["items"]) == 1


@pytest.mark.asyncio
async def test_list_runs_maps_row_fields(service: WorkerRunLogQueryService) -> None:
    row = _make_row(worker="token_worker", status=WorkerStatusEnum.FAILED, batch_size=3)
    service.uow.worker_run_log_repository.list_runs = AsyncMock(return_value=([row], 1))

    result = await service.list_runs(
        filters=WorkerRunLogFilter(),
        pagination=Pagination(),
    )

    item = result["items"][0]
    assert item["worker"] == "token_worker"
    assert item["status"] == WorkerStatusEnum.FAILED
    assert item["batch_size"] == 3


@pytest.mark.asyncio
async def test_list_runs_passes_filters_to_repo(
    service: WorkerRunLogQueryService,
) -> None:
    service.uow.worker_run_log_repository.list_runs = AsyncMock(return_value=([], 0))

    await service.list_runs(
        filters=WorkerRunLogFilter(worker="w2", status=WorkerStatusEnum.DONE),
        pagination=Pagination(per_page=5, page=2),
    )

    service.uow.worker_run_log_repository.list_runs.assert_called_once_with(
        worker="w2",
        status=WorkerStatusEnum.DONE,
        started_from=None,
        started_to=None,
        per_page=5,
        page=2,
    )


@pytest.mark.asyncio
async def test_list_runs_returns_empty_items_when_no_rows(
    service: WorkerRunLogQueryService,
) -> None:
    service.uow.worker_run_log_repository.list_runs = AsyncMock(return_value=([], 0))

    result = await service.list_runs(
        filters=WorkerRunLogFilter(),
        pagination=Pagination(),
    )

    assert result["total_count"] == 0
    assert result["items"] == []


@pytest.mark.asyncio
async def test_list_runs_includes_result_field(
    service: WorkerRunLogQueryService,
) -> None:
    row = _make_row(result={"processed": 10, "failed": 1})
    service.uow.worker_run_log_repository.list_runs = AsyncMock(return_value=([row], 1))

    result = await service.list_runs(
        filters=WorkerRunLogFilter(),
        pagination=Pagination(),
    )

    assert result["items"][0]["result"] == {"processed": 10, "failed": 1}
