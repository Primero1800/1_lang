from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient


def _paginated_response(items: list = None, total_count: int = 0) -> dict:
    return {
        "per_page": 25,
        "page": 1,
        "total_count": total_count,
        "items": items or [],
    }


def _run_item(
    worker: str = "w2_generate",
    status: str = "done",
    batch_size: int = 7,
) -> dict:
    return {
        "id": 1,
        "worker": worker,
        "status": status,
        "batch_size": batch_size,
        "finished_at": datetime(2026, 7, 1, tzinfo=timezone.utc).isoformat(),
        "result": None,
        "created_at": datetime(2026, 7, 1, tzinfo=timezone.utc).isoformat(),
    }


# --- GET /workers/runs ---


@pytest.mark.asyncio
async def test_get_all_returns_200(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_worker_run_log_service

    mock_service = AsyncMock()
    mock_service.list_runs.return_value = _paginated_response()
    app.dependency_overrides[get_worker_run_log_service] = lambda: mock_service

    response = await async_client.get("/workers/runs")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_all_returns_pagination_fields(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_worker_run_log_service

    mock_service = AsyncMock()
    mock_service.list_runs.return_value = _paginated_response(
        items=[_run_item()], total_count=1
    )
    app.dependency_overrides[get_worker_run_log_service] = lambda: mock_service

    response = await async_client.get("/workers/runs")
    data = response.json()
    assert data["per_page"] == 25
    assert data["page"] == 1
    assert data["total_count"] == 1
    assert len(data["items"]) == 1


@pytest.mark.asyncio
async def test_get_all_passes_pagination_to_service(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_worker_run_log_service

    mock_service = AsyncMock()
    mock_service.list_runs.return_value = _paginated_response()
    app.dependency_overrides[get_worker_run_log_service] = lambda: mock_service

    await async_client.get("/workers/runs?per_page=10&page=3")

    call_kwargs = mock_service.list_runs.call_args.kwargs
    assert call_kwargs["pagination"].per_page == 10
    assert call_kwargs["pagination"].page == 3


@pytest.mark.asyncio
async def test_get_all_passes_worker_filter_to_service(
    async_client: AsyncClient,
) -> None:
    from app.main import app
    from app.dependencies.services import get_worker_run_log_service

    mock_service = AsyncMock()
    mock_service.list_runs.return_value = _paginated_response()
    app.dependency_overrides[get_worker_run_log_service] = lambda: mock_service

    await async_client.get("/workers/runs?worker=w2")

    call_kwargs = mock_service.list_runs.call_args.kwargs
    assert call_kwargs["filters"].worker == "w2"


@pytest.mark.asyncio
async def test_get_all_passes_status_filter_to_service(
    async_client: AsyncClient,
) -> None:
    from app.main import app
    from app.dependencies.services import get_worker_run_log_service

    mock_service = AsyncMock()
    mock_service.list_runs.return_value = _paginated_response()
    app.dependency_overrides[get_worker_run_log_service] = lambda: mock_service

    await async_client.get("/workers/runs?status=done")

    call_kwargs = mock_service.list_runs.call_args.kwargs
    from app.common.enums import WorkerStatusEnum

    assert call_kwargs["filters"].status == WorkerStatusEnum.DONE


@pytest.mark.asyncio
async def test_get_all_empty_result(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_worker_run_log_service

    mock_service = AsyncMock()
    mock_service.list_runs.return_value = _paginated_response()
    app.dependency_overrides[get_worker_run_log_service] = lambda: mock_service

    response = await async_client.get("/workers/runs")
    data = response.json()
    assert data["total_count"] == 0
    assert data["items"] == []
