from datetime import date, datetime, timezone
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


def _token_item(
    model: str = "mistral-large-latest",
    operation: str = "w2_generate",
    name: str = "system",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> dict:
    return {
        "model": model,
        "date": date(2026, 6, 1).isoformat(),
        "name": name,
        "operation": operation,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "updated_at": datetime(2026, 6, 1, tzinfo=timezone.utc).isoformat(),
    }


# --- GET /tokens/usage ---


@pytest.mark.asyncio
async def test_get_all_returns_200(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_token_usage_service

    mock_service = AsyncMock()
    mock_service.list_usage.return_value = _paginated_response()
    app.dependency_overrides[get_token_usage_service] = lambda: mock_service

    response = await async_client.get("/tokens/usage")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_all_returns_pagination_fields(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_token_usage_service

    mock_service = AsyncMock()
    mock_service.list_usage.return_value = _paginated_response(
        items=[_token_item()], total_count=1
    )
    app.dependency_overrides[get_token_usage_service] = lambda: mock_service

    response = await async_client.get("/tokens/usage")
    data = response.json()
    assert data["per_page"] == 25
    assert data["page"] == 1
    assert data["total_count"] == 1
    assert len(data["items"]) == 1


@pytest.mark.asyncio
async def test_get_all_passes_pagination_to_service(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_token_usage_service

    mock_service = AsyncMock()
    mock_service.list_usage.return_value = _paginated_response()
    app.dependency_overrides[get_token_usage_service] = lambda: mock_service

    await async_client.get("/tokens/usage?per_page=10&page=3")

    call_kwargs = mock_service.list_usage.call_args.kwargs
    assert call_kwargs["pagination"].per_page == 10
    assert call_kwargs["pagination"].page == 3


@pytest.mark.asyncio
async def test_get_all_passes_model_filter_to_service(
    async_client: AsyncClient,
) -> None:
    from app.main import app
    from app.dependencies.services import get_token_usage_service

    mock_service = AsyncMock()
    mock_service.list_usage.return_value = _paginated_response()
    app.dependency_overrides[get_token_usage_service] = lambda: mock_service

    await async_client.get("/tokens/usage?model=mistral-embed")

    call_kwargs = mock_service.list_usage.call_args.kwargs
    assert call_kwargs["filters"].model == "mistral-embed"


@pytest.mark.asyncio
async def test_get_all_passes_operation_filter_to_service(
    async_client: AsyncClient,
) -> None:
    from app.main import app
    from app.dependencies.services import get_token_usage_service

    mock_service = AsyncMock()
    mock_service.list_usage.return_value = _paginated_response()
    app.dependency_overrides[get_token_usage_service] = lambda: mock_service

    await async_client.get("/tokens/usage?operation=w2")

    call_kwargs = mock_service.list_usage.call_args.kwargs
    assert call_kwargs["filters"].operation == "w2"


@pytest.mark.asyncio
async def test_get_all_invalid_operation_returns_error(
    async_client: AsyncClient,
) -> None:
    from app.main import app
    from app.dependencies.services import get_token_usage_service

    app.dependency_overrides[get_token_usage_service] = lambda: AsyncMock()

    response = await async_client.get("/tokens/usage?operation=invalid")
    assert response.status_code in (422, 500)


@pytest.mark.asyncio
async def test_get_all_empty_result(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_token_usage_service

    mock_service = AsyncMock()
    mock_service.list_usage.return_value = _paginated_response()
    app.dependency_overrides[get_token_usage_service] = lambda: mock_service

    response = await async_client.get("/tokens/usage")
    data = response.json()
    assert data["total_count"] == 0
    assert data["items"] == []


# --- GET /tokens/usage/aggregate ---


@pytest.mark.asyncio
async def test_get_aggregated_returns_200(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_token_usage_service

    mock_service = AsyncMock()
    mock_service.aggregate_usage.return_value = {
        "model": None,
        "name": None,
        "operation": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    app.dependency_overrides[get_token_usage_service] = lambda: mock_service

    response = await async_client.get("/tokens/usage/aggregate")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_aggregated_returns_token_sums(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_token_usage_service

    mock_service = AsyncMock()
    mock_service.aggregate_usage.return_value = {
        "model": None,
        "name": None,
        "operation": None,
        "input_tokens": 500,
        "output_tokens": 200,
        "total_tokens": 700,
    }
    app.dependency_overrides[get_token_usage_service] = lambda: mock_service

    response = await async_client.get("/tokens/usage/aggregate")
    data = response.json()
    assert data["input_tokens"] == 500
    assert data["output_tokens"] == 200
    assert data["total_tokens"] == 700


@pytest.mark.asyncio
async def test_get_aggregated_passes_filters_to_service(
    async_client: AsyncClient,
) -> None:
    from app.main import app
    from app.dependencies.services import get_token_usage_service

    mock_service = AsyncMock()
    mock_service.aggregate_usage.return_value = {
        "model": "mistral-large-latest",
        "name": None,
        "operation": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    app.dependency_overrides[get_token_usage_service] = lambda: mock_service

    await async_client.get("/tokens/usage/aggregate?model=mistral-large-latest")

    call_kwargs = mock_service.aggregate_usage.call_args.kwargs
    assert call_kwargs["filters"].model == "mistral-large-latest"


@pytest.mark.asyncio
async def test_get_aggregated_invalid_operation_returns_error(
    async_client: AsyncClient,
) -> None:
    from app.main import app
    from app.dependencies.services import get_token_usage_service

    app.dependency_overrides[get_token_usage_service] = lambda: AsyncMock()

    response = await async_client.get("/tokens/usage/aggregate?operation=bad")
    assert response.status_code in (422, 500)
