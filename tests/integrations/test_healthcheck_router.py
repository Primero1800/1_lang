import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_success(async_client: AsyncClient) -> None:
    response = await async_client.get("/health_check")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_check_db_fail(async_client: AsyncClient, mocker) -> None:
    from app.uow import UnitOfWork

    mocker.patch.object(
        UnitOfWork, "__aenter__", side_effect=Exception("DB connection failed")
    )
    response = await async_client.get("/health_check")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_health_check_qdrant_fail(async_client: AsyncClient, mocker) -> None:
    from app.dependencies.infrastructure import get_vector_client
    from app.main import app

    mock_vector_client = mocker.AsyncMock(spec=["collection_exists"])
    mock_vector_client.collection_exists.side_effect = Exception(
        "Qdrant broker unavailable"
    )
    app.dependency_overrides[get_vector_client] = lambda: mock_vector_client

    response = await async_client.get("/health_check")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_health_check_queue_fail(async_client: AsyncClient, mocker) -> None:
    from app.dependencies.infrastructure import get_queue_client
    from app.main import app

    mock_queue_client = mocker.AsyncMock(spec=["check_connection"])
    mock_queue_client.check_connection.side_effect = Exception("Redis unavailable")
    app.dependency_overrides[get_queue_client] = lambda: mock_queue_client

    response = await async_client.get("/health_check")
    assert response.status_code == 503
