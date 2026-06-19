import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient

# --- /test_routes/t1_search ---


@pytest.mark.asyncio
async def test_t1_search_success(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_test_service_without_session

    mock_service = AsyncMock()
    mock_service.t1_search.return_value = {
        "original_1": {"tag": "behavior", "score": 0.95, "gender": "male", "phrases": ["phrase_1"]}
    }
    app.dependency_overrides[get_test_service_without_session] = lambda: mock_service

    response = await async_client.post(
        "/test_routes/t1_search",
        files={"image": ("test.jpg", b"fake image data", "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "original_1" in data
    assert data["original_1"]["tag"] == "behavior"


@pytest.mark.asyncio
async def test_t1_search_vector_db_exception_returns_503(async_client: AsyncClient) -> None:
    from app.main import app
    from app.common.exceptions import VectorDBException
    from app.dependencies.services import get_test_service_without_session

    mock_service = AsyncMock()
    mock_service.t1_search.side_effect = VectorDBException("qdrant unavailable")
    app.dependency_overrides[get_test_service_without_session] = lambda: mock_service

    response = await async_client.post(
        "/test_routes/t1_search",
        files={"image": ("test.jpg", b"fake image data", "image/jpeg")},
    )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_t1_search_empty_result_returns_200(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_test_service_without_session

    mock_service = AsyncMock()
    mock_service.t1_search.return_value = {}
    app.dependency_overrides[get_test_service_without_session] = lambda: mock_service

    response = await async_client.post(
        "/test_routes/t1_search",
        files={"image": ("test.jpg", b"fake image data", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {}
