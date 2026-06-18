import pytest
from unittest.mock import AsyncMock, MagicMock

from qdrant_client.models import PointStruct

from app.repositories.phrase_loading_repository import PhraseLoadingRepository


@pytest.fixture
def mock_vector_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def loading_repo(mock_vector_client: AsyncMock) -> PhraseLoadingRepository:
    return PhraseLoadingRepository(vector_client=mock_vector_client)


def _make_point(phrase_id: int) -> PointStruct:
    return PointStruct(
        id=phrase_id,
        vector=[0.1, 0.2],
        payload={"original": f"phrase_{phrase_id}"},
    )


@pytest.mark.asyncio
async def test_bulk_upsert_empty_returns_zero(
    loading_repo: PhraseLoadingRepository, mock_vector_client: AsyncMock
) -> None:
    count, failed_ids = await loading_repo.bulk_upsert([])
    assert count == 0
    assert failed_ids == set()
    mock_vector_client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_bulk_upsert_returns_count_on_success(
    loading_repo: PhraseLoadingRepository, mock_vector_client: AsyncMock
) -> None:
    mock_vector_client.upsert.return_value = MagicMock()  # non-None → success
    points = [_make_point(i) for i in range(3)]

    count, failed_ids = await loading_repo.bulk_upsert(points)

    assert count == 3
    assert failed_ids == set()
    mock_vector_client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_bulk_upsert_client_returns_none_adds_failed_ids(
    loading_repo: PhraseLoadingRepository, mock_vector_client: AsyncMock
) -> None:
    mock_vector_client.upsert.return_value = None
    points = [_make_point(1), _make_point(2)]

    count, failed_ids = await loading_repo.bulk_upsert(points)

    assert count == 0
    assert failed_ids == {1, 2}
