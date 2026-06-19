import pytest
from unittest.mock import AsyncMock, MagicMock

from qdrant_client.models import PointStruct

from app.repositories.phrase_vector_repository import PhraseVectorRepository


@pytest.fixture
def mock_vector_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def vector_repo(mock_vector_client: AsyncMock) -> PhraseVectorRepository:
    return PhraseVectorRepository(vector_client=mock_vector_client)


def _make_point(phrase_id: int) -> PointStruct:
    return PointStruct(
        id=phrase_id,
        vector=[0.1, 0.2],
        payload={"original": f"phrase_{phrase_id}"},
    )


@pytest.mark.asyncio
async def test_bulk_upsert_empty_returns_zero(
    vector_repo: PhraseVectorRepository, mock_vector_client: AsyncMock
) -> None:
    count, failed_ids = await vector_repo.bulk_upsert([])
    assert count == 0
    assert failed_ids == set()
    mock_vector_client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_bulk_upsert_returns_count_on_success(
    vector_repo: PhraseVectorRepository, mock_vector_client: AsyncMock
) -> None:
    mock_vector_client.upsert.return_value = MagicMock()
    points = [_make_point(i) for i in range(3)]

    count, failed_ids = await vector_repo.bulk_upsert(points)

    assert count == 3
    assert failed_ids == set()
    mock_vector_client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_search_batch_delegates_to_client(
    vector_repo: PhraseVectorRepository, mock_vector_client: AsyncMock
) -> None:
    mock_vector_client.search_batch.return_value = [[]]
    result = await vector_repo.search_batch(
        vectors=[[0.1, 0.2]],
        tags=["behavior"],
        lang="ru",
    )
    mock_vector_client.search_batch.assert_called_once()
    assert result == [[]]


@pytest.mark.asyncio
async def test_search_batch_empty_vectors_calls_with_empty_requests(
    vector_repo: PhraseVectorRepository, mock_vector_client: AsyncMock
) -> None:
    mock_vector_client.search_batch.return_value = []
    result = await vector_repo.search_batch(vectors=[], tags=[], lang="ru")
    call_kwargs = mock_vector_client.search_batch.call_args.kwargs
    assert call_kwargs["requests"] == []
    assert result == []


@pytest.mark.asyncio
async def test_search_batch_builds_one_request_per_vector(
    vector_repo: PhraseVectorRepository, mock_vector_client: AsyncMock
) -> None:
    mock_vector_client.search_batch.return_value = [[], []]
    await vector_repo.search_batch(
        vectors=[[0.1, 0.2], [0.3, 0.4]],
        tags=["behavior", "mood"],
        lang="en",
    )
    call_kwargs = mock_vector_client.search_batch.call_args.kwargs
    assert len(call_kwargs["requests"]) == 2


@pytest.mark.asyncio
async def test_bulk_upsert_exception_adds_failed_ids(
    vector_repo: PhraseVectorRepository, mock_vector_client: AsyncMock
) -> None:
    from app.common.exceptions import VectorDBException

    mock_vector_client.upsert.side_effect = VectorDBException("upsert failed")
    points = [_make_point(1), _make_point(2)]

    count, failed_ids = await vector_repo.bulk_upsert(points)

    assert count == 0
    assert failed_ids == {1, 2}
