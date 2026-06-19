import pytest
from unittest.mock import AsyncMock, MagicMock

from qdrant_client.models import PointStruct

from app.services.base import BaseDeps
from app.repositories.phrase_vector_repository import PhraseVectorRepository
from app.services.phrase_loading_service import PhraseLoadingService


@pytest.fixture
def phrase_loading_service() -> PhraseLoadingService:
    base_deps = MagicMock(spec=BaseDeps)
    base_deps.uow_factory = MagicMock()
    base_deps.ai_client = MagicMock()
    base_deps.ai_client2 = MagicMock()
    base_deps.vector_client = MagicMock()
    base_deps.vector_client_main = MagicMock()
    loading_repository = MagicMock(spec=PhraseVectorRepository)
    return PhraseLoadingService(
        base_deps=base_deps, loading_repository=loading_repository
    )


def _make_mock_uow():
    mock_uow = AsyncMock()
    mock_uow.phrase_repository = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=None)
    return mock_uow


def _make_phrase(phrase_id: int) -> MagicMock:
    p = MagicMock()
    p.id = phrase_id
    p.original = f"phrase_{phrase_id}"
    p.tag = "behavior"
    p.lang = "ru"
    return p


# --- _build_points ---


def test_build_points_all_present(
    phrase_loading_service: PhraseLoadingService,
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    embeddings_map = {1: [0.1, 0.2], 2: [0.3, 0.4]}
    variants_map = {1: {"A": {}}, 2: {"A": {}}}

    points, failed_ids = phrase_loading_service._build_points(
        phrases, embeddings_map, variants_map
    )

    assert len(points) == 2
    assert failed_ids == set()
    assert all(isinstance(p, PointStruct) for p in points)


def test_build_points_missing_embedding(
    phrase_loading_service: PhraseLoadingService,
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    embeddings_map = {1: [0.1, 0.2]}  # id=2 missing
    variants_map = {1: {"A": {}}, 2: {"A": {}}}

    points, failed_ids = phrase_loading_service._build_points(
        phrases, embeddings_map, variants_map
    )

    assert len(points) == 1
    assert 2 in failed_ids


def test_build_points_missing_variants(
    phrase_loading_service: PhraseLoadingService,
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    embeddings_map = {1: [0.1, 0.2], 2: [0.3, 0.4]}
    variants_map = {1: {"A": {}}}  # id=2 missing

    points, failed_ids = phrase_loading_service._build_points(
        phrases, embeddings_map, variants_map
    )

    assert len(points) == 1
    assert 2 in failed_ids


# --- w5_load ---


@pytest.mark.asyncio
async def test_w5_load_empty_batch_returns_skipped(
    phrase_loading_service: PhraseLoadingService, mocker
) -> None:
    mocker.patch.object(
        phrase_loading_service,
        "_fetch_batch",
        new=AsyncMock(return_value=([], {}, {})),
    )
    result = await phrase_loading_service.w5_load(batch_size=10)
    assert result == {"processed": 0, "failed": 0, "skipped": 1}


@pytest.mark.asyncio
async def test_w5_load_success(
    phrase_loading_service: PhraseLoadingService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    mocker.patch.object(
        phrase_loading_service,
        "_fetch_batch",
        new=AsyncMock(
            return_value=(
                phrases,
                {1: [0.1, 0.2], 2: [0.3, 0.4]},
                {1: {"A": {}}, 2: {"A": {}}},
            )
        ),
    )
    phrase_loading_service.loading_repository.bulk_upsert = AsyncMock(
        return_value=(2, set())
    )

    mock_uow = _make_mock_uow()
    phrase_loading_service.uow_factory = mock_uow

    result = await phrase_loading_service.w5_load(batch_size=10)
    assert result["processed"] == 2
    assert result["failed"] == 0
    assert result["skipped"] == 0
    assert result["upserted"] == 2
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_w5_load_upsert_returns_zero_all_failed(
    phrase_loading_service: PhraseLoadingService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    mocker.patch.object(
        phrase_loading_service,
        "_fetch_batch",
        new=AsyncMock(
            return_value=(
                phrases,
                {1: [0.1, 0.2], 2: [0.3, 0.4]},
                {1: {"A": {}}, 2: {"A": {}}},
            )
        ),
    )
    phrase_loading_service.loading_repository.bulk_upsert = AsyncMock(
        return_value=(0, {1, 2})
    )

    mock_uow = _make_mock_uow()
    phrase_loading_service.uow_factory = mock_uow

    result = await phrase_loading_service.w5_load(batch_size=10)
    assert result["processed"] == 0
    assert result["failed"] == 2
    assert result["upserted"] == 0
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_w5_load_partial_build_failure(
    phrase_loading_service: PhraseLoadingService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2), _make_phrase(3)]
    mocker.patch.object(
        phrase_loading_service,
        "_fetch_batch",
        new=AsyncMock(
            return_value=(
                phrases,
                {1: [0.1, 0.2], 2: [0.3, 0.4]},  # id=3 missing embedding
                {1: {"A": {}}, 2: {"A": {}}, 3: {"A": {}}},
            )
        ),
    )
    phrase_loading_service.loading_repository.bulk_upsert = AsyncMock(
        return_value=(2, set())
    )

    mock_uow = _make_mock_uow()
    phrase_loading_service.uow_factory = mock_uow

    result = await phrase_loading_service.w5_load(batch_size=10)
    assert result["processed"] == 2
    assert result["failed"] == 1
    assert mock_uow.phrase_repository.update_status.call_count == 2
