import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.base import BaseDeps
from app.services.phrase_embedding_service import PhraseEmbeddingService


@pytest.fixture
def phrase_embedding_service() -> PhraseEmbeddingService:
    base_deps = MagicMock(spec=BaseDeps)
    base_deps.uow_factory = MagicMock()
    base_deps.ai_client = MagicMock()
    base_deps.ai_client2 = MagicMock()
    base_deps.vector_client = MagicMock()
    base_deps.vector_client_main = MagicMock()
    return PhraseEmbeddingService(base_deps=base_deps)


def _make_mock_uow():
    mock_uow = AsyncMock()
    mock_uow.phrase_repository = AsyncMock()
    mock_uow.phrase_embedding_repository = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=None)
    return mock_uow


def _make_phrase(phrase_id: int) -> MagicMock:
    p = MagicMock()
    p.id = phrase_id
    p.original = f"phrase_{phrase_id}"
    return p


# --- _fetch_batch ---


@pytest.mark.asyncio
async def test_fetch_batch_returns_empty_when_nothing_ready(
    phrase_embedding_service: PhraseEmbeddingService,
) -> None:
    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.get_batch_for_processing.return_value = []
    phrase_embedding_service.uow_factory = mock_uow

    result = await phrase_embedding_service._fetch_batch(batch_size=10)

    assert result == []
    mock_uow.phrase_repository.update_status.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_batch_returns_batch_and_marks_in_progress(
    phrase_embedding_service: PhraseEmbeddingService,
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.get_batch_for_processing.return_value = phrases
    phrase_embedding_service.uow_factory = mock_uow

    result = await phrase_embedding_service._fetch_batch(batch_size=10)

    assert len(result) == 2
    assert result[0].id == 1
    mock_uow.phrase_repository.update_status.assert_called_once()


# --- _call_embed ---


@pytest.mark.asyncio
async def test_call_embed_returns_empty_when_no_support(
    phrase_embedding_service: PhraseEmbeddingService,
) -> None:
    phrase_embedding_service.ai_client.supports_embed = False
    result = await phrase_embedding_service._call_embed([_make_phrase(1)])
    assert result == {}


@pytest.mark.asyncio
async def test_call_embed_returns_empty_on_count_mismatch(
    phrase_embedding_service: PhraseEmbeddingService,
) -> None:
    phrase_embedding_service.ai_client.supports_embed = True
    phrase_embedding_service.ai_client.embed = AsyncMock(return_value=[[0.1, 0.2]])
    batch = [_make_phrase(1), _make_phrase(2)]  # 2 phrases but 1 vector returned
    result = await phrase_embedding_service._call_embed(batch)
    assert result == {}


# --- w4_embed ---


@pytest.mark.asyncio
async def test_w4_embed_empty_batch_returns_skipped(
    phrase_embedding_service: PhraseEmbeddingService, mocker
) -> None:
    mocker.patch.object(
        phrase_embedding_service, "_fetch_batch", new=AsyncMock(return_value=[])
    )
    result = await phrase_embedding_service.w4_embed(batch_size=10)
    assert result == {"processed": 0, "failed": 0, "skipped": 1}


@pytest.mark.asyncio
async def test_w4_embed_embed_returns_empty_all_failed(
    phrase_embedding_service: PhraseEmbeddingService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    mocker.patch.object(
        phrase_embedding_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_embedding_service, "_call_embed", new=AsyncMock(return_value={})
    )

    mock_uow = _make_mock_uow()
    phrase_embedding_service.uow_factory = mock_uow

    result = await phrase_embedding_service.w4_embed(batch_size=10)
    assert result == {"processed": 0, "failed": 2, "skipped": 0}
    mock_uow.phrase_embedding_repository.bulk_upsert_embeddings.assert_not_called()
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_w4_embed_success(
    phrase_embedding_service: PhraseEmbeddingService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    mocker.patch.object(
        phrase_embedding_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_embedding_service,
        "_call_embed",
        new=AsyncMock(return_value={1: [0.1, 0.2], 2: [0.3, 0.4]}),
    )

    mock_uow = _make_mock_uow()
    phrase_embedding_service.uow_factory = mock_uow

    result = await phrase_embedding_service.w4_embed(batch_size=10)
    assert result == {"processed": 2, "failed": 0, "skipped": 0}
    mock_uow.phrase_embedding_repository.bulk_upsert_embeddings.assert_called_once()
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_w4_embed_partial_match(
    phrase_embedding_service: PhraseEmbeddingService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2), _make_phrase(3)]
    mocker.patch.object(
        phrase_embedding_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_embedding_service,
        "_call_embed",
        new=AsyncMock(return_value={1: [0.1, 0.2]}),  # only id=1 returned
    )

    mock_uow = _make_mock_uow()
    phrase_embedding_service.uow_factory = mock_uow

    result = await phrase_embedding_service.w4_embed(batch_size=10)
    assert result == {"processed": 1, "failed": 2, "skipped": 0}
    assert mock_uow.phrase_repository.update_status.call_count == 2
