import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


from app.common.exceptions import EmbeddingPipelineException
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
    base_deps.queue_client = AsyncMock()
    return PhraseEmbeddingService(base_deps=base_deps)


def _make_mock_uow() -> AsyncMock:
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


# --- _embed ---


@pytest.mark.asyncio
async def test_embed_returns_id_to_vector_map(
    phrase_embedding_service: PhraseEmbeddingService,
) -> None:
    batch = [_make_phrase(1), _make_phrase(2)]
    phrase_embedding_service._embeddings = MagicMock()
    phrase_embedding_service._embeddings.aembed_with_usage = AsyncMock(
        return_value=([[0.1, 0.2], [0.3, 0.4]], 80)
    )

    result = await phrase_embedding_service._embed(batch)
    await asyncio.sleep(0)

    assert result == {1: [0.1, 0.2], 2: [0.3, 0.4]}


@pytest.mark.asyncio
async def test_embed_raises_on_count_mismatch(
    phrase_embedding_service: PhraseEmbeddingService,
) -> None:
    batch = [_make_phrase(1), _make_phrase(2)]
    phrase_embedding_service._embeddings = MagicMock()
    phrase_embedding_service._embeddings.aembed_with_usage = AsyncMock(
        return_value=([[0.1, 0.2]], 40)  # 1 vector for 2 phrases
    )

    with pytest.raises(EmbeddingPipelineException):
        await phrase_embedding_service._embed(batch)


# --- _save_vectors ---


@pytest.mark.asyncio
async def test_save_vectors_persists_and_returns_counts(
    phrase_embedding_service: PhraseEmbeddingService,
) -> None:
    matched = {1: [0.1, 0.2], 2: [0.3, 0.4]}
    mock_uow = _make_mock_uow()
    phrase_embedding_service.uow_factory = mock_uow

    result = await phrase_embedding_service._save_vectors(matched, sent_ids={1, 2})

    assert result == {"processed": 2, "failed": 0, "skipped": 0}
    mock_uow.phrase_embedding_repository.bulk_upsert_embeddings.assert_called_once()
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_save_vectors_marks_missing_ids_as_failed(
    phrase_embedding_service: PhraseEmbeddingService,
) -> None:
    matched = {1: [0.1, 0.2]}
    mock_uow = _make_mock_uow()
    phrase_embedding_service.uow_factory = mock_uow

    result = await phrase_embedding_service._save_vectors(matched, sent_ids={1, 2, 3})

    assert result["processed"] == 1
    assert result["failed"] == 2
    assert mock_uow.phrase_repository.update_status.call_count == 2


# --- w4_embed ---


@pytest.mark.asyncio
async def test_w4_embed_empty_batch_returns_skipped(
    phrase_embedding_service: PhraseEmbeddingService, mocker
) -> None:
    mocker.patch.object(
        phrase_embedding_service, "_fetch_batch", new=AsyncMock(return_value=[])
    )
    result = await phrase_embedding_service.w4_embed(batch_size=10)
    assert result == {"processed": 0, "failed": 0, "skipped": 0}


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
        "_embed",
        new=AsyncMock(return_value={1: [0.1, 0.2], 2: [0.3, 0.4]}),
    )

    mock_uow = _make_mock_uow()
    phrase_embedding_service.uow_factory = mock_uow

    result = await phrase_embedding_service.w4_embed(batch_size=10)

    assert result == {"processed": 2, "failed": 0, "skipped": 0}
    mock_uow.phrase_embedding_repository.bulk_upsert_embeddings.assert_called_once()
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_w4_embed_chain_failure_marks_all_failed(
    phrase_embedding_service: PhraseEmbeddingService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    mocker.patch.object(
        phrase_embedding_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_embedding_service,
        "_embed",
        new=AsyncMock(side_effect=RuntimeError("API down")),
    )

    mock_uow = _make_mock_uow()
    phrase_embedding_service.uow_factory = mock_uow

    with pytest.raises(EmbeddingPipelineException):
        await phrase_embedding_service.w4_embed(batch_size=10)

    mock_uow.phrase_repository.update_status.assert_called_once()
