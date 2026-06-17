import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.base import BaseDeps
from app.services.phrase_data_service import PhraseDataService


@pytest.fixture
def phrase_data_service() -> PhraseDataService:
    base_deps = MagicMock(spec=BaseDeps)
    base_deps.uow_factory = MagicMock()
    base_deps.ai_client = MagicMock()
    base_deps.ai_client2 = MagicMock()
    base_deps.vector_client = MagicMock()
    return PhraseDataService(base_deps=base_deps)


def _make_mock_uow():
    mock_uow = AsyncMock()
    mock_uow.phrase_repository = AsyncMock()
    mock_uow.phrase_data_repository = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=None)
    return mock_uow


def _make_phrase(phrase_id: int, lang: str = "ru") -> MagicMock:
    p = MagicMock()
    p.id = phrase_id
    p.lang = lang
    return p


def _tone() -> dict:
    return {"male": ["a", "b", "c", "d", "e"], "female": ["f", "g", "h", "i", "j"]}


def _variants() -> dict:
    t = _tone()
    return {"A": t, "B": t, "C": t, "D": t, "E": t}


def _w2_raw(phrase_ids: list[int]) -> str:
    results = [{"id": pid, **_variants()} for pid in phrase_ids]
    return json.dumps({"results": results})


# --- _parse_w2_response ---


def test_parse_w2_response_valid(phrase_data_service: PhraseDataService) -> None:
    raw = _w2_raw([1, 2])
    result = phrase_data_service._parse_w2_response(raw)
    assert set(result.keys()) == {1, 2}
    assert "A" in result[1]
    assert "id" not in result[1]


def test_parse_w2_response_invalid_returns_empty(
    phrase_data_service: PhraseDataService,
) -> None:
    result = phrase_data_service._parse_w2_response("not json {{ at all")
    assert result == {}


# --- w2_generate ---


@pytest.mark.asyncio
async def test_w2_generate_empty_batch_returns_skipped(
    phrase_data_service: PhraseDataService, mocker
) -> None:
    mocker.patch.object(
        phrase_data_service, "_fetch_batch", new=AsyncMock(return_value=[])
    )
    result = await phrase_data_service.w2_generate(batch_size=7)
    assert result == {"processed": 0, "failed": 0, "skipped": 1}


@pytest.mark.asyncio
async def test_w2_generate_mistral_none_all_failed(
    phrase_data_service: PhraseDataService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    mocker.patch.object(
        phrase_data_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_data_service, "_call_mistral", new=AsyncMock(return_value=None)
    )

    mock_uow = _make_mock_uow()
    phrase_data_service.uow_factory = mock_uow

    result = await phrase_data_service.w2_generate(batch_size=7)
    assert result == {"processed": 0, "failed": 2, "skipped": 0}
    mock_uow.phrase_data_repository.bulk_upsert_variants.assert_not_called()
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_w2_generate_success(
    phrase_data_service: PhraseDataService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    mocker.patch.object(
        phrase_data_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_data_service, "_call_mistral", new=AsyncMock(return_value="raw")
    )
    mocker.patch.object(
        phrase_data_service,
        "_parse_w2_response",
        return_value={1: _variants(), 2: _variants()},
    )

    mock_uow = _make_mock_uow()
    phrase_data_service.uow_factory = mock_uow

    result = await phrase_data_service.w2_generate(batch_size=7)
    assert result == {"processed": 2, "failed": 0, "skipped": 0}
    mock_uow.phrase_data_repository.bulk_upsert_variants.assert_called_once()
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_w2_generate_partial_match(
    phrase_data_service: PhraseDataService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2), _make_phrase(3)]
    mocker.patch.object(
        phrase_data_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_data_service, "_call_mistral", new=AsyncMock(return_value="raw")
    )
    mocker.patch.object(
        phrase_data_service,
        "_parse_w2_response",
        return_value={1: _variants()},  # only id=1 returned
    )

    mock_uow = _make_mock_uow()
    phrase_data_service.uow_factory = mock_uow

    result = await phrase_data_service.w2_generate(batch_size=7)
    assert result == {"processed": 1, "failed": 2, "skipped": 0}
    assert mock_uow.phrase_repository.update_status.call_count == 2
