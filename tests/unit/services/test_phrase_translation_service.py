import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.base import BaseDeps
from app.services.phrase_translation_service import PhraseTranslationService


@pytest.fixture
def phrase_translation_service() -> PhraseTranslationService:
    base_deps = MagicMock(spec=BaseDeps)
    base_deps.uow_factory = MagicMock()
    base_deps.ai_client = MagicMock()
    base_deps.ai_client2 = MagicMock()
    base_deps.vector_client = MagicMock()
    base_deps.vector_client_main = MagicMock()
    return PhraseTranslationService(base_deps=base_deps)


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
    p.tag = "behavior"
    p.original = f"phrase_{phrase_id}"
    return p


def _tone() -> dict:
    return {"male": ["a", "b", "c", "d", "e"], "female": ["f", "g", "h", "i", "j"]}


def _variants() -> dict:
    t = _tone()
    return {"A": t, "B": t, "C": t, "D": t, "E": t}


def _matched(phrase_ids: list[int]) -> dict[int, dict]:
    return {
        pid: {"translated": f"translated_{pid}", "variants": _variants()}
        for pid in phrase_ids
    }


def _w3_raw(phrase_ids: list[int]) -> str:
    results = [
        {"id": pid, "translated": f"translated_{pid}", **_variants()}
        for pid in phrase_ids
    ]
    return json.dumps({"results": results})


# --- _parse_w3_response ---


def test_parse_w3_response_valid(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    raw = _w3_raw([1, 2])
    result = phrase_translation_service._parse_w3_response(raw)
    assert set(result.keys()) == {1, 2}
    assert result[1]["translated"] == "translated_1"
    assert "A" in result[1]["variants"]
    assert "id" not in result[1]


def test_parse_w3_response_invalid_returns_empty(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    result = phrase_translation_service._parse_w3_response("not json {{ at all")
    assert result == {}


# --- w3_translate ---


@pytest.mark.asyncio
async def test_w3_translate_empty_batch_returns_skipped(
    phrase_translation_service: PhraseTranslationService, mocker
) -> None:
    mocker.patch.object(
        phrase_translation_service, "_fetch_batch", new=AsyncMock(return_value=[])
    )
    result = await phrase_translation_service.w3_translate(batch_size=5)
    assert result == {"processed": 0, "failed": 0, "skipped": 1}


@pytest.mark.asyncio
async def test_w3_translate_mistral_none_all_failed(
    phrase_translation_service: PhraseTranslationService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    mocker.patch.object(
        phrase_translation_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_translation_service,
        "_fetch_variants",
        new=AsyncMock(return_value={}),
    )
    mocker.patch.object(
        phrase_translation_service, "_call_mistral", new=AsyncMock(return_value=None)
    )

    mock_uow = _make_mock_uow()
    phrase_translation_service.uow_factory = mock_uow

    result = await phrase_translation_service.w3_translate(batch_size=5)
    assert result == {"processed": 0, "failed": 2, "skipped": 0}
    mock_uow.phrase_repository.bulk_create.assert_not_called()
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_w3_translate_success(
    phrase_translation_service: PhraseTranslationService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    mocker.patch.object(
        phrase_translation_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_translation_service,
        "_fetch_variants",
        new=AsyncMock(return_value={1: _variants(), 2: _variants()}),
    )
    mocker.patch.object(
        phrase_translation_service, "_call_mistral", new=AsyncMock(return_value="raw")
    )
    mocker.patch.object(
        phrase_translation_service,
        "_parse_w3_response",
        return_value=_matched([1, 2]),
    )

    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.bulk_create.return_value = [10, 11]
    mock_uow.phrase_repository.get_ids_by_originals.return_value = {
        "translated_1": 10,
        "translated_2": 11,
    }
    phrase_translation_service.uow_factory = mock_uow

    result = await phrase_translation_service.w3_translate(batch_size=5)
    assert result == {"processed": 2, "failed": 0, "skipped": 0}
    mock_uow.phrase_repository.bulk_create.assert_called_once()
    mock_uow.phrase_data_repository.bulk_upsert_variants.assert_called_once()
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_w3_translate_partial_match(
    phrase_translation_service: PhraseTranslationService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2), _make_phrase(3)]
    mocker.patch.object(
        phrase_translation_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_translation_service,
        "_fetch_variants",
        new=AsyncMock(return_value={}),
    )
    mocker.patch.object(
        phrase_translation_service, "_call_mistral", new=AsyncMock(return_value="raw")
    )
    mocker.patch.object(
        phrase_translation_service,
        "_parse_w3_response",
        return_value=_matched([1]),  # only id=1 returned
    )

    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.bulk_create.return_value = [10]
    mock_uow.phrase_repository.get_ids_by_originals.return_value = {"translated_1": 10}
    phrase_translation_service.uow_factory = mock_uow

    result = await phrase_translation_service.w3_translate(batch_size=5)
    assert result == {"processed": 1, "failed": 2, "skipped": 0}
    assert mock_uow.phrase_repository.update_status.call_count == 2
