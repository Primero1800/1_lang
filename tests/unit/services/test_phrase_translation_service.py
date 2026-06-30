import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from app.common.exceptions import TranslationPipelineException
from app.pyd.ai_schemas import ToneVariants, TranslatedPhrase, TranslationResponse
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
    base_deps.queue_client = AsyncMock()
    return PhraseTranslationService(base_deps=base_deps)


def _make_mock_uow() -> AsyncMock:
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


def _tone_variants() -> ToneVariants:
    return ToneVariants(
        male=["a", "b", "c", "d", "e"], female=["f", "g", "h", "i", "j"]
    )


def _translated_phrase(phrase_id: int) -> TranslatedPhrase:
    t = _tone_variants()
    return TranslatedPhrase(
        id=phrase_id, translated=f"translated_{phrase_id}", A=t, B=t, C=t, D=t, E=t
    )


def _translation_response(*phrase_ids: int) -> TranslationResponse:
    return TranslationResponse(results=[_translated_phrase(pid) for pid in phrase_ids])


def _matched(*phrase_ids: int) -> dict:
    t = _tone_variants()
    variants = {
        "A": t.model_dump(),
        "B": t.model_dump(),
        "C": t.model_dump(),
        "D": t.model_dump(),
        "E": t.model_dump(),
    }
    return {
        pid: {"translated": f"translated_{pid}", "variants": variants}
        for pid in phrase_ids
    }


# --- _fetch_batch ---


@pytest.mark.asyncio
async def test_fetch_batch_returns_empty_when_nothing_ready(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.get_first_for_processing.return_value = None
    phrase_translation_service.uow_factory = mock_uow

    result = await phrase_translation_service._fetch_batch(batch_size=5)

    assert result == []
    mock_uow.phrase_repository.get_batch_for_processing.assert_not_called()
    mock_uow.phrase_repository.update_status.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_batch_returns_batch_and_marks_in_progress(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    first = _make_phrase(1)
    rest = [_make_phrase(2), _make_phrase(3)]
    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.get_first_for_processing.return_value = first
    mock_uow.phrase_repository.get_batch_for_processing.return_value = rest
    phrase_translation_service.uow_factory = mock_uow

    result = await phrase_translation_service._fetch_batch(batch_size=5)

    assert len(result) == 3
    assert result[0].id == 1
    mock_uow.phrase_repository.update_status.assert_called_once()


# --- _fetch_variants ---


@pytest.mark.asyncio
async def test_fetch_variants_returns_map(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    record = MagicMock()
    record.phrase_id = 1
    record.variants = {"A": "x"}
    mock_uow = _make_mock_uow()
    mock_uow.phrase_data_repository.get_by_phrase_ids.return_value = [record]
    phrase_translation_service.uow_factory = mock_uow

    result = await phrase_translation_service._fetch_variants([1])

    assert result == {1: {"A": "x"}}


# --- _build_w3_message ---


@pytest.mark.asyncio
async def test_build_w3_message_returns_system_and_human(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    messages = await phrase_translation_service._build_w3_message(
        {"batch": [_make_phrase(1)], "variants": {}, "lang": "ru"}
    )
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


@pytest.mark.asyncio
async def test_build_w3_message_merges_variants_into_payload(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    p = _make_phrase(7)
    variants = {7: {"A": "tone_data"}}
    messages = await phrase_translation_service._build_w3_message(
        {"batch": [p], "variants": variants, "lang": "ru"}
    )
    payload = json.loads(messages[1].content)
    assert payload[0]["id"] == 7
    assert payload[0]["A"] == "tone_data"


# --- _fire_token_task ---


@pytest.mark.asyncio
async def test_fire_token_task_raises_when_parsed_is_none(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    with pytest.raises(TranslationPipelineException):
        await phrase_translation_service._fire_token_task({"raw": None, "parsed": None})


@pytest.mark.asyncio
async def test_fire_token_task_returns_translation_response(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    tr = _translation_response(1)
    raw_mock = MagicMock()
    raw_mock.usage_metadata = {"input_tokens": 80, "output_tokens": 40}

    result = await phrase_translation_service._fire_token_task(
        {"raw": raw_mock, "parsed": tr}
    )
    await asyncio.sleep(0)

    assert result is tr


@pytest.mark.asyncio
async def test_fire_token_task_schedules_xadd(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    tr = _translation_response(1)
    raw_mock = MagicMock()
    raw_mock.usage_metadata = {"input_tokens": 80, "output_tokens": 40}

    await phrase_translation_service._fire_token_task({"raw": raw_mock, "parsed": tr})
    await asyncio.sleep(0)

    phrase_translation_service.queue_client.xadd.assert_called_once()


# --- _parse_translations ---


@pytest.mark.asyncio
async def test_parse_translations_maps_ids_to_dicts(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    result = await phrase_translation_service._parse_translations(
        _translation_response(1, 2)
    )
    assert set(result.keys()) == {1, 2}
    assert result[1]["translated"] == "translated_1"
    assert "A" in result[1]["variants"]
    assert "id" not in result[1]


@pytest.mark.asyncio
async def test_parse_translations_skips_empty_translated(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    tr = TranslationResponse(results=[TranslatedPhrase(id=5, translated="   ")])
    result = await phrase_translation_service._parse_translations(tr)
    assert result == {}


# --- _save_translations ---


@pytest.mark.asyncio
async def test_save_translations_creates_phrases_and_variants(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    batch = [_make_phrase(1), _make_phrase(2)]
    matched = _matched(1, 2)
    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.bulk_create.return_value = [10, 11]
    mock_uow.phrase_repository.get_ids_by_originals.return_value = {
        "translated_1": 10,
        "translated_2": 11,
    }
    phrase_translation_service.uow_factory = mock_uow

    result = await phrase_translation_service._save_translations(
        matched, sent_ids={1, 2}, batch=batch, opposite_lang="en"
    )

    assert result == {"processed": 2, "failed": 0, "skipped": 0}
    mock_uow.phrase_repository.bulk_create.assert_called_once()
    mock_uow.phrase_data_repository.bulk_upsert_variants.assert_called_once()
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_save_translations_marks_missing_ids_as_failed(
    phrase_translation_service: PhraseTranslationService,
) -> None:
    batch = [_make_phrase(1), _make_phrase(2), _make_phrase(3)]
    matched = _matched(1)
    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.bulk_create.return_value = [10]
    mock_uow.phrase_repository.get_ids_by_originals.return_value = {"translated_1": 10}
    phrase_translation_service.uow_factory = mock_uow

    result = await phrase_translation_service._save_translations(
        matched, sent_ids={1, 2, 3}, batch=batch, opposite_lang="en"
    )

    assert result["processed"] == 1
    assert result["failed"] == 2
    assert mock_uow.phrase_repository.update_status.call_count == 2


# --- w3_translate ---


@pytest.mark.asyncio
async def test_w3_translate_empty_batch_returns_skipped(
    phrase_translation_service: PhraseTranslationService, mocker
) -> None:
    mocker.patch.object(
        phrase_translation_service, "_fetch_batch", new=AsyncMock(return_value=[])
    )
    result = await phrase_translation_service.w3_translate(batch_size=5)
    assert result == {"processed": 0, "failed": 0, "skipped": 0}


@pytest.mark.asyncio
async def test_w3_translate_success(
    phrase_translation_service: PhraseTranslationService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]
    tr = _translation_response(1, 2)

    async def _fake_llm(_):
        return {"raw": MagicMock(usage_metadata=None), "parsed": tr}

    mocker.patch.object(
        phrase_translation_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_translation_service,
        "_fetch_variants",
        new=AsyncMock(return_value={}),
    )
    phrase_translation_service._llm = MagicMock()
    phrase_translation_service._llm.with_structured_output.return_value = (
        RunnableLambda(_fake_llm)
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


@pytest.mark.asyncio
async def test_w3_translate_chain_failure_marks_all_failed(
    phrase_translation_service: PhraseTranslationService, mocker
) -> None:
    phrases = [_make_phrase(1), _make_phrase(2)]

    async def _fail(_):
        raise RuntimeError("model unavailable")

    mocker.patch.object(
        phrase_translation_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    mocker.patch.object(
        phrase_translation_service,
        "_fetch_variants",
        new=AsyncMock(return_value={}),
    )
    phrase_translation_service._llm = MagicMock()
    phrase_translation_service._llm.with_structured_output.return_value = (
        RunnableLambda(_fail)
    )

    mock_uow = _make_mock_uow()
    phrase_translation_service.uow_factory = mock_uow

    with pytest.raises(TranslationPipelineException):
        await phrase_translation_service.w3_translate(batch_size=5)

    mock_uow.phrase_repository.update_status.assert_called_once()
