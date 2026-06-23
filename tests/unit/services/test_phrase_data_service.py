import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from app.common.exceptions import GenerationPipelineException
from app.pyd.ai_schemas import PhraseVariants, ToneVariants, VariantsResponse
from app.services.base import BaseDeps
from app.services.phrase_data_service import PhraseDataService


@pytest.fixture
def phrase_data_service() -> PhraseDataService:
    """
    :returns:
        service: PhraseDataService with mocked infrastructure; queue_client is AsyncMock
        so asyncio.create_task(queue_client.xadd(...)) works in _fire_token_task
    """
    base_deps = MagicMock(spec=BaseDeps)
    base_deps.uow_factory = MagicMock()
    base_deps.ai_client = MagicMock()
    base_deps.ai_client2 = MagicMock()
    base_deps.vector_client = MagicMock()
    base_deps.vector_client_main = MagicMock()
    base_deps.queue_client = AsyncMock()
    return PhraseDataService(base_deps=base_deps)


def _make_mock_uow() -> AsyncMock:
    mock_uow = AsyncMock()
    mock_uow.phrase_repository = AsyncMock()
    mock_uow.phrase_data_repository = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=None)
    return mock_uow


def _make_phrase(phrase_id: int) -> MagicMock:
    p = MagicMock()
    p.id = phrase_id
    p.lang = "ru"
    p.original = f"phrase {phrase_id}"
    p.tag = "behavior"
    return p


def _tone_variants() -> ToneVariants:
    return ToneVariants(
        male=["a", "b", "c", "d", "e"], female=["f", "g", "h", "i", "j"]
    )


def _phrase_variants(phrase_id: int) -> PhraseVariants:
    t = _tone_variants()
    return PhraseVariants(id=phrase_id, A=t, B=t, C=t, D=t, E=t)


def _variants_response(*phrase_ids: int) -> VariantsResponse:
    return VariantsResponse(results=[_phrase_variants(pid) for pid in phrase_ids])


# --- _fetch_batch ---


@pytest.mark.asyncio
async def test_fetch_batch_returns_empty_when_nothing_ready(
    phrase_data_service: PhraseDataService,
) -> None:
    """
    :param:
        phrase_data_service: service fixture

    :returns:
        None
    """
    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.get_first_for_processing.return_value = None
    phrase_data_service.uow_factory = mock_uow

    result = await phrase_data_service._fetch_batch(batch_size=5)

    assert result == []
    mock_uow.phrase_repository.get_batch_for_processing.assert_not_called()
    mock_uow.phrase_repository.update_status.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_batch_returns_batch_and_marks_in_progress(
    phrase_data_service: PhraseDataService,
) -> None:
    """
    :param:
        phrase_data_service: service fixture

    :returns:
        None
    """
    first = _make_phrase(1)
    rest = [_make_phrase(2), _make_phrase(3)]
    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.get_first_for_processing.return_value = first
    mock_uow.phrase_repository.get_batch_for_processing.return_value = rest
    phrase_data_service.uow_factory = mock_uow

    result = await phrase_data_service._fetch_batch(batch_size=5)

    assert len(result) == 3
    assert result[0].id == 1
    mock_uow.phrase_repository.update_status.assert_called_once()


# --- _build_w2_message ---


@pytest.mark.asyncio
async def test_build_w2_message_returns_system_and_human(
    phrase_data_service: PhraseDataService,
) -> None:
    """
    :param:
        phrase_data_service: service fixture

    :returns:
        None
    """
    messages = await phrase_data_service._build_w2_message(
        {"batch": [_make_phrase(1)], "lang": "ru"}
    )
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


@pytest.mark.asyncio
async def test_build_w2_message_payload_contains_phrase_id(
    phrase_data_service: PhraseDataService,
) -> None:
    """HumanMessage content must be valid JSON with id, phrase, and tag fields

    :param:
        phrase_data_service: service fixture

    :returns:
        None
    """
    p = _make_phrase(42)
    p.original = "walks fast"
    p.tag = "behavior"
    messages = await phrase_data_service._build_w2_message({"batch": [p], "lang": "ru"})
    payload = json.loads(messages[1].content)
    assert payload[0]["id"] == 42
    assert payload[0]["phrase"] == "walks fast"
    assert payload[0]["tag"] == "behavior"


# --- _fire_token_task ---


@pytest.mark.asyncio
async def test_fire_token_task_raises_when_parsed_is_none(
    phrase_data_service: PhraseDataService,
) -> None:
    """
    :param:
        phrase_data_service: service fixture

    :returns:
        None
    """
    with pytest.raises(GenerationPipelineException):
        await phrase_data_service._fire_token_task({"raw": None, "parsed": None})


@pytest.mark.asyncio
async def test_fire_token_task_returns_variants_response(
    phrase_data_service: PhraseDataService,
) -> None:
    """
    :param:
        phrase_data_service: service fixture

    :returns:
        None
    """
    vr = _variants_response(1)
    raw_mock = MagicMock()
    raw_mock.usage_metadata = {"input_tokens": 50, "output_tokens": 30}

    result = await phrase_data_service._fire_token_task({"raw": raw_mock, "parsed": vr})
    await asyncio.sleep(0)

    assert result is vr


@pytest.mark.asyncio
async def test_fire_token_task_schedules_xadd(
    phrase_data_service: PhraseDataService,
) -> None:
    """queue_client.xadd must be called once with token usage payload

    :param:
        phrase_data_service: service fixture

    :returns:
        None
    """
    vr = _variants_response(1)
    raw_mock = MagicMock()
    raw_mock.usage_metadata = {"input_tokens": 50, "output_tokens": 30}

    await phrase_data_service._fire_token_task({"raw": raw_mock, "parsed": vr})
    await asyncio.sleep(0)

    phrase_data_service.queue_client.xadd.assert_called_once()


# --- _parse_variants ---


@pytest.mark.asyncio
async def test_parse_variants_maps_ids_to_dicts(
    phrase_data_service: PhraseDataService,
) -> None:
    """
    :param:
        phrase_data_service: service fixture

    :returns:
        None
    """
    result = await phrase_data_service._parse_variants(_variants_response(1, 2))
    assert set(result.keys()) == {1, 2}
    assert "A" in result[1]
    assert "id" not in result[1]


@pytest.mark.asyncio
async def test_parse_variants_skips_all_none_entry(
    phrase_data_service: PhraseDataService,
) -> None:
    """Phrase with all-None tone slots must be excluded from the result

    :param:
        phrase_data_service: service fixture

    :returns:
        None
    """
    vr = VariantsResponse(results=[PhraseVariants(id=99)])
    result = await phrase_data_service._parse_variants(vr)
    assert result == {}


# --- _save_results ---


@pytest.mark.asyncio
async def test_save_results_persists_and_returns_counts(
    phrase_data_service: PhraseDataService,
) -> None:
    """
    :param:
        phrase_data_service: service fixture

    :returns:
        None
    """
    matched = {1: {"A": "x"}, 2: {"A": "y"}}
    mock_uow = _make_mock_uow()
    phrase_data_service.uow_factory = mock_uow

    result = await phrase_data_service._save_results(matched, sent_ids={1, 2})

    assert result == {"processed": 2, "failed": 0, "skipped": 0}
    mock_uow.phrase_data_repository.bulk_upsert_variants.assert_called_once()
    mock_uow.phrase_repository.update_status.assert_called_once()


@pytest.mark.asyncio
async def test_save_results_marks_missing_ids_as_failed(
    phrase_data_service: PhraseDataService,
) -> None:
    """IDs sent but absent from matched must be marked GENERATING_FAILED

    :param:
        phrase_data_service: service fixture

    :returns:
        None
    """
    matched = {1: {"A": "x"}}
    mock_uow = _make_mock_uow()
    phrase_data_service.uow_factory = mock_uow

    result = await phrase_data_service._save_results(matched, sent_ids={1, 2, 3})

    assert result["processed"] == 1
    assert result["failed"] == 2
    assert mock_uow.phrase_repository.update_status.call_count == 2


# --- w2_generate ---


@pytest.mark.asyncio
async def test_w2_generate_empty_batch_returns_skipped(
    phrase_data_service: PhraseDataService, mocker
) -> None:
    """
    :param:
        phrase_data_service: service fixture
        mocker: pytest-mock fixture

    :returns:
        None
    """
    mocker.patch.object(
        phrase_data_service, "_fetch_batch", new=AsyncMock(return_value=[])
    )
    result = await phrase_data_service.w2_generate(batch_size=7)
    assert result == {"processed": 0, "failed": 0, "skipped": 1}


@pytest.mark.asyncio
async def test_w2_generate_success(
    phrase_data_service: PhraseDataService, mocker
) -> None:
    """Full chain: fake LLM returns structured output; end-to-end result is correct

    :param:
        phrase_data_service: service fixture
        mocker: pytest-mock fixture

    :returns:
        None
    """
    phrases = [_make_phrase(1), _make_phrase(2)]
    vr = _variants_response(1, 2)

    async def _fake_llm(_):
        return {"raw": MagicMock(usage_metadata=None), "parsed": vr}

    mocker.patch.object(
        phrase_data_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    phrase_data_service._llm = MagicMock()
    phrase_data_service._llm.with_structured_output.return_value = RunnableLambda(
        _fake_llm
    )

    mock_uow = _make_mock_uow()
    phrase_data_service.uow_factory = mock_uow

    result = await phrase_data_service.w2_generate(batch_size=7)

    assert result == {"processed": 2, "failed": 0, "skipped": 0}
    mock_uow.phrase_data_repository.bulk_upsert_variants.assert_called_once()


@pytest.mark.asyncio
async def test_w2_generate_chain_failure_marks_all_failed(
    phrase_data_service: PhraseDataService, mocker
) -> None:
    """Chain error must mark all sent phrases as GENERATING_FAILED and raise GenerationPipelineException

    :param:
        phrase_data_service: service fixture
        mocker: pytest-mock fixture

    :returns:
        None
    """
    phrases = [_make_phrase(1), _make_phrase(2)]

    async def _fail(_):
        raise RuntimeError("model unavailable")

    mocker.patch.object(
        phrase_data_service, "_fetch_batch", new=AsyncMock(return_value=phrases)
    )
    phrase_data_service._llm = MagicMock()
    phrase_data_service._llm.with_structured_output.return_value = RunnableLambda(_fail)

    mock_uow = _make_mock_uow()
    phrase_data_service.uow_factory = mock_uow

    with pytest.raises(GenerationPipelineException):
        await phrase_data_service.w2_generate(batch_size=7)

    mock_uow.phrase_repository.update_status.assert_called_once()
