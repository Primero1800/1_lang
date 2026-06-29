import asyncio
import base64
import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from app.common.enums import PhraseStatusEnum
from app.common.exceptions import VisionPipelineException
from app.pyd.ai_schemas import PhraseItem, VisionBatchOutput, VisionOutput
from app.services.base import BaseDeps
from app.services.phrase_service import PhraseService


@pytest.fixture
def phrase_service() -> PhraseService:
    base_deps = MagicMock(spec=BaseDeps)
    base_deps.uow_factory = MagicMock()
    base_deps.ai_client = MagicMock()
    base_deps.ai_client2 = MagicMock()
    base_deps.vector_client = MagicMock()
    base_deps.vector_client_main = MagicMock()
    base_deps.queue_client = AsyncMock()
    return PhraseService(base_deps=base_deps)


def _make_mock_uow() -> AsyncMock:
    mock_uow = AsyncMock()
    mock_uow.phrase_repository = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=None)
    return mock_uow


def _vo(*phrase_tag_pairs: tuple[str, str]) -> VisionOutput:
    return VisionOutput(
        phrases=[PhraseItem(phrase=p, tag=t) for p, t in phrase_tag_pairs]
    )


# --- _encode_images ---


@pytest.mark.asyncio
async def test_encode_images_converts_bytes_to_base64(
    phrase_service: PhraseService,
) -> None:
    raw = b"test image bytes"
    result = await phrase_service._encode_images({"images_raw": [raw], "lang": "ru"})
    assert result["images_b64"] == [base64.b64encode(raw).decode()]
    assert result["lang"] == "ru"


@pytest.mark.asyncio
async def test_encode_images_passes_lang_through(phrase_service: PhraseService) -> None:
    result = await phrase_service._encode_images({"images_raw": [b"x"], "lang": "en"})
    assert result["lang"] == "en"


# --- _build_vision_message ---


@pytest.mark.asyncio
async def test_build_vision_message_returns_single_human_message(
    phrase_service: PhraseService,
) -> None:
    result = await phrase_service._build_vision_message(
        {"images_b64": ["abc123"], "lang": "ru"}
    )
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)


@pytest.mark.asyncio
async def test_build_vision_message_embeds_image_url(
    phrase_service: PhraseService,
) -> None:
    result = await phrase_service._build_vision_message(
        {"images_b64": ["abc123"], "lang": "ru"}
    )
    content = result[0].content
    image_items = [
        c for c in content if isinstance(c, dict) and c.get("type") == "image_url"
    ]
    assert len(image_items) == 1
    assert "abc123" in image_items[0]["image_url"]["url"]


# --- _fire_token_task ---


@pytest.mark.asyncio
async def test_fire_token_task_raises_when_parsed_is_none(
    phrase_service: PhraseService,
) -> None:
    with pytest.raises(VisionPipelineException):
        await phrase_service._fire_token_task({"raw": None, "parsed": None})


@pytest.mark.asyncio
async def test_fire_token_task_returns_vision_output(
    phrase_service: PhraseService,
) -> None:
    vo = _vo(("typing fast", "behavior"))
    batch = VisionBatchOutput(photos=[vo])
    raw_mock = MagicMock()
    raw_mock.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    result = await phrase_service._fire_token_task({"raw": raw_mock, "parsed": batch})
    await asyncio.sleep(0)

    assert isinstance(result, VisionOutput)
    assert result.phrases[0].phrase == "typing fast"


@pytest.mark.asyncio
async def test_fire_token_task_schedules_xadd(phrase_service: PhraseService) -> None:
    vo = _vo(("typing fast", "behavior"))
    batch = VisionBatchOutput(photos=[vo])
    raw_mock = MagicMock()
    raw_mock.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    await phrase_service._fire_token_task({"raw": raw_mock, "parsed": batch})
    await asyncio.sleep(0)

    phrase_service.queue_client.xadd.assert_called_once()


# --- _build_rows ---


@pytest.mark.asyncio
async def test_build_rows_normalises_and_lowercases(
    phrase_service: PhraseService,
) -> None:
    result = await phrase_service._build_rows(_vo(("Hello World", "behavior")), "ru")
    assert len(result) == 1
    assert result[0]["original"] == "hello world"
    assert result[0]["tag"] == "behavior"
    assert result[0]["lang"] == "ru"
    assert result[0]["status"] == PhraseStatusEnum.DRAFT


@pytest.mark.asyncio
async def test_build_rows_strips_special_chars(phrase_service: PhraseService) -> None:
    result = await phrase_service._build_rows(_vo(("hello, world!", "mood")), "en")
    assert result[0]["original"] == "hello world"


@pytest.mark.asyncio
async def test_build_rows_skips_empty_phrase(phrase_service: PhraseService) -> None:
    result = await phrase_service._build_rows(_vo(("!!!???", "behavior")), "ru")
    assert result == []


@pytest.mark.asyncio
async def test_build_rows_deduplicates_by_original(
    phrase_service: PhraseService,
) -> None:
    vo = VisionOutput(
        phrases=[
            PhraseItem(phrase="same phrase", tag="behavior"),
            PhraseItem(phrase="same phrase", tag="mood"),
        ]
    )
    result = await phrase_service._build_rows(vo, "ru")
    assert len(result) == 1
    assert result[0]["tag"] == "behavior"


# --- _save_phrases ---


@pytest.mark.asyncio
async def test_save_phrases_raises_on_empty_rows(phrase_service: PhraseService) -> None:
    with pytest.raises(VisionPipelineException):
        await phrase_service._save_phrases([])


@pytest.mark.asyncio
async def test_save_phrases_returns_counts(phrase_service: PhraseService) -> None:
    rows = [
        {
            "original": "a",
            "tag": "behavior",
            "lang": "ru",
            "status": PhraseStatusEnum.DRAFT,
        },
        {
            "original": "b",
            "tag": "mood",
            "lang": "ru",
            "status": PhraseStatusEnum.DRAFT,
        },
    ]
    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.bulk_create = AsyncMock(return_value=[1])
    phrase_service.uow_factory = mock_uow

    result = await phrase_service._save_phrases(rows)

    assert result["phrases_found"] == 2
    assert result["inserted"] == 1
    assert result["skipped"] == 1


# --- upload_images ---


@pytest.mark.asyncio
async def test_upload_images_wraps_unexpected_error(
    phrase_service: PhraseService,
) -> None:
    async def _raise(_):
        raise RuntimeError("model unavailable")

    phrase_service._llm = MagicMock()
    phrase_service._llm.with_structured_output.return_value = RunnableLambda(_raise)

    with pytest.raises(VisionPipelineException):
        await phrase_service.upload_images(images_raw=[b"img"], lang="ru")


@pytest.mark.asyncio
async def test_upload_images_success(phrase_service: PhraseService) -> None:
    vo = _vo(("typing fast", "behavior"), ("looks neat", "appearance"))
    batch = VisionBatchOutput(photos=[vo])

    async def _fake_llm(_):
        return {"raw": MagicMock(usage_metadata=None), "parsed": batch}

    phrase_service._llm = MagicMock()
    phrase_service._llm.with_structured_output.return_value = RunnableLambda(_fake_llm)

    mock_uow = _make_mock_uow()
    mock_uow.phrase_repository.bulk_create = AsyncMock(return_value=[1, 2])
    phrase_service.uow_factory = mock_uow

    result = await phrase_service.upload_images(images_raw=[b"img"], lang="ru")

    assert result["phrases_found"] == 2
    assert result["inserted"] == 2
    assert result["skipped"] == 0
