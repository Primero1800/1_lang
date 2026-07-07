import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

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


def _item(concrete: str, abstract: str, tag: str) -> PhraseItem:
    return PhraseItem(concrete=concrete, abstract=abstract, tag=tag)


def _vo(*items: tuple[str, str, str]) -> VisionOutput:
    return VisionOutput(
        phrases=[PhraseItem(concrete=c, abstract=a, tag=t) for c, a, t in items]
    )


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
    vo = _vo(("typing fast", "lost in motion", "behavior"))
    batch = VisionBatchOutput(photos=[vo])
    raw_mock = MagicMock()
    raw_mock.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    result = await phrase_service._fire_token_task({"raw": raw_mock, "parsed": batch})
    await asyncio.sleep(0)

    assert isinstance(result, VisionOutput)
    assert result.phrases[0].concrete == "typing fast"


@pytest.mark.asyncio
async def test_fire_token_task_schedules_xadd(phrase_service: PhraseService) -> None:
    vo = _vo(("typing fast", "lost in motion", "behavior"))
    batch = VisionBatchOutput(photos=[vo])
    raw_mock = MagicMock()
    raw_mock.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    await phrase_service._fire_token_task({"raw": raw_mock, "parsed": batch})
    await asyncio.sleep(0)

    phrase_service.queue_client.xadd.assert_called_once()


# --- _build_rows ---


@pytest.mark.asyncio
async def test_build_rows_concatenates_concrete_and_abstract(
    phrase_service: PhraseService,
) -> None:
    vo = _vo(("typing fast", "lost in motion", "behavior"))
    result = await phrase_service._build_rows(vo, "ru")
    assert len(result) == 1
    assert result[0]["original"] == "typing fast. lost in motion"
    assert result[0]["tag"] == "behavior"
    assert result[0]["lang"] == "ru"
    assert result[0]["status"] == PhraseStatusEnum.DRAFT


@pytest.mark.asyncio
async def test_build_rows_deduplicates_by_original(
    phrase_service: PhraseService,
) -> None:
    vo = VisionOutput(
        phrases=[
            _item("same phrase", "same abstract", "behavior"),
            _item("same phrase", "same abstract", "mood"),
        ]
    )
    result = await phrase_service._build_rows(vo, "ru")
    assert len(result) == 1
    assert result[0]["tag"] == "behavior"


@pytest.mark.asyncio
async def test_build_rows_returns_multiple_distinct_items(
    phrase_service: PhraseService,
) -> None:
    vo = _vo(
        ("typing fast", "focused mind", "behavior"),
        ("neat outfit", "clean appearance", "appearance"),
    )
    result = await phrase_service._build_rows(vo, "ru")
    assert len(result) == 2


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
    vo = _vo(
        ("typing fast", "lost in motion", "behavior"),
        ("neat outfit", "clean style", "appearance"),
    )
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
