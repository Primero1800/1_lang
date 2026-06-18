import pytest
from unittest.mock import AsyncMock, MagicMock

from app.common.enums import PhraseStatusEnum
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
    return PhraseService(base_deps=base_deps)


@pytest.mark.asyncio
async def test_parse_pixtral_response_plain_list(phrase_service: PhraseService) -> None:
    raw = str(
        [
            [
                {"behavior": ["typing fast", "looks focused"]},
                {"appearance": ["neat and tidy"]},
            ]
        ]
    )
    result = await phrase_service._parse_pixtral_response(raw)
    assert len(result) == 3
    assert {"phrase": "typing fast", "tag": "behavior"} in result
    assert {"phrase": "neat and tidy", "tag": "appearance"} in result


@pytest.mark.asyncio
async def test_parse_pixtral_response_with_code_fence(
    phrase_service: PhraseService,
) -> None:
    inner = str([[{"behavior": ["slouching in chair"]}]])
    raw = f"```python\n{inner}\n```"
    result = await phrase_service._parse_pixtral_response(raw)
    assert result == [{"phrase": "slouching in chair", "tag": "behavior"}]


@pytest.mark.asyncio
async def test_parse_pixtral_response_invalid_returns_empty(
    phrase_service: PhraseService,
) -> None:
    result = await phrase_service._parse_pixtral_response("this is not valid python!!!")
    assert result == []


@pytest.mark.asyncio
async def test_parse_pixtral_response_deduplicates(
    phrase_service: PhraseService,
) -> None:
    raw = str(
        [
            [{"behavior": ["same phrase here"]}],
            [{"behavior": ["same phrase here"]}],
        ]
    )
    result = await phrase_service._parse_pixtral_response(raw)
    assert result == [{"phrase": "same phrase here", "tag": "behavior"}]


@pytest.mark.asyncio
async def test_build_rows_normal_phrase(phrase_service: PhraseService) -> None:
    parsed = [{"phrase": "Hello World", "tag": "behavior"}]
    result = await phrase_service._build_rows(parsed, "ru")
    assert len(result) == 1
    assert result[0]["original"] == "hello world"
    assert result[0]["tag"] == "behavior"
    assert result[0]["lang"] == "ru"
    assert result[0]["status"] == PhraseStatusEnum.DRAFT


@pytest.mark.asyncio
async def test_build_rows_strips_special_chars(phrase_service: PhraseService) -> None:
    parsed = [{"phrase": "hello, world!", "tag": "mood"}]
    result = await phrase_service._build_rows(parsed, "en")
    assert result[0]["original"] == "hello world"


@pytest.mark.asyncio
async def test_build_rows_skips_empty_phrase(phrase_service: PhraseService) -> None:
    parsed = [{"phrase": "!!!???", "tag": "behavior"}]
    result = await phrase_service._build_rows(parsed, "ru")
    assert result == []


@pytest.mark.asyncio
async def test_upload_images_recognize_none(
    phrase_service: PhraseService, mocker
) -> None:
    mocker.patch.object(phrase_service, "_recognize", new=AsyncMock(return_value=None))
    result = await phrase_service.upload_images(images_raw=[b"img"], lang="ru")
    assert result == {"phrases_found": 0, "inserted": 0, "skipped": 0}


@pytest.mark.asyncio
async def test_upload_images_empty_parsed(
    phrase_service: PhraseService, mocker
) -> None:
    mocker.patch.object(
        phrase_service, "_recognize", new=AsyncMock(return_value="some raw")
    )
    mocker.patch.object(
        phrase_service, "_parse_pixtral_response", new=AsyncMock(return_value=[])
    )
    result = await phrase_service.upload_images(images_raw=[b"img"], lang="ru")
    assert result == {"phrases_found": 0, "inserted": 0, "skipped": 0}


@pytest.mark.asyncio
async def test_upload_images_empty_rows(phrase_service: PhraseService, mocker) -> None:
    mocker.patch.object(
        phrase_service, "_recognize", new=AsyncMock(return_value="some raw")
    )
    mocker.patch.object(
        phrase_service,
        "_parse_pixtral_response",
        new=AsyncMock(return_value=[{"phrase": "x", "tag": "y"}]),
    )
    mocker.patch.object(phrase_service, "_build_rows", new=AsyncMock(return_value=[]))
    result = await phrase_service.upload_images(images_raw=[b"img"], lang="ru")
    assert result == {"phrases_found": 0, "inserted": 0, "skipped": 0}


@pytest.mark.asyncio
async def test_upload_images_success(phrase_service: PhraseService, mocker) -> None:
    rows = [
        {"original": "a", "tag": "1", "lang": "ru", "status": PhraseStatusEnum.DRAFT},
        {"original": "b", "tag": "1", "lang": "ru", "status": PhraseStatusEnum.DRAFT},
        {"original": "c", "tag": "1", "lang": "ru", "status": PhraseStatusEnum.DRAFT},
    ]
    mocker.patch.object(phrase_service, "_recognize", new=AsyncMock(return_value="raw"))
    mocker.patch.object(
        phrase_service,
        "_parse_pixtral_response",
        new=AsyncMock(return_value=[{"phrase": "x", "tag": "y"}]),
    )
    mocker.patch.object(phrase_service, "_build_rows", new=AsyncMock(return_value=rows))
    mocker.patch.object(phrase_service, "_save_phrases", new=AsyncMock(return_value=2))
    result = await phrase_service.upload_images(images_raw=[b"img"], lang="ru")
    assert result == {"phrases_found": 3, "inserted": 2, "skipped": 1}
