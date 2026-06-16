import pytest

from app.services.prompt_service import (
    PROMPT_PIXTRAL_EN,
    PROMPT_PIXTRAL_RU,
    PromptService,
)


@pytest.fixture
def prompt_service() -> PromptService:
    return PromptService()


def test_get_pixtral_vision_ru(prompt_service: PromptService) -> None:
    result = prompt_service.get("pixtral_vision", "ru")
    assert result == PROMPT_PIXTRAL_RU
    assert len(result) > 0


def test_get_pixtral_vision_en(prompt_service: PromptService) -> None:
    result = prompt_service.get("pixtral_vision", "en")
    assert result == PROMPT_PIXTRAL_EN
    assert len(result) > 0


def test_ru_and_en_prompts_differ(prompt_service: PromptService) -> None:
    ru = prompt_service.get("pixtral_vision", "ru")
    en = prompt_service.get("pixtral_vision", "en")
    assert ru != en


def test_get_unknown_marker_raises(prompt_service: PromptService) -> None:
    with pytest.raises(ValueError, match="No prompt for marker="):
        prompt_service.get("nonexistent_marker", "ru")


def test_get_unknown_lang_raises(prompt_service: PromptService) -> None:
    with pytest.raises(ValueError, match="No prompt for marker="):
        prompt_service.get("pixtral_vision", "de")
