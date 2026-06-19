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


# --- get_t1_vision_prompt ---


def test_get_t1_vision_prompt_contains_requested_tags() -> None:
    prompt = PromptService.get_t1_vision_prompt(
        lang="ru", allowed_tags=["behavior", "mood"]
    )
    assert "behavior" in prompt
    assert "mood" in prompt


def test_get_t1_vision_prompt_excludes_omitted_tags() -> None:
    prompt = PromptService.get_t1_vision_prompt(lang="ru", allowed_tags=["behavior"])
    assert "appearance" not in prompt


def test_get_t1_vision_prompt_ru_and_en_differ() -> None:
    ru = PromptService.get_t1_vision_prompt(lang="ru", allowed_tags=["behavior"])
    en = PromptService.get_t1_vision_prompt(lang="en", allowed_tags=["behavior"])
    assert ru != en


def test_get_t1_vision_prompt_contains_json_example() -> None:
    prompt = PromptService.get_t1_vision_prompt(
        lang="en", allowed_tags=["behavior", "age"]
    )
    assert "behavior" in prompt
    assert "gender" in prompt
