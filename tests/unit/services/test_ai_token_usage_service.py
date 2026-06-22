import pytest
from datetime import date
from unittest.mock import AsyncMock

from app.services.ai_token_usage_service import AiTokenUsageService


@pytest.fixture
def mock_uow() -> AsyncMock:
    uow = AsyncMock()
    uow.ai_token_usage_repository = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    return uow


@pytest.fixture
def token_service(mock_uow: AsyncMock) -> AiTokenUsageService:
    return AiTokenUsageService(uow=mock_uow)


# --- accumulate ---


@pytest.mark.asyncio
async def test_accumulate_delegates_to_repository(
    token_service: AiTokenUsageService, mock_uow: AsyncMock
) -> None:
    await token_service.accumulate(
        model="mistral-large-latest",
        operation="w2_generate",
        input_tokens=100,
        output_tokens=50,
    )
    mock_uow.ai_token_usage_repository.accumulate.assert_called_once_with(
        model="mistral-large-latest",
        operation="w2_generate",
        input_tokens=100,
        output_tokens=50,
        name="system",
        usage_date=None,
    )


@pytest.mark.asyncio
async def test_accumulate_passes_custom_name_and_date(
    token_service: AiTokenUsageService, mock_uow: AsyncMock
) -> None:
    today = date(2026, 6, 22)
    await token_service.accumulate(
        model="groq-llama",
        operation="w3_translate",
        input_tokens=200,
        output_tokens=80,
        name="admin",
        usage_date=today,
    )
    mock_uow.ai_token_usage_repository.accumulate.assert_called_once_with(
        model="groq-llama",
        operation="w3_translate",
        input_tokens=200,
        output_tokens=80,
        name="admin",
        usage_date=today,
    )
