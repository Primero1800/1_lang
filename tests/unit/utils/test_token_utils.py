import pytest
from unittest.mock import AsyncMock, patch

from app.utils.token_utils import record_token_usage


@pytest.mark.asyncio
async def test_record_token_usage_calls_service_accumulate() -> None:
    mock_uow = AsyncMock()
    mock_service = AsyncMock()

    with (
        patch(
            "app.utils.token_utils.get_uow_factory",
            new=AsyncMock(return_value=mock_uow),
        ),
        patch("app.utils.token_utils.AiTokenUsageService", return_value=mock_service),
    ):
        await record_token_usage(
            model="mistral-large-latest",
            operation="w2_generate",
            input_tokens=100,
            output_tokens=50,
        )

    mock_service.accumulate.assert_called_once_with(
        model="mistral-large-latest",
        operation="w2_generate",
        input_tokens=100,
        output_tokens=50,
        name="system",
        usage_date=None,
    )


@pytest.mark.asyncio
async def test_record_token_usage_swallows_integrity_exception() -> None:
    from app.common.exceptions import IntegrityDataException

    with patch(
        "app.utils.token_utils.get_uow_factory",
        side_effect=IntegrityDataException("dup"),
    ):
        await record_token_usage(
            model="m", operation="op", input_tokens=1, output_tokens=1
        )


@pytest.mark.asyncio
async def test_record_token_usage_swallows_generic_exception() -> None:
    with patch(
        "app.utils.token_utils.get_uow_factory", side_effect=RuntimeError("db down")
    ):
        await record_token_usage(
            model="m", operation="op", input_tokens=1, output_tokens=1
        )
