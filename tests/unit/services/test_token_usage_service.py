from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.pyd.requests import AITokenFilter, Pagination
from app.services.token_usage_service import TokenUsageService


def _make_row(
    model: str = "mistral-large-latest",
    operation: str = "w2_generate",
    name: str = "system",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    row = MagicMock()
    row.model = model
    row.operation = operation
    row.name = name
    row.date = date(2026, 6, 1)
    row.input_tokens = input_tokens
    row.output_tokens = output_tokens
    row.updated_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
    return row


@pytest.fixture
def service() -> TokenUsageService:
    base_deps = MagicMock()
    uow = MagicMock()
    uow.ai_token_usage_repository = AsyncMock()
    return TokenUsageService(base_deps=base_deps, uow=uow)


# --- list_usage ---


@pytest.mark.asyncio
async def test_list_usage_returns_paginated_structure(
    service: TokenUsageService,
) -> None:
    row = _make_row()
    service.uow.ai_token_usage_repository.list_usage.return_value = ([row], 1)

    result = await service.list_usage(
        filters=AITokenFilter(), pagination=Pagination(per_page=10, page=1)
    )

    assert result["per_page"] == 10
    assert result["page"] == 1
    assert result["total_count"] == 1
    assert len(result["items"]) == 1


@pytest.mark.asyncio
async def test_list_usage_maps_row_fields(service: TokenUsageService) -> None:
    row = _make_row(model="mistral-embed", operation="w4_embed", name="eval")
    service.uow.ai_token_usage_repository.list_usage.return_value = ([row], 1)

    result = await service.list_usage(filters=AITokenFilter(), pagination=Pagination())

    item = result["items"][0]
    assert item["model"] == "mistral-embed"
    assert item["operation"] == "w4_embed"
    assert item["name"] == "eval"


@pytest.mark.asyncio
async def test_list_usage_calculates_total_tokens(service: TokenUsageService) -> None:
    row = _make_row(input_tokens=100, output_tokens=50)
    service.uow.ai_token_usage_repository.list_usage.return_value = ([row], 1)

    result = await service.list_usage(filters=AITokenFilter(), pagination=Pagination())

    assert result["items"][0]["total_tokens"] == 150


@pytest.mark.asyncio
async def test_list_usage_exclude_input_zeroes_input_tokens(
    service: TokenUsageService,
) -> None:
    row = _make_row(input_tokens=100, output_tokens=50)
    service.uow.ai_token_usage_repository.list_usage.return_value = ([row], 1)

    result = await service.list_usage(
        filters=AITokenFilter(exclude_input=True), pagination=Pagination()
    )

    item = result["items"][0]
    assert item["input_tokens"] == 0
    assert item["output_tokens"] == 50
    assert item["total_tokens"] == 50


@pytest.mark.asyncio
async def test_list_usage_exclude_output_zeroes_output_tokens(
    service: TokenUsageService,
) -> None:
    row = _make_row(input_tokens=100, output_tokens=50)
    service.uow.ai_token_usage_repository.list_usage.return_value = ([row], 1)

    result = await service.list_usage(
        filters=AITokenFilter(exclude_output=True), pagination=Pagination()
    )

    item = result["items"][0]
    assert item["input_tokens"] == 100
    assert item["output_tokens"] == 0
    assert item["total_tokens"] == 100


@pytest.mark.asyncio
async def test_list_usage_returns_empty_items_when_no_rows(
    service: TokenUsageService,
) -> None:
    service.uow.ai_token_usage_repository.list_usage.return_value = ([], 0)

    result = await service.list_usage(filters=AITokenFilter(), pagination=Pagination())

    assert result["total_count"] == 0
    assert result["items"] == []


# --- aggregate_usage ---


@pytest.mark.asyncio
async def test_aggregate_usage_returns_filter_context_fields(
    service: TokenUsageService,
) -> None:
    service.uow.ai_token_usage_repository.aggregate_usage.return_value = {
        "input_tokens": 200,
        "output_tokens": 80,
    }

    result = await service.aggregate_usage(
        filters=AITokenFilter(model="mistral", operation="w2", name="system")
    )

    assert result["model"] == "mistral"
    assert result["operation"] == "w2"
    assert result["name"] == "system"


@pytest.mark.asyncio
async def test_aggregate_usage_calculates_total_tokens(
    service: TokenUsageService,
) -> None:
    service.uow.ai_token_usage_repository.aggregate_usage.return_value = {
        "input_tokens": 200,
        "output_tokens": 80,
    }

    result = await service.aggregate_usage(filters=AITokenFilter())
    assert result["total_tokens"] == 280


@pytest.mark.asyncio
async def test_aggregate_usage_exclude_input_zeroes_input(
    service: TokenUsageService,
) -> None:
    service.uow.ai_token_usage_repository.aggregate_usage.return_value = {
        "input_tokens": 200,
        "output_tokens": 80,
    }

    result = await service.aggregate_usage(filters=AITokenFilter(exclude_input=True))

    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 80
    assert result["total_tokens"] == 80


@pytest.mark.asyncio
async def test_aggregate_usage_exclude_output_zeroes_output(
    service: TokenUsageService,
) -> None:
    service.uow.ai_token_usage_repository.aggregate_usage.return_value = {
        "input_tokens": 200,
        "output_tokens": 80,
    }

    result = await service.aggregate_usage(filters=AITokenFilter(exclude_output=True))

    assert result["output_tokens"] == 0
    assert result["input_tokens"] == 200
    assert result["total_tokens"] == 200
