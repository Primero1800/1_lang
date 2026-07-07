import pytest
import pytest_asyncio
from datetime import date
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.ai_token_usage_repository import AiTokenUsageRepository


@pytest_asyncio.fixture
async def session(test_session_maker, empty_db) -> AsyncSession:
    async with test_session_maker() as s:
        yield s


@pytest.fixture
def token_repo(session: AsyncSession) -> AiTokenUsageRepository:
    return AiTokenUsageRepository(session=session)


# --- accumulate ---


@pytest.mark.asyncio
async def test_accumulate_inserts_new_record(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    await token_repo.accumulate(
        model="mistral-large-latest",
        operation="w2_generate",
        input_tokens=100,
        output_tokens=50,
        usage_date=date(2026, 6, 22),
    )
    await session.commit()

    from sqlalchemy import select
    from app.models.ai_token_usage import AiTokenUsage

    result = await session.execute(select(AiTokenUsage))
    rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].input_tokens == 100
    assert rows[0].output_tokens == 50
    assert rows[0].model == "mistral-large-latest"


@pytest.mark.asyncio
async def test_accumulate_on_conflict_adds_tokens(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    kwargs = dict(
        model="mistral-large-latest",
        operation="w2_generate",
        input_tokens=100,
        output_tokens=50,
        usage_date=date(2026, 6, 22),
    )
    await token_repo.accumulate(**kwargs)
    await session.commit()
    await token_repo.accumulate(**kwargs)
    await session.commit()

    from sqlalchemy import select
    from app.models.ai_token_usage import AiTokenUsage

    result = await session.execute(select(AiTokenUsage))
    rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].input_tokens == 200
    assert rows[0].output_tokens == 100


# --- bulk_accumulate ---


@pytest.mark.asyncio
async def test_bulk_accumulate_inserts_multiple_rows(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    rows = [
        {
            "model": "mistral-large-latest",
            "operation": "w2_generate",
            "name": "system",
            "date": date(2026, 6, 22),
            "input_tokens": 100,
            "output_tokens": 50,
        },
        {
            "model": "mistral-embed",
            "operation": "w4_embed",
            "name": "system",
            "date": date(2026, 6, 22),
            "input_tokens": 200,
            "output_tokens": 0,
        },
    ]
    await token_repo.bulk_accumulate(rows)
    await session.commit()

    from sqlalchemy import select
    from app.models.ai_token_usage import AiTokenUsage

    result = await session.execute(select(AiTokenUsage).order_by(AiTokenUsage.model))
    inserted = result.scalars().all()
    assert len(inserted) == 2


@pytest.mark.asyncio
async def test_bulk_accumulate_on_conflict_sums_tokens(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    row = {
        "model": "mistral-large-latest",
        "operation": "w2_generate",
        "name": "system",
        "date": date(2026, 6, 22),
        "input_tokens": 100,
        "output_tokens": 50,
    }
    await token_repo.bulk_accumulate([row])
    await session.commit()
    await token_repo.bulk_accumulate([row])
    await session.commit()

    from sqlalchemy import select
    from app.models.ai_token_usage import AiTokenUsage

    result = await session.execute(select(AiTokenUsage))
    rows = result.scalars().all()
    assert len(rows) == 1
    assert rows[0].input_tokens == 200
    assert rows[0].output_tokens == 100


@pytest.mark.asyncio
async def test_bulk_accumulate_empty_is_noop(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    await token_repo.bulk_accumulate([])
    await session.commit()

    from sqlalchemy import select
    from app.models.ai_token_usage import AiTokenUsage

    result = await session.execute(select(AiTokenUsage))
    assert result.scalars().all() == []


# --- _build_conditions ---


def test_build_conditions_empty_for_no_filters(
    token_repo: AiTokenUsageRepository,
) -> None:
    from app.pyd.requests import AITokenFilter

    conditions = token_repo._build_conditions(AITokenFilter())
    assert conditions == []


def test_build_conditions_adds_date_from(
    token_repo: AiTokenUsageRepository,
) -> None:
    from app.pyd.requests import AITokenFilter

    conditions = token_repo._build_conditions(AITokenFilter(date_from=date(2026, 1, 1)))
    assert len(conditions) == 1


def test_build_conditions_adds_date_to(
    token_repo: AiTokenUsageRepository,
) -> None:
    from app.pyd.requests import AITokenFilter

    conditions = token_repo._build_conditions(AITokenFilter(date_to=date(2026, 12, 31)))
    assert len(conditions) == 1


def test_build_conditions_adds_model_filter(
    token_repo: AiTokenUsageRepository,
) -> None:
    from app.pyd.requests import AITokenFilter

    conditions = token_repo._build_conditions(AITokenFilter(model="mistral"))
    assert len(conditions) == 1


def test_build_conditions_adds_operation_prefix(
    token_repo: AiTokenUsageRepository,
) -> None:
    from app.pyd.requests import AITokenFilter

    conditions = token_repo._build_conditions(AITokenFilter(operation="w2"))
    assert len(conditions) == 1


def test_build_conditions_combines_multiple(
    token_repo: AiTokenUsageRepository,
) -> None:
    from app.pyd.requests import AITokenFilter

    conditions = token_repo._build_conditions(
        AITokenFilter(
            date_from=date(2026, 1, 1),
            date_to=date(2026, 12, 31),
            model="mistral",
            operation="w2",
        )
    )
    assert len(conditions) == 4


# --- list_usage ---


def _row(
    model: str = "mistral-large-latest",
    operation: str = "w2_generate",
    name: str = "system",
    usage_date: date = date(2026, 6, 1),
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> dict:
    return {
        "model": model,
        "operation": operation,
        "name": name,
        "date": usage_date,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


@pytest.mark.asyncio
async def test_list_usage_returns_rows_and_count(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    from app.pyd.requests import AITokenFilter, Pagination

    await token_repo.bulk_accumulate(
        [_row(), _row(model="mistral-embed", operation="w4_embed")]
    )
    await session.commit()

    rows, total = await token_repo.list_usage(
        filters=AITokenFilter(), pagination=Pagination()
    )
    assert total == 2
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_list_usage_paginates_correctly(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    from app.pyd.requests import AITokenFilter, Pagination

    await token_repo.bulk_accumulate(
        [_row(usage_date=date(2026, 6, d)) for d in range(1, 6)]
    )
    await session.commit()

    rows, total = await token_repo.list_usage(
        filters=AITokenFilter(), pagination=Pagination(per_page=2, page=2)
    )
    assert total == 5
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_list_usage_orders_by_date_desc(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    from app.pyd.requests import AITokenFilter, Pagination

    await token_repo.bulk_accumulate(
        [
            _row(usage_date=date(2026, 6, 1)),
            _row(
                model="mistral-embed",
                operation="w4_embed",
                usage_date=date(2026, 6, 10),
            ),
        ]
    )
    await session.commit()

    rows, _ = await token_repo.list_usage(
        filters=AITokenFilter(), pagination=Pagination()
    )
    assert rows[0].date >= rows[1].date


@pytest.mark.asyncio
async def test_list_usage_filters_by_model(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    from app.pyd.requests import AITokenFilter, Pagination

    await token_repo.bulk_accumulate(
        [
            _row(model="mistral-large-latest"),
            _row(model="mistral-embed", operation="w4_embed"),
        ]
    )
    await session.commit()

    rows, total = await token_repo.list_usage(
        filters=AITokenFilter(model="embed"), pagination=Pagination()
    )
    assert total == 1
    assert rows[0].model == "mistral-embed"


@pytest.mark.asyncio
async def test_list_usage_returns_empty_on_no_match(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    from app.pyd.requests import AITokenFilter, Pagination

    rows, total = await token_repo.list_usage(
        filters=AITokenFilter(), pagination=Pagination()
    )
    assert total == 0
    assert rows == []


# --- aggregate_usage ---


@pytest.mark.asyncio
async def test_aggregate_usage_sums_tokens(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    from app.pyd.requests import AITokenFilter

    await token_repo.bulk_accumulate(
        [
            _row(input_tokens=100, output_tokens=50),
            _row(
                model="mistral-embed",
                operation="w4_embed",
                input_tokens=200,
                output_tokens=0,
            ),
        ]
    )
    await session.commit()

    result = await token_repo.aggregate_usage(filters=AITokenFilter())
    assert result["input_tokens"] == 300
    assert result["output_tokens"] == 50


@pytest.mark.asyncio
async def test_aggregate_usage_returns_zeros_for_empty_table(
    token_repo: AiTokenUsageRepository, session: AsyncSession
) -> None:
    from app.pyd.requests import AITokenFilter

    result = await token_repo.aggregate_usage(filters=AITokenFilter())
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0
