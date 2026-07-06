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
