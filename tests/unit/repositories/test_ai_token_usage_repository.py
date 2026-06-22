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
