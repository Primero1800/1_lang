import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import PhraseStatusEnum
from app.repositories.phrase_repository import PhraseRepository


@pytest_asyncio.fixture
async def db_session(test_session_maker, empty_db) -> AsyncSession:
    async with test_session_maker() as session:
        yield session


@pytest.mark.asyncio
async def test_bulk_create_empty_list_returns_zero(db_session: AsyncSession) -> None:
    repo = PhraseRepository(db_session)
    result = await repo.bulk_create([])
    assert result == 0


@pytest.mark.asyncio
async def test_bulk_create_inserts_rows(db_session: AsyncSession) -> None:
    repo = PhraseRepository(db_session)
    rows = [
        {
            "original": "test phrase one",
            "tag": "behavior",
            "lang": "ru",
            "status": PhraseStatusEnum.DRAFT,
        },
        {
            "original": "test phrase two",
            "tag": "appearance",
            "lang": "ru",
            "status": PhraseStatusEnum.DRAFT,
        },
    ]
    result = await repo.bulk_create(rows)
    assert result == 2


@pytest.mark.asyncio
async def test_bulk_create_on_conflict_skips_duplicate(
    test_session_maker, empty_db
) -> None:
    rows = [
        {
            "original": "dup phrase",
            "tag": "behavior",
            "lang": "ru",
            "status": PhraseStatusEnum.DRAFT,
        }
    ]

    async with test_session_maker() as session:
        repo = PhraseRepository(session)
        await repo.bulk_create(rows)
        await session.commit()

    async with test_session_maker() as session:
        repo = PhraseRepository(session)
        result = await repo.bulk_create(rows)
        assert result == 0
