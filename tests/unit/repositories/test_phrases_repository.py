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
async def test_bulk_create_empty_list_returns_empty(db_session: AsyncSession) -> None:
    repo = PhraseRepository(db_session)
    result = await repo.bulk_create([])
    assert result == []


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
    assert len(result) == 2
    assert all(isinstance(i, int) for i in result)


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
        assert result == []
