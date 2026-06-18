import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import PhraseStatusEnum
from app.models.phrase_data import PhraseData
from app.models.phrases import Phrase
from app.repositories.phrase_data_repository import PhraseDataRepository
from app.repositories.phrase_repository import PhraseRepository


@pytest_asyncio.fixture
async def db_session(test_session_maker, empty_db) -> AsyncSession:
    async with test_session_maker() as session:
        yield session


def _tone() -> dict:
    return {"male": ["a", "b", "c", "d", "e"], "female": ["f", "g", "h", "i", "j"]}


def _variants() -> dict:
    t = _tone()
    return {"A": t, "B": t, "C": t, "D": t, "E": t}


async def _create_phrase(
    session: AsyncSession, original: str = "test phrase"
) -> Phrase:
    repo = PhraseRepository(session)
    await repo.bulk_create(
        [
            {
                "original": original,
                "tag": "behavior",
                "lang": "ru",
                "status": PhraseStatusEnum.DRAFT,
            }
        ]
    )
    await session.commit()
    result = await session.execute(select(Phrase).where(Phrase.original == original))
    return result.scalar_one()


@pytest.mark.asyncio
async def test_bulk_upsert_variants_inserts_row(db_session: AsyncSession) -> None:
    phrase = await _create_phrase(db_session)
    repo = PhraseDataRepository(db_session)

    await repo.bulk_upsert_variants([{"phrase_id": phrase.id, "variants": _variants()}])
    await db_session.commit()

    result = await db_session.execute(
        select(PhraseData).where(PhraseData.phrase_id == phrase.id)
    )
    row = result.scalar_one()
    assert row.variants["A"]["male"] == ["a", "b", "c", "d", "e"]


@pytest.mark.asyncio
async def test_bulk_upsert_variants_updates_on_conflict(
    test_session_maker, empty_db
) -> None:
    variants_v1 = _variants()
    variants_v2 = {
        tone: {"male": ["x", "y", "z", "w", "v"], "female": ["p", "q", "r", "s", "t"]}
        for tone in "ABCDE"
    }

    async with test_session_maker() as session:
        phrase = await _create_phrase(session, original="conflict phrase")
        phrase_id = phrase.id
        repo = PhraseDataRepository(session)
        await repo.bulk_upsert_variants(
            [{"phrase_id": phrase_id, "variants": variants_v1}]
        )
        await session.commit()

    async with test_session_maker() as session:
        repo = PhraseDataRepository(session)
        await repo.bulk_upsert_variants(
            [{"phrase_id": phrase_id, "variants": variants_v2}]
        )
        await session.commit()

    async with test_session_maker() as session:
        result = await session.execute(
            select(PhraseData).where(PhraseData.phrase_id == phrase_id)
        )
        row = result.scalar_one()
        assert row.variants["A"]["male"] == ["x", "y", "z", "w", "v"]


@pytest.mark.asyncio
async def test_bulk_upsert_variants_empty_is_noop(db_session: AsyncSession) -> None:
    repo = PhraseDataRepository(db_session)
    await repo.bulk_upsert_variants([])  # should not raise or insert anything

    result = await db_session.execute(select(PhraseData))
    assert result.scalars().all() == []


@pytest.mark.asyncio
async def test_get_by_phrase_ids_returns_variants(db_session: AsyncSession) -> None:
    phrase = await _create_phrase(db_session)
    repo = PhraseDataRepository(db_session)
    await repo.bulk_upsert_variants([{"phrase_id": phrase.id, "variants": _variants()}])
    await db_session.commit()

    result = await repo.get_by_phrase_ids([phrase.id])
    assert len(result) == 1
    assert result[0].phrase_id == phrase.id


@pytest.mark.asyncio
async def test_get_by_phrase_ids_empty_returns_empty(db_session: AsyncSession) -> None:
    repo = PhraseDataRepository(db_session)
    result = await repo.get_by_phrase_ids([])
    assert result == []
