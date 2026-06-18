import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import PhraseStatusEnum
from app.models.phrase_embeddings import PhraseEmbedding
from app.models.phrases import Phrase
from app.repositories.phrase_embedding_repository import PhraseEmbeddingRepository
from app.repositories.phrase_repository import PhraseRepository


@pytest_asyncio.fixture
async def db_session(test_session_maker, empty_db) -> AsyncSession:
    async with test_session_maker() as session:
        yield session


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
                "status": PhraseStatusEnum.TRANSLATING_DONE,
            }
        ]
    )
    await session.commit()
    result = await session.execute(select(Phrase).where(Phrase.original == original))
    return result.scalar_one()


@pytest.mark.asyncio
async def test_bulk_upsert_embeddings_inserts_row(db_session: AsyncSession) -> None:
    phrase = await _create_phrase(db_session)
    repo = PhraseEmbeddingRepository(db_session)

    await repo.bulk_upsert_embeddings(
        [{"phrase_id": phrase.id, "embedding": [0.1, 0.2, 0.3]}]
    )
    await db_session.commit()

    result = await db_session.execute(
        select(PhraseEmbedding).where(PhraseEmbedding.phrase_id == phrase.id)
    )
    row = result.scalar_one()
    assert row.embedding[0] == pytest.approx(0.1)
    assert row.embedding[1] == pytest.approx(0.2)
    assert row.embedding[2] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_bulk_upsert_embeddings_updates_on_conflict(
    test_session_maker, empty_db
) -> None:
    vector_v1 = [0.1, 0.2, 0.3]
    vector_v2 = [0.9, 0.8, 0.7]

    async with test_session_maker() as session:
        phrase = await _create_phrase(session, original="conflict phrase")
        phrase_id = phrase.id
        repo = PhraseEmbeddingRepository(session)
        await repo.bulk_upsert_embeddings(
            [{"phrase_id": phrase_id, "embedding": vector_v1}]
        )
        await session.commit()

    async with test_session_maker() as session:
        repo = PhraseEmbeddingRepository(session)
        await repo.bulk_upsert_embeddings(
            [{"phrase_id": phrase_id, "embedding": vector_v2}]
        )
        await session.commit()

    async with test_session_maker() as session:
        result = await session.execute(
            select(PhraseEmbedding).where(PhraseEmbedding.phrase_id == phrase_id)
        )
        row = result.scalar_one()
        assert row.embedding[0] == pytest.approx(0.9)
        assert row.embedding[1] == pytest.approx(0.8)
        assert row.embedding[2] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_bulk_upsert_embeddings_empty_is_noop(db_session: AsyncSession) -> None:
    repo = PhraseEmbeddingRepository(db_session)
    await repo.bulk_upsert_embeddings([])

    result = await db_session.execute(select(PhraseEmbedding))
    assert result.scalars().all() == []
