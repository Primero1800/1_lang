from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import PhraseStatusEnum
from app.models.phrases import Phrase
from app.repositories.phrase_repository import PhraseRepository


@pytest_asyncio.fixture
async def db_session(test_session_maker, empty_db) -> AsyncSession:
    async with test_session_maker() as session:
        yield session


async def _insert(
    session: AsyncSession,
    original: str,
    status: PhraseStatusEnum,
    tag: str = "behavior",
    lang: str = "ru",
) -> int:
    repo = PhraseRepository(session)
    ids = await repo.bulk_create(
        [{"original": original, "tag": tag, "lang": lang, "status": status}]
    )
    await session.commit()
    return ids[0]


async def _set_updated_at(session: AsyncSession, phrase_id: int, dt: datetime) -> None:
    await session.execute(
        sa_update(Phrase).where(Phrase.id == phrase_id).values(updated_at=dt)
    )
    await session.commit()


# --- bulk_create ---


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


# --- get_first_for_processing ---


@pytest.mark.asyncio
async def test_get_first_returns_none_when_empty(db_session: AsyncSession) -> None:
    repo = PhraseRepository(db_session)
    result = await repo.get_first_for_processing(
        in_progress_status=PhraseStatusEnum.GENERATING_IN_PROGRESS,
        priority_status=PhraseStatusEnum.GENERATING_FAILED,
        base_statuses=[PhraseStatusEnum.DRAFT],
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_first_returns_priority_status_before_base(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    await _insert(db_session, "draft phrase", PhraseStatusEnum.DRAFT)
    await _insert(db_session, "failed phrase", PhraseStatusEnum.GENERATING_FAILED)

    result = await repo.get_first_for_processing(
        in_progress_status=PhraseStatusEnum.GENERATING_IN_PROGRESS,
        priority_status=PhraseStatusEnum.GENERATING_FAILED,
        base_statuses=[PhraseStatusEnum.DRAFT],
    )
    assert result is not None
    assert result.status == PhraseStatusEnum.GENERATING_FAILED


@pytest.mark.asyncio
async def test_get_first_returns_stuck_in_progress_before_priority(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    stuck_id = await _insert(
        db_session, "stuck phrase", PhraseStatusEnum.GENERATING_IN_PROGRESS
    )
    await _set_updated_at(
        db_session, stuck_id, datetime.now(timezone.utc) - timedelta(hours=1)
    )
    await _insert(db_session, "failed phrase", PhraseStatusEnum.GENERATING_FAILED)

    result = await repo.get_first_for_processing(
        in_progress_status=PhraseStatusEnum.GENERATING_IN_PROGRESS,
        priority_status=PhraseStatusEnum.GENERATING_FAILED,
        base_statuses=[PhraseStatusEnum.DRAFT],
    )
    assert result is not None
    assert result.original == "stuck phrase"


# --- get_batch_for_processing ---


@pytest.mark.asyncio
async def test_get_batch_returns_multiple_phrases(db_session: AsyncSession) -> None:
    repo = PhraseRepository(db_session)
    for i in range(4):
        await _insert(db_session, f"phrase {i}", PhraseStatusEnum.DRAFT)

    result = await repo.get_batch_for_processing(
        in_progress_status=PhraseStatusEnum.GENERATING_IN_PROGRESS,
        priority_status=PhraseStatusEnum.GENERATING_FAILED,
        base_statuses=[PhraseStatusEnum.DRAFT],
        limit=3,
    )
    assert len(result) == 3


@pytest.mark.asyncio
async def test_get_batch_filters_by_lang(db_session: AsyncSession) -> None:
    repo = PhraseRepository(db_session)
    await _insert(db_session, "ru phrase", PhraseStatusEnum.DRAFT, lang="ru")
    await _insert(db_session, "en phrase", PhraseStatusEnum.DRAFT, lang="en")

    result = await repo.get_batch_for_processing(
        in_progress_status=PhraseStatusEnum.GENERATING_IN_PROGRESS,
        priority_status=PhraseStatusEnum.GENERATING_FAILED,
        base_statuses=[PhraseStatusEnum.DRAFT],
        lang="ru",
    )
    assert len(result) == 1
    assert result[0].lang == "ru"


@pytest.mark.asyncio
async def test_get_batch_excludes_given_id(db_session: AsyncSession) -> None:
    repo = PhraseRepository(db_session)
    id1 = await _insert(db_session, "phrase one", PhraseStatusEnum.DRAFT)
    await _insert(db_session, "phrase two", PhraseStatusEnum.DRAFT)

    result = await repo.get_batch_for_processing(
        in_progress_status=PhraseStatusEnum.GENERATING_IN_PROGRESS,
        priority_status=PhraseStatusEnum.GENERATING_FAILED,
        base_statuses=[PhraseStatusEnum.DRAFT],
        exclude_id=id1,
    )
    assert all(p.id != id1 for p in result)


@pytest.mark.asyncio
async def test_get_batch_returns_empty_when_nothing_matches(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    await _insert(db_session, "done phrase", PhraseStatusEnum.GENERATING_DONE)

    result = await repo.get_batch_for_processing(
        in_progress_status=PhraseStatusEnum.GENERATING_IN_PROGRESS,
        priority_status=PhraseStatusEnum.GENERATING_FAILED,
        base_statuses=[PhraseStatusEnum.DRAFT],
    )
    assert result == []


# --- get_ids_by_originals ---


@pytest.mark.asyncio
async def test_get_ids_by_originals_returns_mapping(db_session: AsyncSession) -> None:
    repo = PhraseRepository(db_session)
    rows = [
        {
            "original": "alpha phrase",
            "tag": "behavior",
            "lang": "en",
            "status": PhraseStatusEnum.DRAFT,
        },
        {
            "original": "beta phrase",
            "tag": "appearance",
            "lang": "en",
            "status": PhraseStatusEnum.DRAFT,
        },
    ]
    await repo.bulk_create(rows)
    await db_session.commit()

    result = await repo.get_ids_by_originals(
        originals=["alpha phrase", "beta phrase"], lang="en"
    )
    assert set(result.keys()) == {"alpha phrase", "beta phrase"}
    assert all(isinstance(v, int) for v in result.values())


@pytest.mark.asyncio
async def test_get_ids_by_originals_filters_by_lang(db_session: AsyncSession) -> None:
    repo = PhraseRepository(db_session)
    await repo.bulk_create(
        [
            {
                "original": "gamma phrase",
                "tag": "behavior",
                "lang": "ru",
                "status": PhraseStatusEnum.DRAFT,
            }
        ]
    )
    await db_session.commit()

    result = await repo.get_ids_by_originals(originals=["gamma phrase"], lang="en")
    assert result == {}


# --- update_status ---


@pytest.mark.asyncio
async def test_update_status_changes_targeted_ids(db_session: AsyncSession) -> None:
    repo = PhraseRepository(db_session)
    ids = await repo.bulk_create(
        [
            {
                "original": "phrase a",
                "tag": "behavior",
                "lang": "ru",
                "status": PhraseStatusEnum.DRAFT,
            },
            {
                "original": "phrase b",
                "tag": "behavior",
                "lang": "ru",
                "status": PhraseStatusEnum.DRAFT,
            },
        ]
    )
    await db_session.commit()

    await repo.update_status([ids[0]], PhraseStatusEnum.GENERATING_IN_PROGRESS)
    await db_session.commit()

    from sqlalchemy import select

    rows = (
        (await db_session.execute(select(Phrase).where(Phrase.id.in_(ids))))
        .scalars()
        .all()
    )
    statuses = {p.id: p.status for p in rows}
    assert statuses[ids[0]] == PhraseStatusEnum.GENERATING_IN_PROGRESS
    assert statuses[ids[1]] == PhraseStatusEnum.DRAFT


@pytest.mark.asyncio
async def test_update_status_does_not_affect_other_ids(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    ids = await repo.bulk_create(
        [
            {
                "original": "keep draft",
                "tag": "behavior",
                "lang": "ru",
                "status": PhraseStatusEnum.DRAFT,
            },
        ]
    )
    await db_session.commit()

    await repo.update_status([], PhraseStatusEnum.GENERATING_DONE)
    await db_session.commit()

    from sqlalchemy import select

    phrase = (
        await db_session.execute(select(Phrase).where(Phrase.id == ids[0]))
    ).scalar_one()
    assert phrase.status == PhraseStatusEnum.DRAFT


# --- get_pipeline_status_counts ---


@pytest.mark.asyncio
async def test_get_pipeline_status_counts_non_in_progress(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    await repo.bulk_create(
        [
            {
                "original": "draft 1",
                "tag": "behavior",
                "lang": "ru",
                "status": PhraseStatusEnum.DRAFT,
            },
            {
                "original": "draft 2",
                "tag": "behavior",
                "lang": "ru",
                "status": PhraseStatusEnum.DRAFT,
            },
            {
                "original": "gen done",
                "tag": "behavior",
                "lang": "ru",
                "status": PhraseStatusEnum.GENERATING_DONE,
            },
        ]
    )
    await db_session.commit()

    counts = await repo.get_pipeline_status_counts(stuck_threshold_sec=86400)
    assert counts.get(PhraseStatusEnum.DRAFT.value, 0) == 2
    assert counts.get(PhraseStatusEnum.GENERATING_DONE.value, 0) == 1


@pytest.mark.asyncio
async def test_get_pipeline_status_counts_excludes_loading_done(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    await _insert(db_session, "done phrase", PhraseStatusEnum.LOADING_DONE)

    counts = await repo.get_pipeline_status_counts(stuck_threshold_sec=86400)
    assert PhraseStatusEnum.LOADING_DONE.value not in counts


@pytest.mark.asyncio
async def test_get_pipeline_status_counts_fresh_in_progress_not_stuck(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    await _insert(db_session, "fresh progress", PhraseStatusEnum.GENERATING_IN_PROGRESS)

    counts = await repo.get_pipeline_status_counts(stuck_threshold_sec=86400)
    assert counts.get(PhraseStatusEnum.GENERATING_IN_PROGRESS.value, 0) == 0


@pytest.mark.asyncio
async def test_get_pipeline_status_counts_stale_in_progress_is_stuck(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    phrase_id = await _insert(
        db_session, "stale progress", PhraseStatusEnum.GENERATING_IN_PROGRESS
    )
    await _set_updated_at(
        db_session, phrase_id, datetime.now(timezone.utc) - timedelta(hours=1)
    )

    counts = await repo.get_pipeline_status_counts(stuck_threshold_sec=0)
    assert counts.get(PhraseStatusEnum.GENERATING_IN_PROGRESS.value, 0) == 1


# --- get_sample_per_tag ---


@pytest.mark.asyncio
async def test_get_sample_per_tag_returns_loading_done_phrases(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    await repo.bulk_create(
        [
            {
                "original": f"done phrase {i}",
                "tag": "behavior",
                "lang": "ru",
                "status": PhraseStatusEnum.LOADING_DONE,
            }
            for i in range(5)
        ]
    )
    await db_session.commit()

    result = await repo.get_sample_per_tag(sample_size=3)
    assert all(p.status == PhraseStatusEnum.LOADING_DONE for p in result)


@pytest.mark.asyncio
async def test_get_sample_per_tag_respects_sample_size(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    await repo.bulk_create(
        [
            {
                "original": f"done {i}",
                "tag": "behavior",
                "lang": "ru",
                "status": PhraseStatusEnum.LOADING_DONE,
            }
            for i in range(6)
        ]
    )
    await db_session.commit()

    result = await repo.get_sample_per_tag(sample_size=3)
    assert len(result) <= 3


# --- get_sample_per_tag(load_data=True) ---


@pytest.mark.asyncio
async def test_get_sample_per_tag_with_load_data_returns_phrases(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    await repo.bulk_create(
        [
            {
                "original": f"loaded {i}",
                "tag": "behavior",
                "lang": "ru",
                "status": PhraseStatusEnum.LOADING_DONE,
            }
            for i in range(4)
        ]
    )
    await db_session.commit()

    result = await repo.get_sample_per_tag(sample_size=2, load_data=True)
    assert len(result) <= 2
    assert all(p.status == PhraseStatusEnum.LOADING_DONE for p in result)


@pytest.mark.asyncio
async def test_get_sample_per_tag_with_load_data_returns_empty_when_no_loading_done(
    db_session: AsyncSession,
) -> None:
    repo = PhraseRepository(db_session)
    await _insert(db_session, "draft only", PhraseStatusEnum.DRAFT)

    result = await repo.get_sample_per_tag(sample_size=5, load_data=True)
    assert result == []
