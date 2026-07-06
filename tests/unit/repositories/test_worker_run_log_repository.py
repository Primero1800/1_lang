import pytest
from sqlalchemy import select

from app.common.enums import WorkerStatusEnum
from app.models.worker_run_log import WorkerRunLog
from app.repositories.worker_run_log_repository import WorkerRunLogRepository

# --- create ---


@pytest.mark.asyncio
async def test_create_inserts_running_log(test_session_maker, empty_db) -> None:
    """
    :param:
        test_session_maker: async session factory (testcontainer)
        empty_db: fresh schema per test

    :returns:
        None
    """
    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        log_id = await repo.create("test_worker", batch_size=10)
        await session.commit()

    async with test_session_maker() as session:
        row = await session.get(WorkerRunLog, log_id)

    assert row is not None
    assert row.worker == "test_worker"
    assert row.status == WorkerStatusEnum.RUNNING
    assert row.batch_size == 10
    assert row.finished_at is None


@pytest.mark.asyncio
async def test_create_returns_integer_id(test_session_maker, empty_db) -> None:
    """
    :param:
        test_session_maker: async session factory
        empty_db: fresh schema

    :returns:
        None
    """
    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        log_id = await repo.create("w2_generate")
        await session.commit()

    assert isinstance(log_id, int)
    assert log_id > 0


# --- finish ---


@pytest.mark.asyncio
async def test_finish_updates_status_to_done(test_session_maker, empty_db) -> None:
    """
    :param:
        test_session_maker: async session factory
        empty_db: fresh schema

    :returns:
        None
    """
    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        log_id = await repo.create("test_worker")
        await repo.finish(log_id, WorkerStatusEnum.DONE, result={"messages": 5})
        await session.commit()

    async with test_session_maker() as session:
        row = await session.get(WorkerRunLog, log_id)

    assert row.status == WorkerStatusEnum.DONE
    assert row.result == {"messages": 5}
    assert row.finished_at is not None


@pytest.mark.asyncio
async def test_finish_with_failed_status(test_session_maker, empty_db) -> None:
    """
    :param:
        test_session_maker: async session factory
        empty_db: fresh schema

    :returns:
        None
    """
    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        log_id = await repo.create("test_worker")
        await repo.finish(log_id, WorkerStatusEnum.FAILED, result={"error": "timeout"})
        await session.commit()

    async with test_session_maker() as session:
        row = await session.get(WorkerRunLog, log_id)

    assert row.status == WorkerStatusEnum.FAILED
    assert row.result["error"] == "timeout"


# --- abandon_running ---


@pytest.mark.asyncio
async def test_abandon_running_marks_all_running_as_failed(
    test_session_maker, empty_db
) -> None:
    """
    :param:
        test_session_maker: async session factory
        empty_db: fresh schema

    :returns:
        None
    """
    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        await repo.create("token_worker")
        await repo.create("token_worker")
        await session.commit()

    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        count = await repo.abandon_running("token_worker")
        await session.commit()

    assert count == 2

    async with test_session_maker() as session:
        result = await session.execute(
            select(WorkerRunLog).where(WorkerRunLog.worker == "token_worker")
        )
        rows = result.scalars().all()

    assert all(r.status == WorkerStatusEnum.FAILED for r in rows)
    assert all(r.finished_at is not None for r in rows)


@pytest.mark.asyncio
async def test_abandon_running_ignores_other_workers(
    test_session_maker, empty_db
) -> None:
    """Only logs for the given worker name should be updated

    :param:
        test_session_maker: async session factory
        empty_db: fresh schema

    :returns:
        None
    """
    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        await repo.create("token_worker")
        await repo.create("w2_generate")
        await session.commit()

    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        count = await repo.abandon_running("token_worker")
        await session.commit()

    assert count == 1

    async with test_session_maker() as session:
        result = await session.execute(
            select(WorkerRunLog).where(WorkerRunLog.worker == "w2_generate")
        )
        row = result.scalars().first()

    assert row.status == WorkerStatusEnum.RUNNING


@pytest.mark.asyncio
async def test_abandon_running_returns_zero_when_nothing_running(
    test_session_maker, empty_db
) -> None:
    """
    :param:
        test_session_maker: async session factory
        empty_db: fresh schema

    :returns:
        None
    """
    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        count = await repo.abandon_running("token_worker")
        await session.commit()

    assert count == 0


# --- get_last_runs ---


@pytest.mark.asyncio
async def test_get_last_runs_returns_none_for_worker_with_no_logs(
    test_session_maker, empty_db
) -> None:
    """
    :param:
        test_session_maker: async session factory
        empty_db: fresh schema

    :returns:
        None
    """
    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        result = await repo.get_last_runs(["w2_generate"])

    assert result == {"w2_generate": None}


@pytest.mark.asyncio
async def test_get_last_runs_returns_finished_at_for_done_log(
    test_session_maker, empty_db
) -> None:
    """
    :param:
        test_session_maker: async session factory
        empty_db: fresh schema

    :returns:
        None
    """
    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        log_id = await repo.create("w2_generate")
        await repo.finish(log_id, WorkerStatusEnum.DONE, result={})
        await session.commit()

    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        result = await repo.get_last_runs(["w2_generate"])

    assert result["w2_generate"] is not None


@pytest.mark.asyncio
async def test_get_last_runs_ignores_running_logs(test_session_maker, empty_db) -> None:
    """Only DONE logs should contribute to last_run — RUNNING logs are skipped.

    :param:
        test_session_maker: async session factory
        empty_db: fresh schema

    :returns:
        None
    """
    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        await repo.create("w3_translate")
        await session.commit()

    async with test_session_maker() as session:
        repo = WorkerRunLogRepository(session)
        result = await repo.get_last_runs(["w3_translate"])

    assert result["w3_translate"] is None
