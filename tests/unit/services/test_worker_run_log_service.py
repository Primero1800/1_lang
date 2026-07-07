from unittest.mock import AsyncMock

import pytest

from app.common.enums import WorkerStatusEnum
from app.services.worker_run_log_service import WorkerRunLogService


def _make_uow() -> AsyncMock:
    uow = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.worker_run_log_repository = AsyncMock()
    return uow


# --- start ---


@pytest.mark.asyncio
async def test_start_calls_create_and_returns_id() -> None:
    uow = _make_uow()
    uow.worker_run_log_repository.create = AsyncMock(return_value=42)
    service = WorkerRunLogService(uow=uow)

    log_id = await service.start(worker="w2_generate", batch_size=7)

    uow.worker_run_log_repository.create.assert_called_once_with(
        worker="w2_generate", batch_size=7
    )
    assert log_id == 42


@pytest.mark.asyncio
async def test_start_passes_none_batch_size() -> None:
    uow = _make_uow()
    uow.worker_run_log_repository.create = AsyncMock(return_value=1)
    service = WorkerRunLogService(uow=uow)

    await service.start(worker="token_worker")

    uow.worker_run_log_repository.create.assert_called_once_with(
        worker="token_worker", batch_size=None
    )


# --- finish ---


@pytest.mark.asyncio
async def test_finish_calls_repo_finish_with_all_args() -> None:
    uow = _make_uow()
    service = WorkerRunLogService(uow=uow)

    await service.finish(
        log_id=5, status=WorkerStatusEnum.DONE, result={"processed": 3}
    )

    uow.worker_run_log_repository.finish.assert_called_once_with(
        log_id=5, status=WorkerStatusEnum.DONE, result={"processed": 3}
    )


@pytest.mark.asyncio
async def test_finish_with_failed_status() -> None:
    uow = _make_uow()
    service = WorkerRunLogService(uow=uow)

    await service.finish(
        log_id=7, status=WorkerStatusEnum.FAILED, result={"error": "timeout"}
    )

    uow.worker_run_log_repository.finish.assert_called_once_with(
        log_id=7, status=WorkerStatusEnum.FAILED, result={"error": "timeout"}
    )


@pytest.mark.asyncio
async def test_finish_passes_none_result() -> None:
    uow = _make_uow()
    service = WorkerRunLogService(uow=uow)

    await service.finish(log_id=1, status=WorkerStatusEnum.DONE)

    uow.worker_run_log_repository.finish.assert_called_once_with(
        log_id=1, status=WorkerStatusEnum.DONE, result=None
    )


# --- abandon_running ---


@pytest.mark.asyncio
async def test_abandon_running_calls_repo_and_returns_count() -> None:
    uow = _make_uow()
    uow.worker_run_log_repository.abandon_running = AsyncMock(return_value=3)
    service = WorkerRunLogService(uow=uow)

    count = await service.abandon_running(worker="token_worker")

    uow.worker_run_log_repository.abandon_running.assert_called_once_with(
        worker="token_worker"
    )
    assert count == 3


@pytest.mark.asyncio
async def test_abandon_running_returns_zero_when_nothing_running() -> None:
    uow = _make_uow()
    uow.worker_run_log_repository.abandon_running = AsyncMock(return_value=0)
    service = WorkerRunLogService(uow=uow)

    count = await service.abandon_running(worker="w2_generate")

    assert count == 0
