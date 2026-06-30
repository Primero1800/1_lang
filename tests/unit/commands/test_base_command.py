import pytest
from unittest.mock import AsyncMock, MagicMock

from app.commands.base import BaseCommand
from app.common.enums import WorkerRoleEnum, WorkerStatusEnum
from app.services.base import BaseDeps


class _ConcreteCommand(BaseCommand):
    _ROLE = WorkerRoleEnum.W2

    def __init__(self, base_deps, batch_size, raises=None):
        super().__init__(base_deps, batch_size)
        self._raises = raises

    async def _do_execute(self) -> dict:
        if self._raises:
            raise self._raises
        return {"processed": 2, "failed": 0, "skipped": 0}


@pytest.fixture
def mock_uow() -> AsyncMock:
    uow = AsyncMock()
    uow.worker_run_log_repository = AsyncMock()
    uow.worker_run_log_repository.create = AsyncMock(return_value=42)
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    return uow


@pytest.fixture
def base_deps(mock_uow: AsyncMock) -> MagicMock:
    deps = MagicMock(spec=BaseDeps)
    deps.uow_factory = mock_uow
    return deps


# --- execute: success path ---


@pytest.mark.asyncio
async def test_execute_returns_service_result(base_deps: MagicMock) -> None:
    result = await _ConcreteCommand(base_deps, batch_size=5).execute()
    assert result == {"processed": 2, "failed": 0, "skipped": 0}


@pytest.mark.asyncio
async def test_execute_abandons_stuck_runs_before_creating_log(
    base_deps: MagicMock, mock_uow: AsyncMock
) -> None:
    await _ConcreteCommand(base_deps, batch_size=5).execute()
    mock_uow.worker_run_log_repository.abandon_running.assert_called_once_with(
        WorkerRoleEnum.W2.value
    )


@pytest.mark.asyncio
async def test_execute_creates_log_entry(
    base_deps: MagicMock, mock_uow: AsyncMock
) -> None:
    await _ConcreteCommand(base_deps, batch_size=5).execute()
    mock_uow.worker_run_log_repository.create.assert_called_once_with(
        WorkerRoleEnum.W2.value, batch_size=5
    )


@pytest.mark.asyncio
async def test_execute_finishes_log_as_done(
    base_deps: MagicMock, mock_uow: AsyncMock
) -> None:
    await _ConcreteCommand(base_deps, batch_size=5).execute()
    finish_call = mock_uow.worker_run_log_repository.finish.call_args
    assert finish_call[0][0] == 42
    assert finish_call[0][1] == WorkerStatusEnum.DONE


# --- execute: failure path ---


@pytest.mark.asyncio
async def test_execute_failure_returns_error_dict(base_deps: MagicMock) -> None:
    command = _ConcreteCommand(base_deps, batch_size=5, raises=RuntimeError("API down"))
    result = await command.execute()
    assert "error" in result
    assert "API down" in result["error"]


@pytest.mark.asyncio
async def test_execute_failure_finishes_log_as_failed(
    base_deps: MagicMock, mock_uow: AsyncMock
) -> None:
    command = _ConcreteCommand(base_deps, batch_size=5, raises=RuntimeError("API down"))
    await command.execute()
    finish_call = mock_uow.worker_run_log_repository.finish.call_args
    assert finish_call[0][1] == WorkerStatusEnum.FAILED


@pytest.mark.asyncio
async def test_execute_failure_does_not_propagate_exception(
    base_deps: MagicMock,
) -> None:
    command = _ConcreteCommand(base_deps, batch_size=5, raises=ValueError("bad input"))
    result = await command.execute()
    assert isinstance(result, dict)
