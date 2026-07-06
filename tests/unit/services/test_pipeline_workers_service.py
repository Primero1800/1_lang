import asyncio
import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.common.enums import WorkerRoleEnum
from app.core.config import settings
from app.services.pipeline_workers_service import PipelineWorkersService


@pytest.fixture
def workers_service() -> PipelineWorkersService:
    return PipelineWorkersService(queue_client=AsyncMock())


@pytest.fixture
def running_service() -> PipelineWorkersService:
    svc = PipelineWorkersService(queue_client=AsyncMock())
    svc._is_running = True
    return svc


async def _messages(*msgs):
    for msg in msgs:
        yield msg


def _pubsub(*messages) -> MagicMock:
    ps = MagicMock()
    ps.listen.return_value = _messages(*messages)
    ps.unsubscribe = AsyncMock()
    ps.aclose = AsyncMock()
    return ps


def _snap(role: WorkerRoleEnum, ready: int, last_run: float | None) -> dict:
    return {role.value: {"ready": ready, "last_run": last_run, "ts": int(time.time())}}


# --- _should_run ---
# Logic: False if ready<=0; True if last_run is None;
# otherwise True if ready>=batch_size OR elapsed>=timeout_sec


def test_should_run_false_when_ready_zero(
    workers_service: PipelineWorkersService,
) -> None:
    assert (
        workers_service._should_run(
            WorkerRoleEnum.W2, _snap(WorkerRoleEnum.W2, 0, None)
        )
        is False
    )


def test_should_run_true_on_first_dispatch_with_any_ready(
    workers_service: PipelineWorkersService,
) -> None:
    assert (
        workers_service._should_run(
            WorkerRoleEnum.W2, _snap(WorkerRoleEnum.W2, 1, None)
        )
        is True
    )


def test_should_run_true_when_batch_full_regardless_of_cooldown(
    workers_service: PipelineWorkersService,
) -> None:
    batch_size = settings.PIPELINE_W2_BATCH_SIZE
    snap = _snap(WorkerRoleEnum.W2, batch_size, time.time() - 1)
    assert workers_service._should_run(WorkerRoleEnum.W2, snap) is True


def test_should_run_true_when_cooldown_expired_even_if_batch_not_full(
    workers_service: PipelineWorkersService,
) -> None:
    batch_size = settings.PIPELINE_W2_BATCH_SIZE
    timeout_sec = settings.PIPELINE_W2_TIMEOUT_SEC
    snap = _snap(WorkerRoleEnum.W2, batch_size - 1, time.time() - timeout_sec - 5)
    assert workers_service._should_run(WorkerRoleEnum.W2, snap) is True


def test_should_run_false_when_batch_not_full_and_cooldown_not_expired(
    workers_service: PipelineWorkersService,
) -> None:
    batch_size = settings.PIPELINE_W2_BATCH_SIZE
    timeout_sec = settings.PIPELINE_W2_TIMEOUT_SEC
    snap = _snap(
        WorkerRoleEnum.W2, max(1, batch_size - 1), time.time() - timeout_sec // 2
    )
    assert workers_service._should_run(WorkerRoleEnum.W2, snap) is False


def test_should_run_false_when_worker_missing_from_snapshot(
    workers_service: PipelineWorkersService,
) -> None:
    assert workers_service._should_run(WorkerRoleEnum.W2, {}) is False


def test_should_run_checks_correct_role_key(
    workers_service: PipelineWorkersService,
) -> None:
    snap = {
        "w2": {"ready": 0, "last_run": None, "ts": 0},
        "w3": {"ready": 99, "last_run": None, "ts": 0},
    }
    assert workers_service._should_run(WorkerRoleEnum.W2, snap) is False
    assert workers_service._should_run(WorkerRoleEnum.W3, snap) is True


# --- _run_worker ---


_BASE_DEPS_PATCH = "app.services.pipeline_workers_service.get_base_deps_standalone"
_SLEEP_PATCH = "app.services.pipeline_workers_service.asyncio.sleep"


@pytest.mark.asyncio
async def test_run_worker_skips_non_message_type(
    running_service: PipelineWorkersService,
) -> None:
    ps = _pubsub({"type": "subscribe", "data": None})
    running_service._queue_client.subscribe.return_value = ps
    running_service._execute = AsyncMock()

    with patch(_BASE_DEPS_PATCH, AsyncMock(return_value=MagicMock())):
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            await running_service._run_worker(WorkerRoleEnum.W2)

    running_service._execute.assert_not_called()


@pytest.mark.asyncio
async def test_run_worker_skips_malformed_json(
    running_service: PipelineWorkersService,
) -> None:
    ps = _pubsub({"type": "message", "data": "not-valid-json{{{"})
    running_service._queue_client.subscribe.return_value = ps
    running_service._execute = AsyncMock()

    with patch(_BASE_DEPS_PATCH, AsyncMock(return_value=MagicMock())):
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            await running_service._run_worker(WorkerRoleEnum.W2)

    running_service._execute.assert_not_called()


@pytest.mark.asyncio
async def test_run_worker_calls_execute_when_should_run_true(
    running_service: PipelineWorkersService,
) -> None:
    snapshot = json.dumps({WorkerRoleEnum.W2.value: {"ready": 99, "last_run": None}})
    ps = _pubsub({"type": "message", "data": snapshot})
    running_service._queue_client.subscribe.return_value = ps
    running_service._execute = AsyncMock()

    with patch(_BASE_DEPS_PATCH, AsyncMock(return_value=MagicMock())):
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            with patch.object(running_service, "_should_run", return_value=True):
                await running_service._run_worker(WorkerRoleEnum.W2)

    running_service._execute.assert_called_once()


@pytest.mark.asyncio
async def test_run_worker_skips_execute_when_should_not_run(
    running_service: PipelineWorkersService,
) -> None:
    snapshot = json.dumps({WorkerRoleEnum.W2.value: {"ready": 1, "last_run": 0}})
    ps = _pubsub({"type": "message", "data": snapshot})
    running_service._queue_client.subscribe.return_value = ps
    running_service._execute = AsyncMock()

    with patch(_BASE_DEPS_PATCH, AsyncMock(return_value=MagicMock())):
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            with patch.object(running_service, "_should_run", return_value=False):
                await running_service._run_worker(WorkerRoleEnum.W2)

    running_service._execute.assert_not_called()


@pytest.mark.asyncio
async def test_run_worker_breaks_when_not_running() -> None:
    svc = PipelineWorkersService(queue_client=AsyncMock())
    # _is_running defaults to False
    snapshot = json.dumps({WorkerRoleEnum.W2.value: {"ready": 99, "last_run": None}})
    ps = _pubsub({"type": "message", "data": snapshot})
    svc._queue_client.subscribe.return_value = ps
    svc._execute = AsyncMock()

    with patch(_BASE_DEPS_PATCH, AsyncMock(return_value=MagicMock())):
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            await svc._run_worker(WorkerRoleEnum.W2)

    svc._execute.assert_not_called()


@pytest.mark.asyncio
async def test_run_worker_handles_cancelled_error(
    running_service: PipelineWorkersService,
) -> None:
    async def _cancelled():
        raise asyncio.CancelledError()
        yield  # makes this an async generator

    ps = MagicMock()
    ps.listen.return_value = _cancelled()
    ps.unsubscribe = AsyncMock()
    ps.aclose = AsyncMock()
    running_service._queue_client.subscribe.return_value = ps

    with patch(_BASE_DEPS_PATCH, AsyncMock(return_value=MagicMock())):
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            await running_service._run_worker(WorkerRoleEnum.W2)  # must not raise


@pytest.mark.asyncio
async def test_run_worker_always_unsubscribes(
    running_service: PipelineWorkersService,
) -> None:
    ps = _pubsub()  # empty: loop ends immediately
    running_service._queue_client.subscribe.return_value = ps

    with patch(_BASE_DEPS_PATCH, AsyncMock(return_value=MagicMock())):
        with patch(_SLEEP_PATCH, new_callable=AsyncMock):
            await running_service._run_worker(WorkerRoleEnum.W2)

    ps.unsubscribe.assert_called_once()
    ps.aclose.assert_called_once()


# --- _execute ---


@pytest.mark.asyncio
async def test_execute_instantiates_command_with_config_batch_size(
    workers_service: PipelineWorkersService,
) -> None:
    mock_command = AsyncMock()
    mock_command_class = MagicMock(return_value=mock_command)

    with patch.dict(
        "app.services.pipeline_workers_service._WORKER_CONFIGS",
        {
            WorkerRoleEnum.W2: {
                "batch_size": 5,
                "timeout_sec": 60,
                "command_class": mock_command_class,
            }
        },
    ):
        mock_base_deps = MagicMock()
        await workers_service._execute(WorkerRoleEnum.W2, mock_base_deps)

    mock_command_class.assert_called_once_with(mock_base_deps, 5)


@pytest.mark.asyncio
async def test_execute_calls_command_execute(
    workers_service: PipelineWorkersService,
) -> None:
    mock_command = AsyncMock()
    mock_command_class = MagicMock(return_value=mock_command)

    with patch.dict(
        "app.services.pipeline_workers_service._WORKER_CONFIGS",
        {
            WorkerRoleEnum.W2: {
                "batch_size": 5,
                "timeout_sec": 60,
                "command_class": mock_command_class,
            }
        },
    ):
        await workers_service._execute(WorkerRoleEnum.W2, MagicMock())

    mock_command.execute.assert_called_once()
