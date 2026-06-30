import time
import pytest
from unittest.mock import MagicMock

from app.adapters.queue_client import MessageQueueClientAbstract
from app.common.enums import WorkerRoleEnum
from app.core.config import settings
from app.services.pipeline_workers_service import PipelineWorkersService


@pytest.fixture
def workers_service() -> PipelineWorkersService:
    return PipelineWorkersService(
        queue_client=MagicMock(spec=MessageQueueClientAbstract)
    )


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
