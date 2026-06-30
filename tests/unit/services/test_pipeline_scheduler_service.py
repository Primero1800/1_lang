import json
import time
from datetime import datetime
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.common.enums import PhraseStatusEnum
from app.core.config import settings
from app.services.base import BaseDeps
from app.services.pipeline_scheduler_service import PipelineSchedulerService


@pytest.fixture
def scheduler_service() -> PipelineSchedulerService:
    base_deps = MagicMock(spec=BaseDeps)
    base_deps.uow_factory = MagicMock()
    base_deps.ai_client = MagicMock()
    base_deps.ai_client2 = MagicMock()
    base_deps.vector_client = MagicMock()
    base_deps.vector_client_main = MagicMock()
    base_deps.queue_client = AsyncMock()
    return PipelineSchedulerService(base_deps=base_deps)


def _empty_last_runs() -> dict:
    return {"w2": None, "w3": None, "w4": None, "w5": None}


# --- _build_snapshot ---


@pytest.mark.asyncio
async def test_build_snapshot_contains_all_workers(
    scheduler_service: PipelineSchedulerService,
) -> None:
    snapshot = await scheduler_service._build_snapshot({}, _empty_last_runs())
    assert set(snapshot.keys()) == {"w2", "w3", "w4", "w5"}


@pytest.mark.asyncio
async def test_build_snapshot_sums_base_and_in_progress_for_w2(
    scheduler_service: PipelineSchedulerService,
) -> None:
    counts = {
        PhraseStatusEnum.DRAFT.value: 3,
        PhraseStatusEnum.GENERATING_FAILED.value: 2,
        PhraseStatusEnum.GENERATING_IN_PROGRESS.value: 1,
    }
    snapshot = await scheduler_service._build_snapshot(counts, _empty_last_runs())
    assert snapshot["w2"]["ready"] == 6


@pytest.mark.asyncio
async def test_build_snapshot_zeros_when_no_counts(
    scheduler_service: PipelineSchedulerService,
) -> None:
    snapshot = await scheduler_service._build_snapshot({}, _empty_last_runs())
    for worker in ("w2", "w3", "w4", "w5"):
        assert snapshot[worker]["ready"] == 0


@pytest.mark.asyncio
async def test_build_snapshot_last_run_none_when_no_history(
    scheduler_service: PipelineSchedulerService,
) -> None:
    snapshot = await scheduler_service._build_snapshot({}, _empty_last_runs())
    assert snapshot["w3"]["last_run"] is None


@pytest.mark.asyncio
async def test_build_snapshot_last_run_is_integer_timestamp(
    scheduler_service: PipelineSchedulerService,
) -> None:
    dt = datetime(2026, 1, 1, 12, 0, 0)
    last_runs = {**_empty_last_runs(), "w2": dt}
    snapshot = await scheduler_service._build_snapshot({}, last_runs)
    assert isinstance(snapshot["w2"]["last_run"], int)
    assert snapshot["w2"]["last_run"] == int(dt.timestamp())


@pytest.mark.asyncio
async def test_build_snapshot_ts_is_current_time(
    scheduler_service: PipelineSchedulerService,
) -> None:
    before = int(time.time())
    snapshot = await scheduler_service._build_snapshot({}, _empty_last_runs())
    after = int(time.time())
    assert before <= snapshot["w2"]["ts"] <= after


# --- _publish_snapshot ---


@pytest.mark.asyncio
async def test_publish_snapshot_calls_publish_once(
    scheduler_service: PipelineSchedulerService,
) -> None:
    snapshot = {"w2": {"ready": 3, "last_run": None, "ts": 1}}
    await scheduler_service._publish_snapshot(snapshot)
    scheduler_service.queue_client.publish.assert_called_once()


@pytest.mark.asyncio
async def test_publish_snapshot_sends_correct_channel_and_json(
    scheduler_service: PipelineSchedulerService,
) -> None:
    snapshot = {"w2": {"ready": 7, "last_run": None, "ts": 1}}
    await scheduler_service._publish_snapshot(snapshot)
    channel, message = scheduler_service.queue_client.publish.call_args[0]
    assert channel == settings.REDIS_PIPELINE_CHANNEL
    assert json.loads(message)["w2"]["ready"] == 7


# --- run ---


@pytest.mark.asyncio
async def test_run_publishes_snapshot(
    scheduler_service: PipelineSchedulerService,
) -> None:
    mock_uow = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=None)
    mock_uow.phrase_repository.get_pipeline_status_counts = AsyncMock(
        return_value={PhraseStatusEnum.DRAFT.value: 5}
    )
    mock_uow.worker_run_log_repository.get_last_runs = AsyncMock(
        return_value=_empty_last_runs()
    )
    scheduler_service.uow_factory = mock_uow

    await scheduler_service.run()

    scheduler_service.queue_client.publish.assert_called_once()


@pytest.mark.asyncio
async def test_run_snapshot_reflects_status_counts(
    scheduler_service: PipelineSchedulerService,
) -> None:
    mock_uow = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=None)
    mock_uow.phrase_repository.get_pipeline_status_counts = AsyncMock(
        return_value={PhraseStatusEnum.DRAFT.value: 10}
    )
    mock_uow.worker_run_log_repository.get_last_runs = AsyncMock(
        return_value=_empty_last_runs()
    )
    scheduler_service.uow_factory = mock_uow

    await scheduler_service.run()

    _, message = scheduler_service.queue_client.publish.call_args[0]
    published = json.loads(message)
    assert published["w2"]["ready"] == 10
