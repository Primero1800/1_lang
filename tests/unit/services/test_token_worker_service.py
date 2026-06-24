import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from app.adapters.queue_client import MessageQueueClientAbstract
from app.core.config import settings
from app.services.token_worker_service import TokenWorkerService


@pytest.fixture
def mock_queue_client() -> MessageQueueClientAbstract:
    """
    :returns:
        queue_client: AsyncMock with all stream methods wired
    """
    q = AsyncMock(spec=MessageQueueClientAbstract)
    return q


@pytest.fixture
def service(mock_queue_client) -> TokenWorkerService:
    """
    :param:
        mock_queue_client: injected queue client

    :returns:
        service: TokenWorkerService under test
    """
    return TokenWorkerService(queue_client=mock_queue_client)


def _make_mock_uow() -> AsyncMock:
    mock_uow = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=None)
    mock_uow.worker_run_log_repository = AsyncMock()
    mock_uow.worker_run_log_repository.create = AsyncMock(return_value=1)
    mock_uow.ai_token_usage_repository = AsyncMock()
    return mock_uow


def _make_messages(entries: list[tuple[str, dict]]) -> list:
    """Build the xreadgroup return value format: [(stream, [(msg_id, fields), ...])]"""
    return [
        (settings.REDIS_TOKENS_STREAM, [(msg_id, fields) for msg_id, fields in entries])
    ]


# --- _process ---


@pytest.mark.asyncio
async def test_process_returns_false_when_no_messages(
    service: TokenWorkerService, mock_queue_client
) -> None:
    """
    :param:
        service: service under test
        mock_queue_client: returns empty list from xreadgroup

    :returns:
        None
    """
    mock_queue_client.xreadgroup.return_value = []
    result = await service._process(cursor=">")
    assert result is False


@pytest.mark.asyncio
async def test_process_aggregates_single_message_and_persists(
    service: TokenWorkerService, mock_queue_client
) -> None:
    """
    :param:
        service: service under test
        mock_queue_client: returns one message

    :returns:
        None
    """
    mock_queue_client.xreadgroup.return_value = _make_messages(
        [
            (
                "1-0",
                {
                    "model": "mistral-large",
                    "operation": "w2_generate",
                    "input_tokens": "100",
                    "output_tokens": "50",
                },
            ),
        ]
    )
    mock_uow = _make_mock_uow()

    with patch(
        "app.services.token_worker_service.get_uow_factory",
        AsyncMock(return_value=mock_uow),
    ):
        result = await service._process(cursor=">")

    assert result is True
    mock_uow.ai_token_usage_repository.bulk_accumulate.assert_called_once()
    mock_queue_client.xack.assert_called_once_with(
        settings.REDIS_TOKENS_STREAM, settings.REDIS_TOKENS_GROUP, "1-0"
    )


@pytest.mark.asyncio
async def test_process_sums_tokens_for_same_key(
    service: TokenWorkerService, mock_queue_client
) -> None:
    """Two messages with the same (model, operation, name) must be summed into one row

    :param:
        service: service under test
        mock_queue_client: returns two messages for the same key

    :returns:
        None
    """
    mock_queue_client.xreadgroup.return_value = _make_messages(
        [
            (
                "1-0",
                {
                    "model": "m",
                    "operation": "op",
                    "input_tokens": "100",
                    "output_tokens": "50",
                },
            ),
            (
                "2-0",
                {
                    "model": "m",
                    "operation": "op",
                    "input_tokens": "200",
                    "output_tokens": "30",
                },
            ),
        ]
    )
    mock_uow = _make_mock_uow()

    with patch(
        "app.services.token_worker_service.get_uow_factory",
        AsyncMock(return_value=mock_uow),
    ):
        await service._process(cursor=">")

    call_args = mock_uow.ai_token_usage_repository.bulk_accumulate.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0]["input_tokens"] == 300
    assert call_args[0]["output_tokens"] == 80


@pytest.mark.asyncio
async def test_process_keeps_messages_pending_on_db_error(
    service: TokenWorkerService, mock_queue_client
) -> None:
    """DB failure must not ack messages — they stay pending for retry

    :param:
        service: service under test
        mock_queue_client: checked that xack is NOT called

    :returns:
        None
    """
    mock_queue_client.xreadgroup.return_value = _make_messages(
        [
            (
                "1-0",
                {
                    "model": "m",
                    "operation": "op",
                    "input_tokens": "10",
                    "output_tokens": "5",
                },
            ),
        ]
    )
    mock_uow = _make_mock_uow()
    mock_uow.__aenter__.side_effect = Exception("db down")

    with patch(
        "app.services.token_worker_service.get_uow_factory",
        AsyncMock(return_value=mock_uow),
    ):
        with pytest.raises(Exception, match="db down"):
            await service._process(cursor=">")

    mock_queue_client.xack.assert_not_called()


@pytest.mark.asyncio
async def test_process_uses_custom_name_field(
    service: TokenWorkerService, mock_queue_client
) -> None:
    """Messages with an explicit 'name' field must be grouped under that name

    :param:
        service: service under test
        mock_queue_client: returns message with custom name

    :returns:
        None
    """
    mock_queue_client.xreadgroup.return_value = _make_messages(
        [
            (
                "1-0",
                {
                    "model": "m",
                    "operation": "op",
                    "name": "alice",
                    "input_tokens": "10",
                    "output_tokens": "5",
                },
            ),
        ]
    )
    mock_uow = _make_mock_uow()

    with patch(
        "app.services.token_worker_service.get_uow_factory",
        AsyncMock(return_value=mock_uow),
    ):
        await service._process(cursor=">")

    call_args = mock_uow.ai_token_usage_repository.bulk_accumulate.call_args[0][0]
    assert call_args[0]["name"] == "alice"


# --- start ---


@pytest.mark.asyncio
async def test_start_creates_consumer_group_and_abandons_stale(
    service: TokenWorkerService, mock_queue_client
) -> None:
    """start() must create the consumer group and clean up stale RUNNING logs

    :param:
        service: service under test
        mock_queue_client: checked for xgroup_create call

    :returns:
        None
    """
    mock_uow = AsyncMock()
    mock_wls = AsyncMock()
    mock_wls.abandon_running = AsyncMock(return_value=0)

    with (
        patch(
            "app.services.token_worker_service.get_uow_factory",
            AsyncMock(return_value=mock_uow),
        ),
        patch(
            "app.services.token_worker_service.WorkerRunLogService",
            return_value=mock_wls,
        ),
    ):
        await service.start()

    try:
        mock_queue_client.xgroup_create.assert_called_once_with(
            settings.REDIS_TOKENS_STREAM, settings.REDIS_TOKENS_GROUP
        )
        mock_wls.abandon_running.assert_called_once()
    finally:
        if service._task:
            service._task.cancel()
            try:
                await service._task
            except (asyncio.CancelledError, Exception):
                pass


# --- stop ---


@pytest.mark.asyncio
async def test_stop_cancels_running_task(service: TokenWorkerService) -> None:
    """stop() must cancel the background polling task

    :param:
        service: service with an active task

    :returns:
        None
    """

    async def _long_running():
        await asyncio.sleep(3600)

    service._task = asyncio.create_task(_long_running())
    await service.stop()
    assert service._task.cancelled()


@pytest.mark.asyncio
async def test_stop_is_safe_without_task(service: TokenWorkerService) -> None:
    """stop() must not raise when called before start()

    :param:
        service: service with no active task

    :returns:
        None
    """
    await service.stop()
