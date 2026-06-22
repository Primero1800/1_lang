import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.adapters.ai_client import MistralClient
from app.adapters.queue_client import MessageQueueClientAbstract
from app.core.config import settings


@pytest.fixture
def mock_queue_client() -> MessageQueueClientAbstract:
    """
    :returns:
        queue_client: mock queue client with AsyncMock xadd
    """
    q = MagicMock(spec=MessageQueueClientAbstract)
    q.xadd = AsyncMock()
    return q


@pytest.fixture
def mistral_client(mock_queue_client) -> MistralClient:
    """
    :param:
        mock_queue_client: mocked queue client injected into MistralClient

    :returns:
        client: MistralClient with mocked session and queue
    """
    session = MagicMock()
    return MistralClient(aiohttp_session=session, queue_client=mock_queue_client)


# --- _fire_token_task ---


@pytest.mark.asyncio
async def test_fire_token_task_does_nothing_when_result_is_none(
    mistral_client: MistralClient, mock_queue_client
) -> None:
    """No task should be scheduled when the API returned nothing

    :param:
        mistral_client: client under test
        mock_queue_client: asserted against

    :returns:
        None
    """
    mistral_client._fire_token_task(result=None, model="m", operation="op")
    await asyncio.sleep(0)
    mock_queue_client.xadd.assert_not_called()


@pytest.mark.asyncio
async def test_fire_token_task_does_nothing_when_operation_is_none(
    mistral_client: MistralClient, mock_queue_client
) -> None:
    """No task should be scheduled when operation is None (no tracking needed)

    :param:
        mistral_client: client under test
        mock_queue_client: asserted against

    :returns:
        None
    """
    mistral_client._fire_token_task(
        result={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        model="m",
        operation=None,
    )
    await asyncio.sleep(0)
    mock_queue_client.xadd.assert_not_called()


@pytest.mark.asyncio
async def test_fire_token_task_publishes_token_counts(
    mistral_client: MistralClient, mock_queue_client
) -> None:
    """Should schedule xadd with model, operation, and correct token counts

    :param:
        mistral_client: client under test
        mock_queue_client: asserted against

    :returns:
        None
    """
    result = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    mistral_client._fire_token_task(result=result, model="mistral-large", operation="w2_generate")
    await asyncio.sleep(0)

    mock_queue_client.xadd.assert_called_once_with(
        settings.REDIS_TOKENS_STREAM,
        {
            "model": "mistral-large",
            "operation": "w2_generate",
            "input_tokens": "100",
            "output_tokens": "50",
        },
    )


@pytest.mark.asyncio
async def test_fire_token_task_sets_output_tokens_to_zero_for_embed(
    mistral_client: MistralClient, mock_queue_client
) -> None:
    """Embed calls have no completion tokens — output_tokens must be '0'

    :param:
        mistral_client: client under test
        mock_queue_client: asserted against

    :returns:
        None
    """
    result = {"usage": {"prompt_tokens": 80, "completion_tokens": 40}}
    mistral_client._fire_token_task(
        result=result, model="mistral-embed", operation="w4_embed", is_embed=True
    )
    await asyncio.sleep(0)

    _, call_kwargs = mock_queue_client.xadd.call_args
    fields = mock_queue_client.xadd.call_args[0][1]
    assert fields["output_tokens"] == "0"
    assert fields["input_tokens"] == "80"
