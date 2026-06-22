import pytest
import redis.asyncio as aioredis
from unittest.mock import AsyncMock

from app.adapters.queue_client import RedisClient


@pytest.fixture
def redis_client() -> RedisClient:
    """
    :returns:
        client: RedisClient with a mocked internal aioredis connection
    """
    client = RedisClient()
    client._client = AsyncMock()
    return client


# --- xadd ---


@pytest.mark.asyncio
async def test_xadd_delegates_to_redis(redis_client: RedisClient) -> None:
    """
    :param:
        redis_client: fixture with mocked internal Redis connection

    :returns:
        None
    """
    await redis_client.xadd("my-stream", {"key": "value"})
    redis_client._client.xadd.assert_called_once_with("my-stream", {"key": "value"})


# --- xgroup_create ---


@pytest.mark.asyncio
async def test_xgroup_create_delegates_to_redis(redis_client: RedisClient) -> None:
    """
    :param:
        redis_client: fixture with mocked internal Redis connection

    :returns:
        None
    """
    await redis_client.xgroup_create("my-stream", "my-group")
    redis_client._client.xgroup_create.assert_called_once_with(
        "my-stream", "my-group", id="$", mkstream=True
    )


@pytest.mark.asyncio
async def test_xgroup_create_swallows_busygroup_error(
    redis_client: RedisClient,
) -> None:
    """BUSYGROUP means the group already exists — should be silently ignored

    :param:
        redis_client: fixture with mocked internal Redis connection

    :returns:
        None
    """
    redis_client._client.xgroup_create.side_effect = aioredis.ResponseError(
        "BUSYGROUP Consumer Group name already exists"
    )
    await redis_client.xgroup_create("my-stream", "my-group")


@pytest.mark.asyncio
async def test_xgroup_create_reraises_other_errors(redis_client: RedisClient) -> None:
    """Non-BUSYGROUP ResponseError should propagate

    :param:
        redis_client: fixture with mocked internal Redis connection

    :returns:
        None
    """
    redis_client._client.xgroup_create.side_effect = aioredis.ResponseError(
        "WRONGTYPE Operation against a key holding the wrong kind of value"
    )
    with pytest.raises(aioredis.ResponseError):
        await redis_client.xgroup_create("my-stream", "my-group")


# --- xreadgroup ---


@pytest.mark.asyncio
async def test_xreadgroup_returns_messages(redis_client: RedisClient) -> None:
    """
    :param:
        redis_client: fixture with mocked internal Redis connection

    :returns:
        None
    """
    fake_messages = [("my-stream", [("1-0", {"key": "val"})])]
    redis_client._client.xreadgroup.return_value = fake_messages

    result = await redis_client.xreadgroup(
        group="grp", consumer="c1", stream="my-stream", count=5
    )

    redis_client._client.xreadgroup.assert_called_once_with(
        "grp", "c1", {"my-stream": ">"}, count=5
    )
    assert result == fake_messages


@pytest.mark.asyncio
async def test_xreadgroup_returns_empty_list_when_none(
    redis_client: RedisClient,
) -> None:
    """Redis returns None when no messages are available

    :param:
        redis_client: fixture with mocked internal Redis connection

    :returns:
        None
    """
    redis_client._client.xreadgroup.return_value = None
    result = await redis_client.xreadgroup(
        group="grp", consumer="c1", stream="my-stream"
    )
    assert result == []


@pytest.mark.asyncio
async def test_xreadgroup_passes_custom_cursor(redis_client: RedisClient) -> None:
    """cursor='0' is used to re-read pending (unacknowledged) messages

    :param:
        redis_client: fixture with mocked internal Redis connection

    :returns:
        None
    """
    redis_client._client.xreadgroup.return_value = []
    await redis_client.xreadgroup(
        group="grp", consumer="c1", stream="my-stream", cursor="0"
    )
    redis_client._client.xreadgroup.assert_called_once_with(
        "grp", "c1", {"my-stream": "0"}, count=10
    )


# --- xack ---


@pytest.mark.asyncio
async def test_xack_delegates_to_redis(redis_client: RedisClient) -> None:
    """
    :param:
        redis_client: fixture with mocked internal Redis connection

    :returns:
        None
    """
    await redis_client.xack("my-stream", "my-group", "1-0", "2-0")
    redis_client._client.xack.assert_called_once_with(
        "my-stream", "my-group", "1-0", "2-0"
    )


# --- client property ---


def test_client_property_raises_if_not_started() -> None:
    """
    :returns:
        None
    """
    client = RedisClient()
    with pytest.raises(RuntimeError, match="not started"):
        _ = client.client
