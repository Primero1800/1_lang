import abc
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio.client import PubSub

from app.core.config import settings


class MessageQueueClientAbstract(abc.ABC):
    """Abstract base class for message queue client implementations"""

    @abc.abstractmethod
    async def start(self) -> None:
        """Start the client and initialise any internal resources

        :returns:
            None
        """

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop the client and release any internal resources

        :returns:
            None
        """

    @abc.abstractmethod
    async def check_connection(self) -> None:
        """Verify that the connection to the queue broker is alive

        :raise:
            Exception: if the broker is unreachable

        :returns:
            None
        """

    @abc.abstractmethod
    async def xadd(self, stream: str, fields: dict[str, str]) -> None:
        """Publish a message to a stream

        :param:
            stream: stream name
            fields: message payload as string key-value pairs

        :returns:
            None
        """

    @abc.abstractmethod
    async def xgroup_create(self, stream: str, group: str) -> None:
        """Create a consumer group, creating the stream if it does not exist

        :param:
            stream: stream name
            group: consumer group name

        :returns:
            None
        """

    @abc.abstractmethod
    async def xreadgroup(
        self,
        group: str,
        consumer: str,
        stream: str,
        count: int = 10,
        cursor: str = ">",
    ) -> list[Any]:
        """Read messages from a stream as a consumer group member

        :param:
            group: consumer group name
            consumer: consumer identifier
            stream: stream name
            count: maximum number of messages to fetch
            cursor: '>' for new undelivered messages, '0' to reclaim pending

        :returns:
            messages: list of stream entries (broker-specific format)
        """

    @abc.abstractmethod
    async def xack(self, stream: str, group: str, *message_ids: str) -> None:
        """Acknowledge processed messages

        :param:
            stream: stream name
            group: consumer group name
            message_ids: one or more message IDs to acknowledge

        :returns:
            None
        """

    @abc.abstractmethod
    async def publish(self, channel: str, message: str) -> None:
        """Publish a message to a Pub/Sub channel

        :param:
            channel: channel name
            message: message payload as string

        :returns:
            None
        """

    @abc.abstractmethod
    async def subscribe(self, channel: str) -> PubSub:
        """Subscribe to a Pub/Sub channel and return the subscription object

        :param:
            channel: channel name to subscribe to

        :returns:
            pubsub: active PubSub subscription object
        """


class RedisClient(MessageQueueClientAbstract):
    """Async Redis client wrapper"""

    def __init__(self) -> None:
        """Initialize with no active connection (call start() before use)

        :returns:
            None
        """
        self._client: aioredis.Redis | None = None

    async def start(self) -> None:
        """Create the Redis connection pool

        :returns:
            None
        """
        self._client = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
        )

    async def stop(self) -> None:
        """Close the Redis connection

        :returns:
            None
        """
        if self._client:
            await self._client.aclose()
            self._client = None

    async def check_connection(self) -> None:
        """Ping Redis to verify the connection is alive

        :returns:
            None
        """
        await self.client.ping()

    async def xadd(self, stream: str, fields: dict[str, str]) -> None:
        """Append a message to a Redis stream

        :param:
            stream: stream name
            fields: message payload

        :returns:
            None
        """
        await self.client.xadd(stream, fields)  # type: ignore[arg-type]

    async def xgroup_create(self, stream: str, group: str) -> None:
        """Create a consumer group, creating the stream if it does not exist

        :param:
            stream: stream name
            group: consumer group name

        :returns:
            None
        """
        try:
            await self.client.xgroup_create(stream, group, id="$", mkstream=True)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def xreadgroup(
        self,
        group: str,
        consumer: str,
        stream: str,
        count: int = 10,
        cursor: str = ">",
    ) -> list[Any]:
        """Read messages from a stream as a consumer group member

        :param:
            group: consumer group name
            consumer: consumer identifier
            stream: stream name
            count: maximum number of messages to fetch
            cursor: '>' for new undelivered messages, '0' to reclaim pending

        :returns:
            messages: list of (stream_name, [(id, fields), ...]) tuples
        """
        result = await self.client.xreadgroup(
            group, consumer, {stream: cursor}, count=count
        )
        return result or []  # type: ignore[return-value]

    async def xack(self, stream: str, group: str, *message_ids: str) -> None:
        """Acknowledge processed messages

        :param:
            stream: stream name
            group: consumer group name
            message_ids: one or more message IDs to acknowledge

        :returns:
            None
        """
        await self.client.xack(stream, group, *message_ids)

    async def publish(self, channel: str, message: str) -> None:
        """Publish a message to a Redis Pub/Sub channel

        :param:
            channel: channel name
            message: message payload as string

        :returns:
            None
        """
        await self.client.publish(channel, message)

    async def subscribe(self, channel: str) -> PubSub:
        """Create a new PubSub object and subscribe to the given channel

        :param:
            channel: channel name to subscribe to

        :returns:
            pubsub: active PubSub subscription object
        """
        pubsub = self.client.pubsub()
        await pubsub.subscribe(channel)
        return pubsub

    @property
    def client(self) -> aioredis.Redis:
        """Return the active Redis connection, raising if not started

        :raise:
            RuntimeError: if start() has not been called yet

        :returns:
            client: the underlying aioredis.Redis instance
        """
        if not self._client:
            raise RuntimeError("RedisClient is not started")
        return self._client
