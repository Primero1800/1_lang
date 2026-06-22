import abc

import redis.asyncio as aioredis

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
