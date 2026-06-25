import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from app.adapters.queue_client import MessageQueueClientAbstract
from app.adapters.vector_client import VectorClientAbstract
from app.core.config import settings
from app.uow import UnitOfWork


@dataclass
class BaseDeps:
    """Infrastructure dependencies shared across all services"""

    uow_factory: UnitOfWork
    vector_client: VectorClientAbstract
    vector_client_main: VectorClientAbstract
    queue_client: MessageQueueClientAbstract


class BaseServiceAbstract(Protocol):
    """Structural protocol defining the common service interface"""

    async def get(self, *args: Any, **kwargs: Any) -> Any: ...

    async def check(self, *args: Any, **kwargs: Any) -> Any: ...

    async def ask(self, *args: Any, **kwargs: Any) -> Any: ...

    async def generate(self, *args: Any, **kwargs: Any) -> Any: ...


class BaseService(BaseServiceAbstract):
    """Base service to be extended by concrete implementations"""

    def __init__(
        self,
        base_deps: BaseDeps,
        uow: UnitOfWork | None = None,
    ) -> None:
        """Initialize the service with infrastructure dependencies

        :param:
            base_deps: container with shared infrastructure clients
            uow: optional request-scoped UnitOfWork session

        :returns:
            None
        """
        self.uow_factory = base_deps.uow_factory
        self.uow = uow
        self.vector_client = base_deps.vector_client
        self.vector_client_main = base_deps.vector_client_main
        self.queue_client = base_deps.queue_client

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        """Retrieve a resource; override in subclass

        :returns:
            result: the retrieved resource
        """
        raise NotImplementedError

    async def check(self, *args: Any, **kwargs: Any) -> Any:
        """Perform a check operation; override in subclass

        :returns:
            result: the check outcome
        """
        raise NotImplementedError

    async def ask(self, *args: Any, **kwargs: Any) -> Any:
        """Send a query; override in subclass

        :returns:
            result: the query response
        """
        raise NotImplementedError

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate content; override in subclass

        :returns:
            result: the generated content
        """
        raise NotImplementedError

    def _queue_token_usage(
        self,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> None:
        """Fire-and-forget: publish token usage to Redis Streams

        :param:
            model: model identifier string
            operation: pipeline operation name (e.g. 'w2_generate')
            input_tokens: number of input tokens consumed
            output_tokens: number of output tokens consumed (0 for embeddings)

        :returns:
            None
        """
        asyncio.create_task(
            self.queue_client.xadd(
                settings.REDIS_TOKENS_STREAM,
                {
                    "model": model,
                    "operation": operation,
                    "input_tokens": str(input_tokens),
                    "output_tokens": str(output_tokens),
                },
            )
        )
