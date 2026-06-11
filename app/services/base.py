from dataclasses import dataclass
from typing import Any, Protocol

from app.adapters.ai_client import AIClientAbstract
from app.adapters.vector_client import VectorClientAbstract
from app.uow import UnitOfWork


@dataclass
class BaseDeps:
    """Infrastructure dependencies shared across all services"""

    uow_factory: UnitOfWork
    ai_client: AIClientAbstract
    ai_client2: AIClientAbstract
    vector_client: VectorClientAbstract


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
        self.uow_factory = base_deps.uow_factory
        self.uow = uow
        self.ai_client = base_deps.ai_client
        self.ai_client2 = base_deps.ai_client2
        self.vector_client = base_deps.vector_client

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def check(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def ask(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
