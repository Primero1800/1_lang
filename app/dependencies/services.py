from typing import Annotated, Callable, Type, TypeVar

from fastapi import Depends

from app.adapters.ai_client import AIClientAbstract
from app.adapters.vector_client import VectorClientAbstract
from app.dependencies.infrastructure import get_ai_client, get_vector_client
from app.services.base import BaseDeps, BaseService
from app.services.health_check_service import HealthCheckService
from app.services.test_service import TestService
from app.uow import UnitOfWork, get_uow_factory, get_uow


async def get_base_deps(
    uow_factory: Annotated[UnitOfWork, Depends(get_uow_factory)],
    ai_client: Annotated[AIClientAbstract, Depends(get_ai_client)],
    vector_client: Annotated[VectorClientAbstract, Depends(get_vector_client)],
) -> BaseDeps:
    """Assemble and return base infrastructure dependencies"""
    return BaseDeps(
        uow_factory=uow_factory,
        ai_client=ai_client,
        vector_client=vector_client,
    )


T = TypeVar("T", bound=BaseService)


def _create_service(service_class: Type[T]) -> Callable:
    """Factory: service with UoW session"""

    async def _dependency(
        base_deps: Annotated[BaseDeps, Depends(get_base_deps)],
        uow: Annotated[UnitOfWork, Depends(get_uow)],
    ) -> T:
        return service_class(base_deps=base_deps, uow=uow)

    return _dependency


def _create_service_without_session(service_class: Type[T]) -> Callable:
    """Factory: service without UoW session"""

    async def _dependency(
        base_deps: Annotated[BaseDeps, Depends(get_base_deps)],
    ) -> T:
        return service_class(base_deps=base_deps, uow=None)

    return _dependency


get_health_check_service_without_session = _create_service_without_session(
    HealthCheckService
)
get_test_service_without_session = _create_service_without_session(TestService)
