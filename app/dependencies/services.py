from typing import Annotated, Callable, Type, TypeVar

from fastapi import Depends

from app.adapters.ai_client import AIClientAbstract
from app.adapters.vector_client import VectorClientAbstract
from app.dependencies.infrastructure import (
    get_ai_client,
    get_groq_client,
    get_vector_client,
)
from app.services.base import BaseDeps, BaseService
from app.services.health_check_service import HealthCheckService
from app.services.phrase_data_service import PhraseDataService
from app.services.phrase_service import PhraseService
from app.services.test_service import TestService
from app.uow import UnitOfWork, get_uow_factory, get_uow


async def get_base_deps(
    uow_factory: Annotated[UnitOfWork, Depends(get_uow_factory)],
    ai_client: Annotated[AIClientAbstract, Depends(get_ai_client)],
    ai_client2: Annotated[AIClientAbstract, Depends(get_groq_client)],
    vector_client: Annotated[VectorClientAbstract, Depends(get_vector_client)],
) -> BaseDeps:
    """Assemble and return the shared infrastructure dependency container

    :param:
        uow_factory: unit-of-work factory (not context-managed)
        ai_client: primary AI client (Mistral)
        ai_client2: secondary AI client (Groq)
        vector_client: Qdrant vector database client

    :returns:
        base_deps: populated BaseDeps dataclass instance
    """
    return BaseDeps(
        uow_factory=uow_factory,
        ai_client=ai_client,
        ai_client2=ai_client2,
        vector_client=vector_client,
    )


T = TypeVar("T", bound=BaseService)


def _create_service(service_class: Type[T]) -> Callable:
    """Return a FastAPI dependency that instantiates a service with a request-scoped UoW session

    :param:
        service_class: the concrete BaseService subclass to instantiate

    :returns:
        dependency: async FastAPI dependency function
    """

    async def _dependency(
        base_deps: Annotated[BaseDeps, Depends(get_base_deps)],
        uow: Annotated[UnitOfWork, Depends(get_uow)],
    ) -> T:
        return service_class(base_deps=base_deps, uow=uow)

    return _dependency


def _create_service_without_session(service_class: Type[T]) -> Callable:
    """Return a FastAPI dependency that instantiates a service without a session (uses uow_factory)

    :param:
        service_class: the concrete BaseService subclass to instantiate

    :returns:
        dependency: async FastAPI dependency function
    """

    async def _dependency(
        base_deps: Annotated[BaseDeps, Depends(get_base_deps)],
    ) -> T:
        return service_class(base_deps=base_deps, uow=None)

    return _dependency


get_health_check_service_without_session = _create_service_without_session(
    HealthCheckService
)
get_test_service_without_session = _create_service_without_session(TestService)

get_phrase_service = _create_service(PhraseService)
get_phrase_service_without_session = _create_service_without_session(PhraseService)

get_phrase_data_service_without_session = _create_service_without_session(
    PhraseDataService
)
