from typing import Annotated, Callable, Type, TypeVar

from fastapi import Depends

from app.adapters.queue_client import MessageQueueClientAbstract
from app.adapters.vector_client import VectorClientAbstract
from app.dependencies.infrastructure import (
    get_phrase_vector_repository,
    get_queue_client,
    get_vector_client,
    get_vector_client_main,
)
from app.repositories.phrase_vector_repository import PhraseVectorRepository
from app.services.base import BaseDeps, BaseService
from app.services.health_check_service import HealthCheckService
from app.services.phrase_data_service import PhraseDataService
from app.services.phrase_embedding_service import PhraseEmbeddingService
from app.services.phrase_loading_service import PhraseLoadingService
from app.services.phrase_service import PhraseService
from app.services.phrase_translation_service import PhraseTranslationService
from app.services.phrase_search_service import PhraseSearchService
from app.services.token_usage_service import TokenUsageService
from app.uow import UnitOfWork, get_uow_factory, get_uow


async def get_base_deps(
    uow_factory: Annotated[UnitOfWork, Depends(get_uow_factory)],
    vector_client: Annotated[VectorClientAbstract, Depends(get_vector_client)],
    vector_client_main: Annotated[
        VectorClientAbstract, Depends(get_vector_client_main)
    ],
    queue_client: Annotated[MessageQueueClientAbstract, Depends(get_queue_client)],
) -> BaseDeps:
    """Assemble and return the shared infrastructure dependency container

    :param:
        uow_factory: unit-of-work factory (not context-managed)
        vector_client: local Qdrant client (bcp)
        vector_client_main: remote Qdrant client (main)
        queue_client: async Redis client

    :returns:
        base_deps: populated BaseDeps dataclass instance
    """
    return BaseDeps(
        uow_factory=uow_factory,
        vector_client=vector_client,
        vector_client_main=vector_client_main,
        queue_client=queue_client,
    )


async def get_base_deps_standalone() -> BaseDeps:
    """Assemble BaseDeps outside of FastAPI request context (scheduler, background tasks)

    :returns:
        base_deps: populated BaseDeps dataclass instance
    """
    return BaseDeps(
        uow_factory=await get_uow_factory(),
        vector_client=await get_vector_client(),
        vector_client_main=await get_vector_client_main(),
        queue_client=await get_queue_client(),
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


async def get_test_service_without_session(
    base_deps: Annotated[BaseDeps, Depends(get_base_deps)],
    vector_repository: Annotated[
        PhraseVectorRepository, Depends(get_phrase_vector_repository)
    ],
) -> PhraseSearchService:
    """FastAPI dependency for PhraseSearchService with injected Qdrant vector repository

    :param:
        base_deps: shared infrastructure dependencies
        vector_repository: Qdrant-backed repository for phrase search operations

    :returns:
        service: PhraseSearchService instance ready for T1 search pipeline
    """
    return PhraseSearchService(base_deps=base_deps, vector_repository=vector_repository)


get_token_usage_service = _create_service(TokenUsageService)

get_phrase_service = _create_service(PhraseService)
get_phrase_service_without_session = _create_service_without_session(PhraseService)

get_phrase_data_service_without_session = _create_service_without_session(
    PhraseDataService
)

get_phrase_translation_service_without_session = _create_service_without_session(
    PhraseTranslationService
)

get_phrase_embedding_service_without_session = _create_service_without_session(
    PhraseEmbeddingService
)


async def get_phrase_loading_service_without_session(
    base_deps: Annotated[BaseDeps, Depends(get_base_deps)],
    loading_repository: Annotated[
        PhraseVectorRepository, Depends(get_phrase_vector_repository)
    ],
) -> PhraseLoadingService:
    """FastAPI dependency for PhraseLoadingService with injected Qdrant repository

    :param:
        base_deps: shared infrastructure dependencies
        loading_repository: Qdrant-backed repository for phrase upsert operations

    :returns:
        service: PhraseLoadingService instance ready for W5 processing
    """
    return PhraseLoadingService(
        base_deps=base_deps,
        loading_repository=loading_repository,
    )
