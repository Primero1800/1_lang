from typing import Annotated

from fastapi import Depends

from app.adapters.queue_client import MessageQueueClientAbstract, RedisClient
from app.adapters.vector_client import VectorClientAbstract, QdrantVectorClient
from app.core.config import settings
from app.repositories.phrase_vector_repository import PhraseVectorRepository

vector_client: VectorClientAbstract | None = None
vector_client_main: VectorClientAbstract | None = None

queue_client: MessageQueueClientAbstract | None = None


async def get_vector_client() -> VectorClientAbstract:
    """Get or create the local QdrantVectorClient singleton (bcp)

    :returns:
        vector_client: the local QdrantVectorClient instance
    """
    global vector_client
    if not vector_client:
        vector_client = QdrantVectorClient()
    return vector_client


async def get_vector_client_main() -> VectorClientAbstract:
    """Get or create the remote QdrantVectorClient singleton (main)

    :returns:
        vector_client_main: the remote QdrantVectorClient instance
    """
    global vector_client_main
    if not vector_client_main:
        vector_client_main = QdrantVectorClient(use_main=True)
    return vector_client_main


async def get_queue_client() -> MessageQueueClientAbstract:
    """Get or create the message queue client singleton (RedisClient)

    :returns:
        queue_client: the shared RedisClient instance
    """
    global queue_client
    if not queue_client:
        queue_client = RedisClient()
    return queue_client


async def get_phrase_vector_repository(
    local_client: Annotated[VectorClientAbstract, Depends(get_vector_client)],
    main_client: Annotated[VectorClientAbstract, Depends(get_vector_client_main)],
) -> PhraseVectorRepository:
    """Get a PhraseVectorRepository wired to the active Qdrant client

    Uses the remote (main) client when QDRANT_MAIN_ENABLED, otherwise local (bcp).

    :param:
        local_client: local Qdrant client (bcp)
        main_client: remote Qdrant client (main)

    :returns:
        repository: PhraseVectorRepository instance
    """
    client = main_client if settings.QDRANT_MAIN_ENABLED else local_client
    return PhraseVectorRepository(client)
