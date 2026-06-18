from typing import Annotated

import aiohttp
from aiohttp import ClientSession
from fastapi import Depends

from app.adapters.ai_client import AIClientAbstract, MistralClient, GroqClient
from app.adapters.vector_client import VectorClientAbstract, QdrantVectorClient
from app.core.config import settings
from app.repositories.phrase_loading_repository import PhraseLoadingRepository

aiohttp_session: aiohttp.ClientSession | None = None

ai_client: AIClientAbstract | None = None
groq_client: AIClientAbstract | None = None

vector_client: VectorClientAbstract | None = None
vector_client_main: VectorClientAbstract | None = None


async def get_aiohttp_session() -> ClientSession:
    """Get or create the shared aiohttp ClientSession singleton"""
    global aiohttp_session
    if not aiohttp_session:
        aiohttp_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                keepalive_timeout=settings.AIOHTTP_KEEPALIVE_TIMEOUT,
                enable_cleanup_closed=True,
                force_close=False,
            ),
            timeout=aiohttp.ClientTimeout(total=settings.AIOHTTP_TIMEOUT_SECONDS),
        )
    return aiohttp_session


async def get_ai_client(
    session: Annotated[ClientSession, Depends(get_aiohttp_session)],
) -> AIClientAbstract:
    """Get or create the MistralClient singleton

    :param:
        session: the shared aiohttp ClientSession

    :returns:
        ai_client: the MistralClient instance
    """
    global ai_client
    if not ai_client:
        ai_client = MistralClient(session)
    return ai_client


async def get_groq_client(
    session: Annotated[ClientSession, Depends(get_aiohttp_session)],
) -> AIClientAbstract:
    """Get or create the GroqClient singleton

    :param:
        session: the shared aiohttp ClientSession

    :returns:
        groq_client: the GroqClient instance
    """
    global groq_client
    if not groq_client:
        groq_client = GroqClient(session)
    return groq_client


async def get_vector_client() -> VectorClientAbstract:
    """Get or create the local QdrantVectorClient singleton (bcp)

    :returns:
        vector_client: the local QdrantVectorClient instance
    """
    global vector_client
    if not vector_client:
        vector_client = QdrantVectorClient()
    return vector_client


async def get_phrase_loading_repository() -> PhraseLoadingRepository:
    """Get a PhraseLoadingRepository wired to the active Qdrant client

    Uses the remote (main) client when QDRANT_MAIN_ENABLED, otherwise local (bcp).

    :returns:
        repository: PhraseLoadingRepository instance
    """
    if settings.QDRANT_MAIN_ENABLED:
        client = await get_vector_client_main()
    else:
        client = await get_vector_client()
    return PhraseLoadingRepository(client)


async def get_vector_client_main() -> VectorClientAbstract:
    """Get or create the remote QdrantVectorClient singleton (main)

    :returns:
        vector_client_main: the remote QdrantVectorClient instance
    """
    global vector_client_main
    if not vector_client_main:
        vector_client_main = QdrantVectorClient(use_main=True)
    return vector_client_main
