from typing import Annotated

import aiohttp
from aiohttp import ClientSession
from fastapi import Depends

from app.adapters.ai_client import AIClientAbstract, MistralClient, GroqClient
from app.adapters.vector_client import VectorClientAbstract, QdrantVectorClient
from app.core.config import settings

aiohttp_session: aiohttp.ClientSession | None = None

ai_client: AIClientAbstract | None = None
groq_client: AIClientAbstract | None = None

vector_client: VectorClientAbstract | None = None


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
    """Get or create the QdrantVectorClient singleton

    :returns:
        vector_client: the QdrantVectorClient instance
    """
    global vector_client
    if not vector_client:
        vector_client = QdrantVectorClient()
    return vector_client
