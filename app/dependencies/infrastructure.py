from typing import Annotated

import aiohttp
from aiohttp import ClientSession
from fastapi import Depends

from app.adapters.ai_client import AIClientAbstract, MistralClient
from app.core.config import settings

aiohttp_session: aiohttp.ClientSession | None = None

ai_client: AIClientAbstract | None = None


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
    global ai_client
    if not ai_client:
        ai_client = MistralClient(session)
    return ai_client
