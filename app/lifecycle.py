import aiohttp
from fastapi import FastAPI

from app.adapters.vector_client import VectorClientAbstract
from app.common.logging import logger
from app.core.database import initialize_db
from app.dependencies.infrastructure import get_aiohttp_session, get_vector_client


class AppLifecycle:
    """Manages initialization and cleanup of application components"""

    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.aiohttp_session: aiohttp.ClientSession | None = None
        self.vector_client: VectorClientAbstract | None = None

    async def on_startup(self) -> None:
        """Run all startup procedures"""
        await self._initialize_core()
        self.aiohttp_session = await get_aiohttp_session()
        self.vector_client = await get_vector_client()
        await self.vector_client.start()

    async def on_shutdown(self) -> None:
        """Gracefully stop all services and close connections"""
        if self.aiohttp_session:
            await self.aiohttp_session.close()
        if self.vector_client:
            await self.vector_client.stop()
        logger.info("Shutting down the APP")

    async def _initialize_core(self) -> None:
        """Initialize core DB components"""
        try:
            await initialize_db()
            logger.info("Core application components initialized.")
        except Exception as e:
            logger.critical(f"FATAL: Core initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Core application initialization failed: {e}") from e
