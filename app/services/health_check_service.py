import asyncio
import logging

from sqlalchemy import text

from app.common.exceptions import (
    DBHealthCheckError,
    QueueHealthCheckError,
    VectorDBHealthCheckError,
)
from app.common.logging import log_decorator, logger
from app.core.config import settings
from app.services.base import BaseService


class HealthCheckService(BaseService):
    """Service for checking infrastructure component availability"""

    @log_decorator(level=logging.DEBUG)
    async def check(self) -> None:
        """Run all infrastructure health checks concurrently

        :returns:
            None
        """
        checks = [
            self.check_db_status(),
            self.check_vector_db_status(),
            self.check_queue_status(),
        ]
        if settings.QDRANT_MAIN_ENABLED:
            checks.append(self.check_vector_db_main_status())
        await asyncio.gather(*checks)

    @log_decorator(level=logging.DEBUG)
    async def check_db_status(self) -> None:
        """Check PostgreSQL connectivity

        :raise:
            DBHealthCheckError: if DB is unreachable or times out
        """
        ms = int(settings.HEALTH_CHECK_TIMEOUT_SEC * 1000)

        async def _exec() -> None:
            async with self.uow_factory as uow:
                await uow.session.execute(text(f"SET LOCAL statement_timeout = {ms}"))
                await uow.session.execute(text("SELECT 1"))

        try:
            await asyncio.wait_for(_exec(), timeout=settings.HEALTH_CHECK_TIMEOUT_SEC)
        except asyncio.TimeoutError as exc:
            logger.critical("Postgres health check timeout", exc_info=exc)
            raise DBHealthCheckError(
                f"Postgres health check timeout after {settings.HEALTH_CHECK_TIMEOUT_SEC}s"
            ) from exc
        except Exception as exc:
            logger.critical("Postgres health check error", exc_info=exc)
            raise DBHealthCheckError("Postgres health check error") from exc

    @log_decorator(level=logging.DEBUG)
    async def check_vector_db_status(self) -> None:
        """Check VectorDB status by checking whether collection exists

        :raise:
            VectorDBHealthCheckError: if VectorDB colletion is unreachable or times out
        """

        async def _exec() -> bool:
            return await self.vector_client.collection_exists(raise_exception=True)

        try:
            await asyncio.wait_for(_exec(), timeout=settings.HEALTH_CHECK_TIMEOUT_SEC)
        except asyncio.TimeoutError as exc:
            logger.critical("Qdrant health check timeout", exc_info=exc)
            raise VectorDBHealthCheckError(
                f"Qdrant health check timeout after {settings.HEALTH_CHECK_TIMEOUT_SEC}s"
            ) from exc
        except Exception as exc:
            logger.critical("Qdrant health check error", exc_info=exc)
            raise VectorDBHealthCheckError("Qdrant health check error") from exc

    @log_decorator(level=logging.DEBUG)
    async def check_queue_status(self) -> None:
        """Check message queue connectivity via PING

        :raise:
            QueueHealthCheckError: if queue is unreachable or times out
        """

        async def _exec() -> None:
            await self.queue_client.check_connection()

        try:
            await asyncio.wait_for(_exec(), timeout=settings.HEALTH_CHECK_TIMEOUT_SEC)
        except asyncio.TimeoutError as exc:
            logger.critical("Queue health check timeout", exc_info=exc)
            raise QueueHealthCheckError(
                f"Queue health check timeout after {settings.HEALTH_CHECK_TIMEOUT_SEC}s"
            ) from exc
        except Exception as exc:
            logger.critical("Queue health check error", exc_info=exc)
            raise QueueHealthCheckError("Queue health check error") from exc

    @log_decorator(level=logging.DEBUG)
    async def check_vector_db_main_status(self) -> None:
        """Check remote (main) VectorDB status

        :raise:
            VectorDBHealthCheckError: if remote VectorDB is unreachable or times out
        """

        async def _exec() -> bool:
            return await self.vector_client_main.collection_exists(raise_exception=True)

        try:
            await asyncio.wait_for(_exec(), timeout=settings.HEALTH_CHECK_TIMEOUT_SEC)
        except asyncio.TimeoutError as exc:
            logger.critical("Qdrant main health check timeout", exc_info=exc)
            raise VectorDBHealthCheckError(
                f"Qdrant main health check timeout after {settings.HEALTH_CHECK_TIMEOUT_SEC}s"
            ) from exc
        except Exception as exc:
            logger.critical("Qdrant main health check error", exc_info=exc)
            raise VectorDBHealthCheckError("Qdrant main health check error") from exc
