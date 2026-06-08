import asyncio
import logging

from sqlalchemy import text

from app.common.exceptions import DBHealthCheckError
from app.common.logging import log_decorator, logger
from app.core.config import settings
from app.services.base import BaseService


class HealthCheckService(BaseService):
    """Service for checking infrastructure component availability"""

    @log_decorator(level=logging.DEBUG)
    async def check(self) -> None:
        """Run all infrastructure health checks concurrently"""
        await asyncio.gather(
            self.check_db_status(),
        )

    @log_decorator(level=logging.DEBUG)
    async def check_db_status(self) -> None:
        """Check PostgreSQL connectivity

        :raise:
            DBHealthCheckError: if DB is unreachable or times out
        """
        ms = int(settings.HEALTH_CHECK_TIMEOUT_SEC * 1000)
        try:

            async def _exec() -> None:
                async with self.uow_factory as uow:
                    await uow.session.execute(
                        text(f"SET LOCAL statement_timeout = {ms}")
                    )
                    await uow.session.execute(text("SELECT 1"))

            await asyncio.wait_for(_exec(), timeout=settings.HEALTH_CHECK_TIMEOUT_SEC)
        except asyncio.TimeoutError as exc:
            logger.critical("Postgres health check timeout", exc_info=exc)
            raise DBHealthCheckError(
                f"Postgres health check timeout after {settings.HEALTH_CHECK_TIMEOUT_SEC}s"
            ) from exc
        except Exception as exc:
            logger.critical("Postgres health check error", exc_info=exc)
            raise DBHealthCheckError("Postgres health check error") from exc
