import asyncio
import socket
from collections.abc import AsyncGenerator, Callable
from typing import Any

import sqlalchemy.exc
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.logging import logger
from app.core.database import async_database_session_maker


class UnitOfWork:
    """Unit of Work for coordinating database transactions"""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self.session_factory = session_factory

    async def __aenter__(self) -> Any:
        self.session = self.session_factory()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is not None:
                await self.rollback()
                if issubclass(
                    exc_type,
                    (
                        sqlalchemy.exc.OperationalError,
                        sqlalchemy.exc.TimeoutError,
                        ConnectionRefusedError,
                        asyncio.TimeoutError,
                        socket.gaierror,
                    ),
                ):
                    logger.error(f"DB ERROR (Connection/Pool): {exc_val}")
                elif issubclass(
                    exc_type,
                    (
                        sqlalchemy.exc.ProgrammingError,
                        sqlalchemy.exc.DBAPIError,
                    ),
                ) and not issubclass(exc_type, sqlalchemy.exc.IntegrityError):
                    logger.error(f"SQL SYNTAX/CODE ERROR: {exc_val}")
                elif issubclass(exc_type, sqlalchemy.exc.IntegrityError):
                    logger.warning(f"DB Integrity Error: {exc_val}")
            else:
                try:
                    await self.commit()
                except Exception as e:
                    await self.rollback()
                    await self._handle_commit_error(e)
                    raise e
        finally:
            await self.session.close()

    async def _handle_commit_error(self, e: Exception) -> None:
        if isinstance(
            e,
            (
                sqlalchemy.exc.OperationalError,
                sqlalchemy.exc.TimeoutError,
                asyncio.TimeoutError,
                socket.gaierror,
            ),
        ):
            logger.error(f"DB COMMIT FAILED (Connection/Pool): {e}")
        elif isinstance(e, sqlalchemy.exc.IntegrityError):
            logger.warning(f"DB COMMIT FAILED (Integrity): {e}")
        else:
            logger.error(f"DB COMMIT FAILED (Unknown): {e}")

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()


async def get_uow() -> AsyncGenerator[UnitOfWork, None]:
    """FastAPI dependency for a request-scoped UnitOfWork"""
    async with UnitOfWork(session_factory=async_database_session_maker) as uow:
        yield uow


async def get_uow_factory() -> UnitOfWork:
    """FastAPI dependency for a UnitOfWork factory (not context-managed)"""
    return UnitOfWork(session_factory=async_database_session_maker)
