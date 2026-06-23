import asyncio
import socket
from collections.abc import AsyncGenerator, Callable
from typing import Any

import sqlalchemy.exc
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.logging import logger
from app.core.database import async_database_session_maker
from app.repositories.ai_token_usage_repository import AiTokenUsageRepository
from app.repositories.phrase_data_repository import PhraseDataRepository
from app.repositories.phrase_embedding_repository import PhraseEmbeddingRepository
from app.repositories.phrase_repository import PhraseRepository
from app.repositories.worker_run_log_repository import WorkerRunLogRepository


class UnitOfWork:
    """Unit of Work for coordinating database transactions"""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        """Initialize the UnitOfWork with a session factory

        :param:
            session_factory: callable that returns a new AsyncSession

        :returns:
            None
        """
        self.session_factory = session_factory

    async def __aenter__(self) -> Any:
        """Open a session and initialise repositories

        :returns:
            self: the UnitOfWork instance with active session and repositories
        """
        self.session = self.session_factory()
        self.phrase_repository = PhraseRepository(self.session)
        self.phrase_data_repository = PhraseDataRepository(self.session)
        self.phrase_embedding_repository = PhraseEmbeddingRepository(self.session)
        self.ai_token_usage_repository = AiTokenUsageRepository(self.session)
        self.worker_run_log_repository = WorkerRunLogRepository(self.session)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Commit or rollback the transaction and close the session

        :param:
            exc_type: exception class if an error occurred, else None
            exc_val: exception instance if an error occurred, else None
            exc_tb: traceback object if an error occurred, else None

        :returns:
            None
        """
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
        """Log a commit failure at the appropriate level based on error type

        :param:
            e: the exception raised during commit

        :returns:
            None
        """
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
        """Commit the current transaction

        :returns:
            None
        """
        await self.session.commit()

    async def rollback(self) -> None:
        """Roll back the current transaction

        :returns:
            None
        """
        await self.session.rollback()


async def get_uow() -> AsyncGenerator[UnitOfWork, None]:
    """FastAPI dependency for a request-scoped UnitOfWork

    :returns:
        uow: context-managed UnitOfWork; commits on clean exit, rolls back on error
    """
    async with UnitOfWork(session_factory=async_database_session_maker) as uow:
        yield uow


async def get_uow_factory() -> UnitOfWork:
    """FastAPI dependency for a UnitOfWork factory (not context-managed)

    :returns:
        uow: UnitOfWork instance; the caller is responsible for entering and exiting the context
    """
    return UnitOfWork(session_factory=async_database_session_maker)
