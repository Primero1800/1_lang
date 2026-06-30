import logging
from typing import Any

from app.common.enums import WorkerStatusEnum
from app.common.logging import log_decorator
from app.uow import UnitOfWork


class WorkerRunLogService:
    """Service for recording worker batch execution lifecycle to the DB"""

    def __init__(self, uow: UnitOfWork) -> None:
        """Initialize with a UnitOfWork instance

        :param:
            uow: unit of work providing the worker run log repository

        :returns:
            None
        """
        self._uow = uow

    @log_decorator(level=logging.DEBUG)
    async def start(self, worker: str, batch_size: int | None = None) -> int:
        """Create a RUNNING log entry for a worker batch

        :param:
            worker: worker name (e.g. 'token_worker', 'w2_generate')
            batch_size: number of items taken into processing, if known at start

        :returns:
            log_id: primary key of the created entry, pass to finish()
        """
        async with self._uow as uow:
            return await uow.worker_run_log_repository.create(
                worker=worker, batch_size=batch_size
            )

    @log_decorator(level=logging.DEBUG)
    async def finish(
        self,
        log_id: int,
        status: WorkerStatusEnum,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Mark a log entry as finished with terminal status and optional result

        :param:
            log_id: primary key returned by start()
            status: WorkerStatusEnum.DONE or WorkerStatusEnum.FAILED
            result: arbitrary result dict (counters, error messages, etc.)

        :returns:
            None
        """
        async with self._uow as uow:
            await uow.worker_run_log_repository.finish(
                log_id=log_id, status=status, result=result
            )

    @log_decorator(level=logging.DEBUG)
    async def abandon_running(self, worker: str) -> int:
        """Mark all stale RUNNING entries for a worker as FAILED

        :param:
            worker: worker name to clean up on restart

        :returns:
            count: number of rows updated
        """
        async with self._uow as uow:
            return await uow.worker_run_log_repository.abandon_running(worker=worker)
