import abc
import logging

from app.common.enums import WorkerRoleEnum, WorkerStatusEnum
from app.common.logging import log_decorator, logger
from app.services.base import BaseDeps


class BaseCommand(abc.ABC):
    """Abstract base for pipeline worker dispatch commands"""

    _ROLE: WorkerRoleEnum

    def __init__(self, base_deps: BaseDeps, batch_size: int) -> None:
        """Initialize with shared deps and batch size for this execution

        :param:
            base_deps: shared infrastructure dependencies
            batch_size: number of items to process in this run

        :returns:
            None
        """
        self._base_deps = base_deps
        self._batch_size = batch_size

    @abc.abstractmethod
    async def _do_execute(self) -> dict:
        """Run the pipeline step and return the result

        :returns:
            result: service response dict (processed, inserted, skipped, etc.)
        """

    @log_decorator(level=logging.INFO)
    async def execute(self) -> dict:
        """Orchestrate: abandon stuck runs → open log → execute → close log

        :returns:
            result: service response dict with counts or error info
        """
        async with self._base_deps.uow_factory as uow:
            await uow.worker_run_log_repository.abandon_running(self._ROLE.value)
            log_id = await uow.worker_run_log_repository.create(
                self._ROLE.value, batch_size=self._batch_size
            )

        try:
            result = await self._do_execute()
        except Exception as exc:
            logger.error("[%s] execute failed: %s", self._ROLE.value, exc, exc_info=exc)
            result = {"error": str(exc)}
            async with self._base_deps.uow_factory as uow:
                await uow.worker_run_log_repository.finish(
                    log_id, WorkerStatusEnum.FAILED, result=result
                )
            return result

        async with self._base_deps.uow_factory as uow:
            await uow.worker_run_log_repository.finish(
                log_id, WorkerStatusEnum.DONE, result=result
            )
        return result
