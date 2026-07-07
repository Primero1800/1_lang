import logging
from datetime import date, datetime
from typing import Any

from sqlalchemy import ColumnElement, func, select, update

from app.core.config import settings

from app.common.enums import WorkerStatusEnum
from app.common.logging import log_decorator
from app.models.worker_run_log import WorkerRunLog
from app.repositories.base_repository import BaseRepository
from app.repositories.repository_error_handler import repository_error_handler


@repository_error_handler
class WorkerRunLogRepository(BaseRepository):
    """Repository for logging worker batch execution lifecycle"""

    @log_decorator(level=logging.DEBUG)
    async def create(self, worker: str, batch_size: int | None = None) -> int:
        """Insert a new RUNNING log entry and return its id

        :param:
            worker: worker name (e.g. 'token_worker', 'w2_generate')
            batch_size: number of items taken into processing, if known at start

        :returns:
            id: primary key of the created row
        """
        entry = WorkerRunLog(worker=worker, batch_size=batch_size)
        self._session.add(entry)
        await self._session.flush()
        return entry.id

    @log_decorator(level=logging.DEBUG)
    async def finish(
        self,
        log_id: int,
        status: WorkerStatusEnum,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Mark a log entry as finished with final status and optional result payload

        :param:
            log_id: primary key returned by create()
            status: terminal status (DONE or FAILED)
            result: arbitrary result dict (counters, error info, etc.)

        :returns:
            None
        """
        stmt = (
            update(WorkerRunLog)
            .where(WorkerRunLog.id == log_id)
            .values(
                status=status,
                finished_at=datetime.now(tz=settings.default_timezone),
                result=result,
            )
        )
        await self._session.execute(stmt)

    @log_decorator(level=logging.DEBUG)
    async def abandon_running(self, worker: str) -> int:
        """Mark all RUNNING entries for a worker as FAILED (called on worker startup)

        :param:
            worker: worker name to clean up

        :returns:
            count: number of rows updated
        """
        stmt = (
            update(WorkerRunLog)
            .where(
                WorkerRunLog.worker == worker,
                WorkerRunLog.status == WorkerStatusEnum.RUNNING,
            )
            .values(
                status=WorkerStatusEnum.FAILED,
                finished_at=datetime.now(tz=settings.default_timezone),
                result={"error": "abandoned on worker restart"},
            )
        )
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore[attr-defined]

    @log_decorator(level=logging.DEBUG)
    async def get_last_runs(self, workers: list[str]) -> dict[str, datetime | None]:
        """Return the last successful finished_at timestamp per worker

        :param:
            workers: list of worker name strings to query

        :returns:
            last_runs: {worker_name: last finished_at datetime or None if never run}
        """
        stmt = (
            select(WorkerRunLog.worker, WorkerRunLog.finished_at)
            .where(
                WorkerRunLog.worker.in_(workers),
                WorkerRunLog.status == WorkerStatusEnum.DONE,
            )
            .order_by(WorkerRunLog.worker, WorkerRunLog.finished_at.desc())
            .distinct(WorkerRunLog.worker)
        )
        result = await self._session.execute(stmt)
        found = {row.worker: row.finished_at for row in result}
        return {w: found.get(w) for w in workers}

    @log_decorator(level=logging.DEBUG)
    async def list_runs(
        self,
        worker: str | None,
        status: WorkerStatusEnum | None,
        started_from: date | None,
        started_to: date | None,
        per_page: int,
        page: int,
    ) -> tuple[list[WorkerRunLog], int]:
        """Return paginated worker run log rows matching filters and total count

        :param:
            worker: prefix filter on worker name (case-insensitive)
            status: exact status filter
            started_from: lower bound on created_at (date, inclusive)
            started_to: upper bound on created_at (date, inclusive)
            per_page: page size
            page: 1-based page number

        :returns:
            rows: matching WorkerRunLog ORM instances for the requested page
            total_count: total number of matching rows across all pages
        """
        conditions: list[ColumnElement[bool]] = []
        if worker is not None:
            conditions.append(WorkerRunLog.worker.ilike(f"{worker}%"))
        if status is not None:
            conditions.append(WorkerRunLog.status == status)
        if started_from is not None:
            conditions.append(WorkerRunLog.created_at >= started_from)
        if started_to is not None:
            conditions.append(WorkerRunLog.created_at <= started_to)

        total_count: int = (
            await self._session.execute(
                select(func.count()).select_from(WorkerRunLog).where(*conditions)
            )
        ).scalar_one()
        offset = (page - 1) * per_page
        rows = (
            (
                await self._session.execute(
                    select(WorkerRunLog)
                    .where(*conditions)
                    .order_by(WorkerRunLog.created_at.desc())
                    .limit(per_page)
                    .offset(offset)
                )
            )
            .scalars()
            .all()
        )
        return list(rows), total_count
