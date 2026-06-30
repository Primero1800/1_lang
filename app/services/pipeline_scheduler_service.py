import json
import time
from datetime import datetime
from typing import Any, TypedDict

from app.common.enums import PhraseStatusEnum
from app.common.logging import logger
from app.core.config import settings
from app.services.base import BaseService
from app.uow import UnitOfWork


class _WorkerConfig(TypedDict):
    base: list[PhraseStatusEnum]
    in_progress: PhraseStatusEnum


_WORKER_STATUSES: dict[str, _WorkerConfig] = {
    "w2": {
        "base": [PhraseStatusEnum.DRAFT, PhraseStatusEnum.GENERATING_FAILED],
        "in_progress": PhraseStatusEnum.GENERATING_IN_PROGRESS,
    },
    "w3": {
        "base": [PhraseStatusEnum.GENERATING_DONE, PhraseStatusEnum.TRANSLATING_FAILED],
        "in_progress": PhraseStatusEnum.TRANSLATING_IN_PROGRESS,
    },
    "w4": {
        "base": [PhraseStatusEnum.TRANSLATING_DONE, PhraseStatusEnum.EMBEDDING_FAILED],
        "in_progress": PhraseStatusEnum.EMBEDDING_IN_PROGRESS,
    },
    "w5": {
        "base": [PhraseStatusEnum.EMBEDDING_DONE, PhraseStatusEnum.LOADING_FAILED],
        "in_progress": PhraseStatusEnum.LOADING_IN_PROGRESS,
    },
}

_WORKERS = list(_WORKER_STATUSES.keys())


class PipelineSchedulerService(BaseService):
    """ETL scheduler: aggregates phrase pipeline statuses and publishes dispatch snapshot to Redis Pub/Sub"""

    async def _fetch_status_counts(self, uow: UnitOfWork) -> dict[str, int]:
        """Extract: phrase counts per status from DB

        :param:
            uow: active UnitOfWork with phrase_repository

        :returns:
            counts: {status_value: ready_count}
        """
        return await uow.phrase_repository.get_pipeline_status_counts(
            stuck_threshold_sec=settings.STUCK_THRESHOLD * 60
        )

    async def _fetch_last_runs(self, uow: UnitOfWork) -> dict[str, datetime | None]:
        """Extract: last successful run timestamp per pipeline worker

        :param:
            uow: active UnitOfWork with worker_run_log_repository

        :returns:
            last_runs: {worker_name: last finished_at or None}
        """
        return await uow.worker_run_log_repository.get_last_runs(_WORKERS)

    async def _build_snapshot(
        self,
        status_counts: dict[str, int],
        last_runs: dict[str, datetime | None],
    ) -> dict[str, dict[str, Any]]:
        """Transform: merge status counts and last run times into per-worker dispatch snapshot

        :param:
            status_counts: {status_value: ready_count} from _fetch_status_counts
            last_runs: {worker_name: last finished_at or None} from _fetch_last_runs

        :returns:
            snapshot: {worker_name: {"ready": N, "last_run": timestamp_or_null, "ts": now}}
        """
        ts = int(time.time())
        snapshot = {}
        for worker, cfg in _WORKER_STATUSES.items():
            ready = sum(status_counts.get(s.value, 0) for s in cfg["base"])
            ready += status_counts.get(cfg["in_progress"].value, 0)
            last_run = last_runs.get(worker)
            snapshot[worker] = {
                "ready": ready,
                "last_run": int(last_run.timestamp()) if last_run else None,
                "ts": ts,
            }
        return snapshot

    async def _publish_snapshot(self, snapshot: dict[str, dict[str, Any]]) -> None:
        """Load: publish snapshot to Redis Pub/Sub channel

        :param:
            snapshot: per-worker dict from _build_snapshot

        :returns:
            None
        """
        await self.queue_client.publish(
            settings.REDIS_PIPELINE_CHANNEL, json.dumps(snapshot)
        )
        logger.info("[pipeline_scheduler] published snapshot: %s", snapshot)

    async def run(self) -> None:
        """Orchestrate one ETL tick: extract → transform → publish

        Called by APScheduler on configured interval.

        :returns:
            None
        """
        # 1. Extract: phrase status counts and last worker run timestamps
        async with self.uow_factory as uow:
            status_counts = await self._fetch_status_counts(uow)
            last_runs = await self._fetch_last_runs(uow)
        # 2. Transform: build per-worker dispatch snapshot
        snapshot = await self._build_snapshot(status_counts, last_runs)
        # 3. Load: publish to Redis Pub/Sub
        await self._publish_snapshot(snapshot)
