import asyncio
from datetime import date
from typing import Any

from app.adapters.queue_client import MessageQueueClientAbstract
from app.common.enums import WorkerStatusEnum
from app.common.logging import logger
from app.core.config import settings
from app.services.worker_run_log_service import WorkerRunLogService
from app.uow import get_uow_factory

_WORKER_NAME = settings.REDIS_TOKENS_WORKER


class TokenWorkerService:
    """Background service that drains the token usage stream and persists to DB"""

    def __init__(self, queue_client: MessageQueueClientAbstract) -> None:
        """Initialise with a started message queue client

        :param:
            queue_client: message queue client for stream operations

        :returns:
            None
        """
        self._queue_client = queue_client
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Create consumer group, clean up stale logs, and launch the background polling task

        :returns:
            None
        """
        uow = await get_uow_factory()
        abandoned = await WorkerRunLogService(uow=uow).abandon_running(_WORKER_NAME)
        if abandoned:
            logger.warning("[token_worker] marked %d stale log(s) as FAILED", abandoned)

        await self._queue_client.xgroup_create(
            settings.REDIS_TOKENS_STREAM, settings.REDIS_TOKENS_GROUP
        )
        self._task = asyncio.create_task(self._run())
        logger.info("[token_worker] started")

    async def stop(self) -> None:
        """Cancel the polling task and wait for it to finish

        :returns:
            None
        """
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[token_worker] stopped")

    async def _run(self) -> None:
        """Polling loop: reclaim pending messages on startup then continuously drain new ones

        :returns:
            None
        """
        # Reclaim messages delivered but not acked before last restart
        await self._process(cursor="0")

        while True:
            try:
                had_messages = await self._process(cursor=">")
                if not had_messages:
                    await asyncio.sleep(settings.REDIS_TOKENS_POLL_INTERVAL)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[token_worker] unexpected error: %s", exc, exc_info=exc)
                await asyncio.sleep(settings.REDIS_TOKENS_POLL_INTERVAL)

    async def _process(self, cursor: str) -> bool:
        """Read one batch from the stream, aggregate token counts, persist, and ack

        :param:
            cursor: '>' for new messages, '0' to reclaim pending unacked messages

        :returns:
            had_messages: True if at least one message was processed, False if stream was empty
        """
        # 1. Read next batch from the stream
        messages = await self._queue_client.xreadgroup(
            settings.REDIS_TOKENS_GROUP,
            settings.REDIS_TOKENS_WORKER,
            settings.REDIS_TOKENS_STREAM,
            count=settings.REDIS_TOKENS_BATCH_SIZE,
            cursor=cursor,
        )
        if not messages:
            return False

        # 2. Aggregate token counts per (model, operation, name) key
        aggregated: dict[tuple[str, ...], dict[str, Any]] = {}
        msg_ids: list[str] = []

        for _stream, entries in messages:
            for msg_id, fields in entries:
                msg_ids.append(msg_id)
                key = (
                    fields["model"],
                    fields["operation"],
                    fields.get("name", "system"),
                )
                if key not in aggregated:
                    aggregated[key] = {
                        "model": key[0],
                        "operation": key[1],
                        "name": key[2],
                        "date": date.today(),
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                aggregated[key]["input_tokens"] += int(fields["input_tokens"])
                aggregated[key]["output_tokens"] += int(fields["output_tokens"])

        if not msg_ids:
            return False

        # 3. Persist the batch and acknowledge messages in one transaction
        uow = await get_uow_factory()
        try:
            async with uow:
                log_id = await uow.worker_run_log_repository.create(
                    _WORKER_NAME, batch_size=len(msg_ids)
                )
                await uow.ai_token_usage_repository.bulk_accumulate(
                    list(aggregated.values())
                )
                await uow.worker_run_log_repository.finish(
                    log_id,
                    WorkerStatusEnum.DONE,
                    result={"messages": len(msg_ids), "rows_upserted": len(aggregated)},
                )
                await self._queue_client.xack(
                    settings.REDIS_TOKENS_STREAM, settings.REDIS_TOKENS_GROUP, *msg_ids
                )
        except Exception as exc:
            logger.error(
                "[token_worker] batch failed, messages stay pending: %s",
                exc,
                exc_info=exc,
            )
            raise

        logger.debug(
            "[token_worker] processed %d message(s), %d row(s) upserted",
            len(msg_ids),
            len(aggregated),
        )
        return True
