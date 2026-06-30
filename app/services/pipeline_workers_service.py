import asyncio
import json
import logging
import random
import time
from typing import Any, TypedDict

from app.adapters.queue_client import MessageQueueClientAbstract
from app.commands.base import BaseCommand
from app.commands.w2_command import CommandW2
from app.commands.w3_command import CommandW3
from app.commands.w4_command import CommandW4
from app.commands.w5_command import CommandW5
from app.common.enums import WorkerRoleEnum
from app.common.logging import logger, log_decorator
from app.core.config import settings
from app.dependencies.services import get_base_deps_standalone
from app.services.base import BaseDeps


class _WorkerConfig(TypedDict):
    batch_size: int
    timeout_sec: int
    command_class: type[BaseCommand]


_WORKER_CONFIGS: dict[WorkerRoleEnum, _WorkerConfig] = {
    WorkerRoleEnum.W2: {
        "batch_size": settings.PIPELINE_W2_BATCH_SIZE,
        "timeout_sec": settings.PIPELINE_W2_TIMEOUT_SEC,
        "command_class": CommandW2,
    },
    WorkerRoleEnum.W3: {
        "batch_size": settings.PIPELINE_W3_BATCH_SIZE,
        "timeout_sec": settings.PIPELINE_W3_TIMEOUT_SEC,
        "command_class": CommandW3,
    },
    WorkerRoleEnum.W4: {
        "batch_size": settings.PIPELINE_W4_BATCH_SIZE,
        "timeout_sec": settings.PIPELINE_W4_TIMEOUT_SEC,
        "command_class": CommandW4,
    },
    WorkerRoleEnum.W5: {
        "batch_size": settings.PIPELINE_W5_BATCH_SIZE,
        "timeout_sec": settings.PIPELINE_W5_TIMEOUT_SEC,
        "command_class": CommandW5,
    },
}


class PipelineWorkersService:
    """Manages W2-W5 pipeline worker asyncio tasks subscribed to Redis Pub/Sub dispatch channel"""

    _JITTER_MAX_SEC: float = 1.0

    def __init__(self, queue_client: MessageQueueClientAbstract) -> None:
        """Initialize with a shared queue client (pub/sub objects created per worker task)

        :param:
            queue_client: Redis client used to create per-task pub/sub subscriptions

        :returns:
            None
        """
        self._queue_client = queue_client
        self._tasks: list[asyncio.Task] = []
        self._is_running = False

    @log_decorator(level=logging.DEBUG)
    async def start(self) -> None:
        """Create and start one asyncio task per pipeline worker role

        :returns:
            None
        """
        self._is_running = True
        self._tasks = [
            asyncio.create_task(self._run_worker(role), name=f"pipeline_{role.value}")
            for role in WorkerRoleEnum
        ]
        logger.info("[pipeline_workers] started %d workers", len(self._tasks))

    @log_decorator(level=logging.DEBUG)
    async def stop(self) -> None:
        """Cancel all worker tasks and wait for them to finish

        :returns:
            None
        """
        self._is_running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("[pipeline_workers] all workers stopped")

    @log_decorator(level=logging.DEBUG)
    async def _run_worker(self, role: WorkerRoleEnum) -> None:
        """Subscribe to pub/sub channel and loop: read snapshot → decide → execute

        :param:
            role: worker role determining config and execute target

        :returns:
            None
        """
        jitter = random.uniform(0, self._JITTER_MAX_SEC)
        logger.debug("[%s] jitter %.2fs before subscribe", role.value, jitter)
        await asyncio.sleep(jitter)
        base_deps = await get_base_deps_standalone()
        pubsub = await self._queue_client.subscribe(settings.REDIS_PIPELINE_CHANNEL)
        logger.info(
            "[%s] subscribed to %s", role.value, settings.REDIS_PIPELINE_CHANNEL
        )
        try:
            async for message in pubsub.listen():
                if not self._is_running:
                    break
                if message["type"] != "message":
                    continue
                try:
                    snapshot = json.loads(message["data"])
                except json.JSONDecodeError:
                    logger.warning("[%s] malformed message, skipping", role.value)
                    continue
                if self._should_run(role, snapshot):
                    await self._execute(role, base_deps)
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe()
            await pubsub.aclose()

    def _should_run(self, role: WorkerRoleEnum, snapshot: dict[str, Any]) -> bool:
        """Check ready count and cooldown against worker config

        :param:
            role: worker role
            snapshot: decoded pub/sub message from PipelineSchedulerService

        :returns:
            True if the worker should execute now
        """
        cfg = _WORKER_CONFIGS[role]
        worker_data = snapshot.get(role.value, {})
        ready = worker_data.get("ready", 0)
        last_run = worker_data.get("last_run")
        if ready <= 0:
            return False
        if last_run is None:
            return True
        elapsed = time.time() - last_run
        return ready >= cfg["batch_size"] or elapsed >= cfg["timeout_sec"]

    async def _execute(self, role: WorkerRoleEnum, base_deps: BaseDeps) -> None:
        """Instantiate the role command and execute it

        :param:
            role: worker role
            base_deps: shared deps instance for this worker task

        :returns:
            None
        """
        cfg = _WORKER_CONFIGS[role]
        command: BaseCommand = cfg["command_class"](base_deps, cfg["batch_size"])
        await command.execute()
