from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

from app.adapters.queue_client import MessageQueueClientAbstract
from app.adapters.vector_client import VectorClientAbstract
from app.common.logging import logger
from app.core.config import settings
from app.core.database import initialize_db, shutdown_db
from app.dependencies.infrastructure import (
    get_queue_client,
    get_vector_client,
    get_vector_client_main,
)
from app.services.pipeline_workers_service import PipelineWorkersService
from app.services.token_worker_service import TokenWorkerService
from app.utils.pipeline_scheduler_func import run_pipeline_scheduler


class AppLifecycle:
    """Manages initialization and cleanup of application components"""

    def __init__(self, app: FastAPI) -> None:
        """Initialize the lifecycle manager

        :param:
            app: the FastAPI application instance

        :returns:
            None
        """
        self.app = app
        self.vector_client: VectorClientAbstract | None = None
        self.vector_client_main: VectorClientAbstract | None = None
        self.queue_client: MessageQueueClientAbstract | None = None
        self._token_worker: TokenWorkerService | None = None
        self._pipeline_workers: PipelineWorkersService | None = None
        self._scheduler: AsyncIOScheduler | None = None

    async def on_startup(self) -> None:
        """Run all startup procedures

        :returns:
            None
        """
        # 1. Initialize database connection pool
        await self._initialize_core()
        # 2. Create vector database client (local bcp)
        self.vector_client = await get_vector_client()
        # 3. Start vector client and ensure collection exists
        await self.vector_client.start()
        # 4. Start remote (main) vector client if enabled
        if settings.QDRANT_MAIN_ENABLED:
            self.vector_client_main = await get_vector_client_main()
            await self.vector_client_main.start()
        # 5. Start Message queue client
        self.queue_client = await get_queue_client()
        await self.queue_client.start()
        # 6. Start token usage background worker
        self._token_worker = TokenWorkerService(self.queue_client)
        await self._token_worker.start()
        # 7. Start pipeline worker tasks (W2-W5)
        self._pipeline_workers = PipelineWorkersService(self.queue_client)
        await self._pipeline_workers.start()
        # 8. Start pipeline dispatch scheduler
        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            func=run_pipeline_scheduler,
            trigger=CronTrigger(minute=f"*/{settings.pipeline_cron_minutes}"),
            id="pipeline_scheduler",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "[pipeline_scheduler] APScheduler started, interval=%d min",
            settings.pipeline_cron_minutes,
        )

    async def on_shutdown(self) -> None:
        """Gracefully stop all services and close connections

        :returns:
            None
        """
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
        if self._pipeline_workers:
            await self._pipeline_workers.stop()
        if self._token_worker:
            await self._token_worker.stop()
        if self.vector_client:
            await self.vector_client.stop()
        if self.vector_client_main:
            await self.vector_client_main.stop()
        if self.queue_client:
            await self.queue_client.stop()
        await shutdown_db()
        logger.info("Shutting down the APP")

    async def _initialize_core(self) -> None:
        """Initialize core DB components

        :raise:
            RuntimeError: if database initialization fails

        :returns:
            None
        """
        try:
            await initialize_db()
            logger.info("Core application components initialized.")
        except Exception as e:
            logger.critical(f"FATAL: Core initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Core application initialization failed: {e}") from e
