from app.common.logging import logger
from app.services.base import BaseService


class PipelineSchedulerService(BaseService):
    """ETL scheduler: aggregates phrase pipeline statuses and publishes a dispatch snapshot to Redis Pub/Sub"""

    _CHANNEL = "pipeline:status"

    async def run(self) -> None:
        """Execute one ETL tick: extract → transform → publish

        Called by APScheduler every minute.

        :returns:
            None
        """
        logger.info("[pipeline_scheduler] tick")
