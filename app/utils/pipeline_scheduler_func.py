from app.dependencies.services import get_base_deps_standalone
from app.services.pipeline_scheduler_service import PipelineSchedulerService


async def run_pipeline_scheduler() -> None:
    """Build fresh deps and run one ETL tick of the pipeline scheduler.

    Called by APScheduler every minute.

    :returns:
        None
    """
    base_deps = await get_base_deps_standalone()
    await PipelineSchedulerService(base_deps).run()
