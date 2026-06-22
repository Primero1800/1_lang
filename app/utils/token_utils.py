import logging
from datetime import date

from app.common.exceptions import IntegrityDataException
from app.common.logging import log_decorator, logger
from app.services.ai_token_usage_service import AiTokenUsageService
from app.uow import get_uow_factory


@log_decorator(level=logging.INFO)
async def record_token_usage(
    model: str,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    name: str = "system",
    usage_date: date | None = None,
) -> None:
    """Accumulate token usage for a single AI call into the daily aggregate

    :param:
        model: AI model identifier (e.g. 'mistral-large-latest')
        operation: pipeline operation name (e.g. 'w2_generate')
        input_tokens: number of input tokens consumed
        output_tokens: number of output tokens produced
        name: actor name, defaults to 'system'
        usage_date: date to record against, defaults to today

    :returns:
        None
    """
    try:
        uow = await get_uow_factory()
        service = AiTokenUsageService(uow=uow)
        await service.accumulate(
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            name=name,
            usage_date=usage_date,
        )
    except IntegrityDataException as exc:
        logger.warning(f"[tokens] integrity error recording usage: {exc}")
    except Exception as exc:
        logger.warning(f"[tokens] failed to record usage: {exc}")
