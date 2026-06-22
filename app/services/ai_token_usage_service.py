import logging
from datetime import date

from app.common.logging import log_decorator, logger
from app.uow import UnitOfWork


class AiTokenUsageService:
    """Service for accumulating AI token usage into daily aggregates"""

    def __init__(self, uow: UnitOfWork) -> None:
        """Initialize with a UnitOfWork instance

        :param:
            uow: unit of work providing the token usage repository

        :returns:
            None
        """
        self._uow = uow

    @log_decorator(level=logging.DEBUG)
    async def accumulate(
        self,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        name: str = "system",
        usage_date: date | None = None,
    ) -> None:
        """Accumulate token counts for the given model and operation into the daily aggregate

        :param:
            model: AI model identifier
            operation: pipeline operation name (e.g. 'w2_generate')
            input_tokens: number of input tokens to add
            output_tokens: number of output tokens to add
            name: actor name, defaults to 'system'
            usage_date: date to record against, defaults to today

        :returns:
            None
        """
        async with self._uow as uow:
            await uow.ai_token_usage_repository.accumulate(
                model=model,
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                name=name,
                usage_date=usage_date,
            )
        logger.debug(
            f"[tokens] {operation} | {model} | in={input_tokens} out={output_tokens}"
        )

    @log_decorator(level=logging.DEBUG)
    async def bulk_accumulate(self, rows: list[dict]) -> None:
        """Upsert a pre-aggregated batch of token usage rows in a single DB statement

        :param:
            rows: list of dicts with keys model, operation, name, date, input_tokens, output_tokens

        :returns:
            None
        """
        async with self._uow as uow:
            await uow.ai_token_usage_repository.bulk_accumulate(rows)
        logger.debug(f"[tokens] bulk_accumulate {len(rows)} row(s)")
