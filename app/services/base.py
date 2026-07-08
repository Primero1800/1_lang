import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.adapters.queue_client import MessageQueueClientAbstract
from app.adapters.vector_client import VectorClientAbstract
from app.common.enums import PhraseStatusEnum
from app.common.exceptions import BaseCustomException
from app.common.logging import log_decorator, logger
from app.core.config import settings
from app.uow import UnitOfWork


@dataclass
class BaseDeps:
    """Infrastructure dependencies shared across all services"""

    uow_factory: UnitOfWork
    vector_client: VectorClientAbstract
    vector_client_main: VectorClientAbstract
    queue_client: MessageQueueClientAbstract


class BaseService:
    """Base service to be extended by concrete implementations"""

    def __init__(
        self,
        base_deps: BaseDeps,
        uow: UnitOfWork | None = None,
    ) -> None:
        """Initialize the service with infrastructure dependencies

        :param:
            base_deps: container with shared infrastructure clients
            uow: optional request-scoped UnitOfWork session

        :returns:
            None
        """
        self.uow_factory = base_deps.uow_factory
        self.uow = uow
        self.vector_client = base_deps.vector_client
        self.vector_client_main = base_deps.vector_client_main
        self.queue_client = base_deps.queue_client

    def _queue_token_usage(
        self,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> None:
        """Fire-and-forget: publish token usage to Redis Streams

        :param:
            model: model identifier string
            operation: pipeline operation name (e.g. 'w2_generate')
            input_tokens: number of input tokens consumed
            output_tokens: number of output tokens consumed (0 for embeddings)

        :returns:
            None
        """
        asyncio.create_task(
            self.queue_client.xadd(
                settings.REDIS_TOKENS_STREAM,
                {
                    "model": model,
                    "operation": operation,
                    "input_tokens": str(input_tokens),
                    "output_tokens": str(output_tokens),
                },
            )
        )


class BaseWorkerService(BaseService):
    """Base class for pipeline worker services — provides shared batch-fetch and token-fire logic"""

    _llm_model = settings.MISTRAL_MODEL
    _operation = "base_operation"
    _log_operation = "Base, operating"
    _pipeline_exception_class = BaseCustomException

    _base_status = PhraseStatusEnum.DRAFT
    _in_progress_status = PhraseStatusEnum.GENERATING_IN_PROGRESS
    _priority_status = PhraseStatusEnum.GENERATING_FAILED

    _success_status = PhraseStatusEnum.GENERATING_DONE
    _failed_status = PhraseStatusEnum.GENERATING_FAILED

    @log_decorator(level=logging.INFO)
    async def _fire_token_task(self, data: dict[str, Any]) -> Any:
        """Publish token usage to Redis Streams and return the parsed LLM response

        :param:
            data: dict with 'raw' (AIMessage) and 'parsed' (structured LLM output)

        :returns:
            parsed: the structured response from the LLM
        """
        parsed = data.get("parsed")
        if parsed is None:
            raise self._pipeline_exception_class(
                detail="LLM returned invalid structured output"
            )
        usage = (data["raw"].usage_metadata or {}) if data.get("raw") else {}
        self._queue_token_usage(
            model=self._llm_model,
            operation=self._operation,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        return parsed

    @log_decorator(level=logging.INFO)
    async def _fetch_batch(self, batch_size: int) -> list[Any]:
        """Atomically claim a same-language batch of phrases ready for processing

        :param:
            batch_size: maximum number of phrases to claim in one call

        :returns:
            batch: list of claimed Phrase objects (empty if nothing is ready)
        """
        async with self.uow_factory as uow:
            # 1. Pick the highest-priority phrase to anchor the batch lang
            first = await uow.phrase_repository.get_first_for_processing(
                in_progress_status=self._in_progress_status,
                priority_status=self._priority_status,
                base_statuses=[self._base_status],
            )
            if not first:
                return []
            logger.info(
                f"[{self._log_operation}] First chosen: id={first.id}, lang={first.lang}, status={first.status}"
            )
            # 2. Fill the rest of the batch with same-lang phrases
            rest = await uow.phrase_repository.get_batch_for_processing(
                in_progress_status=self._in_progress_status,
                priority_status=self._priority_status,
                base_statuses=[self._base_status],
                lang=first.lang,
                exclude_id=first.id,
                limit=batch_size - 1,
            )
            batch = [first, *rest]
            for i, member in enumerate(batch, start=1):
                logger.info(
                    f"[{self._log_operation}] {i} chosen: id={member.id}, lang={member.lang}, status={member.status}"
                )
            # 3. Mark all claimed phrases as in-progress to prevent duplicate claiming
            await uow.phrase_repository.update_status(
                ids=[p.id for p in batch],
                status=self._in_progress_status,
            )
        return batch
