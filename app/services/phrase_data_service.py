import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr

from app.common.enums import LangEnum, PhraseStatusEnum
from app.common.exceptions import GenerationPipelineException
from app.common.logging import log_decorator, logger
from app.core.config import settings
from app.models.phrases import Phrase
from app.pyd.ai_schemas import VariantsResponse
from app.services.base import BaseService
from app.services.prompt_service import PromptService

_W2_MODEL = settings.MISTRAL_MODEL
_w2_llm = ChatMistralAI(
    model_name=_W2_MODEL,
    api_key=SecretStr(settings.MISTRAL_API_KEY),
    timeout=settings.MISTRAL_TIMEOUT_SEC,
)


class PhraseDataService(BaseService):
    """W2 worker service: fetches draft phrases, generates tone variants via Mistral, saves results"""

    _llm = _w2_llm
    _OPERATION = "w2_generate"

    @log_decorator(level=logging.INFO)
    async def _fetch_batch(self, batch_size: int) -> list[Phrase]:
        """Atomically claim a batch of phrases ready for variant generation

        :param:
            batch_size: maximum number of phrases to claim in one call

        :returns:
            batch: list of claimed Phrase objects (empty if nothing is ready)
        """
        async with self.uow_factory as uow:
            # 1. Pick the highest-priority phrase (DRAFT / stuck) to anchor the batch lang
            first = await uow.phrase_repository.get_first_for_processing(
                in_progress_status=PhraseStatusEnum.GENERATING_IN_PROGRESS,
                priority_status=PhraseStatusEnum.GENERATING_FAILED,
                base_statuses=[PhraseStatusEnum.DRAFT],
            )
            if not first:
                return []
            logger.info(
                f"[W2, generating] First chosen: id={first.id}, lang={first.lang}, status={first.status}"
            )
            # 2. Fill the rest of the batch with same-lang phrases
            rest = await uow.phrase_repository.get_batch_for_processing(
                in_progress_status=PhraseStatusEnum.GENERATING_IN_PROGRESS,
                priority_status=PhraseStatusEnum.GENERATING_FAILED,
                base_statuses=[PhraseStatusEnum.DRAFT],
                lang=first.lang,
                exclude_id=first.id,
                limit=batch_size - 1,
            )
            batch = [first, *rest]
            for i, member in enumerate(batch, start=1):
                logger.info(
                    f"[W2, generating] {i} chosen: id={member.id}, lang={member.lang}, status={member.status}"
                )
            # 3. Mark all claimed phrases as in-progress to prevent duplicate claiming
            await uow.phrase_repository.update_status(
                ids=[p.id for p in batch],
                status=PhraseStatusEnum.GENERATING_IN_PROGRESS,
            )
        return batch

    @log_decorator(level=logging.INFO)
    async def _build_w2_message(self, data: dict[str, Any]) -> list[BaseMessage]:
        """Build system + user messages for the W2 variant generation call

        :param:
            data: dict with 'batch' (list[Phrase]) and 'lang' (str)

        :returns:
            messages: [SystemMessage, HumanMessage] ready for the LLM
        """
        batch: list[Phrase] = data["batch"]
        lang: str = data["lang"]
        system = PromptService.get("mistral_variants", lang)
        payload = json.dumps(
            [{"id": p.id, "phrase": p.original, "tag": p.tag} for p in batch],
            ensure_ascii=False,
        )
        return [SystemMessage(content=system), HumanMessage(content=payload)]

    @log_decorator(level=logging.INFO)
    async def _fire_token_task(self, data: dict[str, Any]) -> VariantsResponse:
        """Publish token usage to Redis Streams and return the parsed response

        :param:
            data: dict with 'raw' (AIMessage) and 'parsed' (VariantsResponse)

        :returns:
            parsed: the structured response from the LLM
        """
        parsed: VariantsResponse | None = data.get("parsed")
        if parsed is None:
            raise GenerationPipelineException(
                detail="LLM returned invalid structured output"
            )
        usage = (data["raw"].usage_metadata or {}) if data.get("raw") else {}
        self._queue_token_usage(
            model=_W2_MODEL,
            operation=self._OPERATION,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        return parsed

    @log_decorator(level=logging.INFO)
    async def _parse_variants(
        self, parsed: VariantsResponse
    ) -> dict[int, dict[str, Any]]:
        """Convert structured LLM response into a phrase_id → variants mapping

        :param:
            parsed: validated VariantsResponse from the LLM

        :returns:
            matched: dict of phrase_id → tone variants dict
        """
        result: dict[int, dict[str, Any]] = {}
        for item in parsed.results:
            variants = {
                k: v
                for k, v in item.model_dump(exclude={"id"}).items()
                if v is not None
            }
            if variants:
                result[item.id] = variants
        return result

    @log_decorator(level=logging.INFO)
    async def _save_results(
        self, matched: dict[int, dict[str, Any]], sent_ids: set[int]
    ) -> dict[str, int]:
        """Persist generated variants and update phrase statuses

        :param:
            matched: phrase_id → variants dict from the LLM
            sent_ids: set of all phrase IDs sent to the LLM in this batch

        :returns:
            result: dict with 'processed', 'failed', and 'skipped' counts
        """
        returned_ids = set(matched.keys())
        failed_ids = sent_ids - returned_ids

        async with self.uow_factory as uow:
            if matched:
                rows = [
                    {"phrase_id": pid, "variants": variants}
                    for pid, variants in matched.items()
                ]
                await uow.phrase_data_repository.bulk_upsert_variants(rows)
            if returned_ids:
                logger.info(f"[W2, generating]: returned ids {returned_ids}")
                await uow.phrase_repository.update_status(
                    ids=list(returned_ids), status=PhraseStatusEnum.GENERATING_DONE
                )
            if failed_ids:
                logger.info(f"[W2, generating]: failed ids {failed_ids}")
                await uow.phrase_repository.update_status(
                    ids=list(failed_ids), status=PhraseStatusEnum.GENERATING_FAILED
                )

        return {
            "processed": len(returned_ids),
            "failed": len(failed_ids),
            "skipped": 0,
        }

    @log_decorator(level=logging.INFO)
    async def w2_generate(self, batch_size: int) -> dict[str, int]:
        """Run one W2 cycle: fetch → call Mistral → parse → save variants → mark done/failed

        :param:
            batch_size: number of phrases per Mistral call

        :returns:
            result: dict with 'processed', 'failed', and 'skipped' counts
        """
        batch = await self._fetch_batch(batch_size)
        if not batch:
            return {"processed": 0, "failed": 0, "skipped": 0}

        lang_raw = batch[0].lang
        lang = lang_raw.value if isinstance(lang_raw, LangEnum) else str(lang_raw)
        sent_ids = {p.id for p in batch}

        # 1. Attach structured output schema to the LLM
        llm = self._llm.with_structured_output(
            VariantsResponse, method="json_mode", include_raw=True
        )

        # 2. Bind sent_ids into the _save_results step
        async def _save_for_batch(matched: dict[int, dict[str, Any]]) -> dict[str, int]:
            return await self._save_results(matched, sent_ids)

        # 3. Assemble the full LangChain processing chain
        chain = (
            RunnableLambda(self._build_w2_message)
            | llm
            | RunnableLambda(self._fire_token_task)
            | RunnableLambda(self._parse_variants)
            | RunnableLambda(_save_for_batch)
        ).with_config(run_name=self._OPERATION)

        # 4. Invoke the chain; mark all phrases as failed on any error
        try:
            return await chain.ainvoke({"batch": batch, "lang": lang})
        except Exception as exc:
            if not isinstance(exc, GenerationPipelineException):
                logger.error("[W2] generation failed: %s", exc, exc_info=exc)
            async with self.uow_factory as uow:
                await uow.phrase_repository.update_status(
                    ids=list(sent_ids), status=PhraseStatusEnum.GENERATING_FAILED
                )
            if isinstance(exc, GenerationPipelineException):
                raise
            raise GenerationPipelineException(detail=str(exc)) from exc
