import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr

from app.common.enums import LangEnum, PhraseStatusEnum
from app.common.exceptions import TranslationPipelineException
from app.common.logging import log_decorator, logger
from app.core.config import settings
from app.models.phrases import Phrase
from app.pyd.ai_schemas import TranslationResponse
from app.services.base import BaseService
from app.services.prompt_service import PromptService

_W3_MODEL = settings.MISTRAL_MODEL
_w3_llm = ChatMistralAI(
    model_name=_W3_MODEL,
    api_key=SecretStr(settings.MISTRAL_API_KEY),
    timeout=settings.MISTRAL_TIMEOUT_SEC,
)


class PhraseTranslationService(BaseService):
    """W3 worker service: fetches generated phrases, translates them via Mistral, saves results"""

    _llm = _w3_llm

    @log_decorator(level=logging.INFO)
    async def _fetch_batch(self, batch_size: int) -> list[Phrase]:
        """Atomically claim a batch of phrases ready for translation

        :param:
            batch_size: maximum number of phrases to claim in one call

        :returns:
            batch: list of claimed Phrase objects (empty if nothing is ready)
        """
        async with self.uow_factory as uow:
            # 1. Pick the highest-priority phrase ready for translation (FAILED / stuck)
            first = await uow.phrase_repository.get_first_for_processing(
                in_progress_status=PhraseStatusEnum.TRANSLATING_IN_PROGRESS,
                priority_status=PhraseStatusEnum.TRANSLATING_FAILED,
                base_statuses=[PhraseStatusEnum.GENERATING_DONE],
            )
            if not first:
                return []
            logger.info(
                f"[W3, translating] First chosen: id={first.id}, lang={first.lang}, status={first.status}"
            )
            # 2. Fill the rest of the batch with same-lang phrases
            rest = await uow.phrase_repository.get_batch_for_processing(
                in_progress_status=PhraseStatusEnum.TRANSLATING_IN_PROGRESS,
                priority_status=PhraseStatusEnum.TRANSLATING_FAILED,
                base_statuses=[PhraseStatusEnum.GENERATING_DONE],
                lang=first.lang,
                exclude_id=first.id,
                limit=batch_size - 1,
            )
            batch = [first, *rest]
            for i, member in enumerate(batch, start=1):
                logger.info(
                    f"[W3, translating] {i} chosen: id={member.id}, lang={member.lang}, status={member.status}"
                )
            # 3. Mark all claimed phrases as in-progress (SKIP LOCKED prevents duplicates)
            await uow.phrase_repository.update_status(
                ids=[p.id for p in batch],
                status=PhraseStatusEnum.TRANSLATING_IN_PROGRESS,
            )
        return batch

    @log_decorator(level=logging.INFO)
    async def _fetch_variants(self, phrase_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch phrase_data variants for the given phrase IDs

        :param:
            phrase_ids: list of phrase IDs to fetch variants for

        :returns:
            variants: dict of {phrase_id: variants_dict}
        """
        async with self.uow_factory as uow:
            records = await uow.phrase_data_repository.get_by_phrase_ids(phrase_ids)
        return {r.phrase_id: r.variants for r in records}

    @log_decorator(level=logging.INFO)
    async def _build_w3_message(self, data: dict) -> list[BaseMessage]:
        """Build system + user messages for the W3 translation call

        :param:
            data: dict with 'batch' (list[Phrase]), 'variants' (dict), and 'lang' (str)

        :returns:
            messages: [SystemMessage, HumanMessage] ready for the LLM
        """
        batch: list[Phrase] = data["batch"]
        variants: dict[int, dict[str, Any]] = data["variants"]
        lang: str = data["lang"]
        system = PromptService.get("mistral_translate", lang)
        payload = json.dumps(
            [
                {
                    "id": p.id,
                    "phrase": p.original,
                    "tag": p.tag,
                    **(variants.get(p.id) or {}),
                }
                for p in batch
            ],
            ensure_ascii=False,
        )
        return [SystemMessage(content=system), HumanMessage(content=payload)]

    @log_decorator(level=logging.INFO)
    async def _fire_token_task(self, data: dict) -> TranslationResponse:
        """Publish token usage to Redis Streams and return the parsed response

        :param:
            data: dict with 'raw' (AIMessage) and 'parsed' (TranslationResponse)

        :returns:
            parsed: the structured response from the LLM
        """
        parsed: TranslationResponse | None = data.get("parsed")
        if parsed is None:
            raise TranslationPipelineException(
                detail="LLM returned invalid structured output"
            )
        usage = (data["raw"].usage_metadata or {}) if data.get("raw") else {}
        self._queue_token_usage(
            model=_W3_MODEL,
            operation="w3_translate",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        return parsed

    @log_decorator(level=logging.INFO)
    async def _parse_translations(
        self, parsed: TranslationResponse
    ) -> dict[int, dict[str, Any]]:
        """Convert structured LLM response into a phrase_id → translation mapping

        :param:
            parsed: validated TranslationResponse from the LLM

        :returns:
            matched: dict of {phrase_id: {"translated": str, "variants": dict}}
        """
        result: dict[int, dict[str, Any]] = {}
        for item in parsed.results:
            translated = item.translated.strip()
            if not translated:
                continue
            tone_variants = {
                k: v
                for k, v in item.model_dump(exclude={"id", "translated"}).items()
                if v is not None
            }
            result[item.id] = {"translated": translated, "variants": tone_variants}
        return result

    @log_decorator(level=logging.INFO)
    async def _save_translations(
        self,
        matched: dict[int, dict[str, Any]],
        sent_ids: set[int],
        batch: list[Phrase],
        opposite_lang: str,
    ) -> dict[str, int]:
        """Persist translated phrases, their variants, and update source phrase statuses

        :param:
            matched: phrase_id → {"translated", "variants"} from the LLM
            sent_ids: set of all phrase IDs sent to the LLM in this batch
            batch: original Phrase objects (used for tag lookup)
            opposite_lang: target language code for new translated phrases

        :returns:
            result: dict with 'processed', 'failed', and 'skipped' counts
        """
        returned_ids = set(matched.keys())
        failed_ids = sent_ids - returned_ids

        async with self.uow_factory as uow:
            if matched:
                id_to_tag = {p.id: p.tag for p in batch}
                # 1. Insert translated texts as new Phrase rows in the target language
                phrase_rows = [
                    {
                        "original": matched[pid]["translated"],
                        "tag": id_to_tag[pid],
                        "lang": opposite_lang,
                        "status": PhraseStatusEnum.TRANSLATING_DONE,
                    }
                    for pid in returned_ids
                ]
                await uow.phrase_repository.bulk_create(phrase_rows)

                # 2. Resolve auto-assigned IDs for the just-inserted translations
                translated_texts = [matched[pid]["translated"] for pid in returned_ids]
                text_to_id = await uow.phrase_repository.get_ids_by_originals(
                    originals=translated_texts, lang=opposite_lang
                )

                # 3. Upsert tone variants for each new translated phrase
                variant_rows = [
                    {
                        "phrase_id": text_to_id[matched[pid]["translated"]],
                        "variants": matched[pid]["variants"],
                    }
                    for pid in returned_ids
                    if matched[pid]["translated"] in text_to_id
                ]
                if variant_rows:
                    await uow.phrase_data_repository.bulk_upsert_variants(variant_rows)

            # 4. Update source phrase statuses (DONE / FAILED)
            if returned_ids:
                logger.info(f"[W3, translating]: returned ids {returned_ids}")
                await uow.phrase_repository.update_status(
                    ids=list(returned_ids), status=PhraseStatusEnum.TRANSLATING_DONE
                )
            if failed_ids:
                logger.info(f"[W3, translating]: failed ids {failed_ids}")
                await uow.phrase_repository.update_status(
                    ids=list(failed_ids), status=PhraseStatusEnum.TRANSLATING_FAILED
                )

        return {
            "processed": len(returned_ids),
            "failed": len(failed_ids),
            "skipped": 0,
        }

    @log_decorator(level=logging.INFO)
    async def w3_translate(self, batch_size: int) -> dict[str, int]:
        """Run one W3 cycle: fetch → get variants → call Mistral → save translations → mark done/failed

        :param:
            batch_size: number of phrases per Mistral call

        :returns:
            result: dict with 'processed', 'failed', and 'skipped' counts
        """
        batch = await self._fetch_batch(batch_size)
        if not batch:
            return {"processed": 0, "failed": 0, "skipped": 1}

        lang_raw = batch[0].lang
        lang = lang_raw.value if isinstance(lang_raw, LangEnum) else str(lang_raw)
        opposite_lang = "en" if lang == "ru" else "ru"
        sent_ids = {p.id for p in batch}

        # 1. Attach structured output schema to the LLM
        llm = self._llm.with_structured_output(
            TranslationResponse, method="json_mode", include_raw=True
        )

        # 2. Bind batch context into the _save_translations step
        async def _save_for_batch(matched: dict[int, dict[str, Any]]) -> dict:
            return await self._save_translations(
                matched, sent_ids, batch, opposite_lang
            )

        # 3. Assemble the full LangChain processing chain
        chain = (
            RunnableLambda(self._build_w3_message)
            | llm
            | RunnableLambda(self._fire_token_task)
            | RunnableLambda(self._parse_translations)
            | RunnableLambda(_save_for_batch)
        )

        # 4. Invoke the chain; mark all phrases as failed on any error
        try:
            variants = await self._fetch_variants(list(sent_ids))
            return await chain.ainvoke(
                {"batch": batch, "variants": variants, "lang": lang}
            )
        except Exception as exc:
            if not isinstance(exc, TranslationPipelineException):
                logger.error("[W3] translation failed: %s", exc, exc_info=exc)
            async with self.uow_factory as uow:
                await uow.phrase_repository.update_status(
                    ids=list(sent_ids), status=PhraseStatusEnum.TRANSLATING_FAILED
                )
            if isinstance(exc, TranslationPipelineException):
                raise
            raise TranslationPipelineException(detail=str(exc)) from exc
