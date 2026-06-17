import json
import logging
from typing import Any

from pydantic import ValidationError

from app.common.enums import LangEnum, PhraseStatusEnum
from app.common.logging import log_decorator, logger
from app.models.phrases import Phrase
from app.pyd.ai_schemas import MistralResponse, TranslatedPhrase
from app.services.base import BaseService
from app.services.prompt_service import PromptService


class PhraseTranslationService(BaseService):
    """W3 worker service: fetches generated phrases, translates them via Mistral, saves results"""

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
    async def _call_mistral(
        self, batch: list[Phrase], variants: dict[int, dict[str, Any]], lang: str
    ) -> str | None:
        """Send a batch of phrases with their variants to Mistral and return the raw JSON response

        :param:
            batch: list of Phrase objects to translate
            variants: dict of {phrase_id: variants_dict} with tone variants per phrase
            lang: source language code used to select the system prompt ('ru' or 'en')

        :returns:
            raw: raw JSON string from Mistral, or None on failure
        """
        if not self.ai_client.supports_chat:
            logger.error("ai_client does not support chat")
            return None
        try:
            system = PromptService.get("mistral_translate", lang)
        except ValueError as exc:
            logger.error("No prompt for lang=%s: %s", lang, exc)
            return None
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
        return await self.ai_client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": payload},
            ],
            options={"response_format": {"type": "json_object"}},
        )

    def _parse_w3_response(self, raw: str) -> dict[int, dict[str, Any]]:
        """Validate and parse the Mistral JSON response into a phrase_id → translation mapping

        :param:
            raw: raw JSON string from Mistral

        :returns:
            matched: dict of {original_phrase_id: {translated: str, variants: dict}}
        """
        try:
            data = MistralResponse[TranslatedPhrase].model_validate_json(raw)
        except ValidationError as exc:
            logger.error("Failed to validate w3 response: %s", exc)
            return {}
        result = {}
        for item in data.results:
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
    async def w3_translate(self, batch_size: int) -> dict[str, int]:
        """Run one W3 cycle: fetch → get variants → call Mistral → save translations → mark done/failed

        :param:
            batch_size: number of phrases per Mistral call

        :returns:
            result: dict with 'processed', 'failed', and 'skipped' counts
        """
        # 1. Fetch a batch of eligible phrases
        batch = await self._fetch_batch(batch_size)
        if not batch:
            return {"processed": 0, "failed": 0, "skipped": 1}

        sent_ids = {p.id for p in batch}
        lang_raw = batch[0].lang
        actual_lang = (
            lang_raw.value if isinstance(lang_raw, LangEnum) else str(lang_raw)
        )
        opposite_lang = "en" if actual_lang == "ru" else "ru"

        # 2. Fetch existing tone variants for all batch phrases
        variants = await self._fetch_variants(list(sent_ids))

        # 3. Call Mistral to translate the batch in one request
        raw = await self._call_mistral(batch, variants, actual_lang)
        matched = self._parse_w3_response(raw) if raw else {}

        # 4. Compute which phrase IDs were returned and which failed
        returned_ids = set(matched.keys())
        failed_ids = sent_ids - returned_ids

        # 5. Persist translated phrases, their variants, and update source phrase statuses
        async with self.uow_factory as uow:
            if matched:
                id_to_tag = {p.id: p.tag for p in batch}

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

                translated_texts = [matched[pid]["translated"] for pid in returned_ids]
                text_to_id = await uow.phrase_repository.get_ids_by_originals(
                    originals=translated_texts, lang=opposite_lang
                )

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
