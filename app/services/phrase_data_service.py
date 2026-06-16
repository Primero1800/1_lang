import json
import logging

import aiohttp
from pydantic import ValidationError

from app.adapters.ai_client import MistralClient
from app.common.logging import log_decorator, logger
from app.models.phrases import Phrase
from app.pyd.ai_schemas import W2MistralResponse
from app.services.base import BaseService
from app.services.prompt_service import PromptService


class PhraseDataService(BaseService):
    """W2 worker service: fetches draft phrases, generates tone variants via Mistral, saves results"""

    @log_decorator(level=logging.INFO)
    async def _fetch_batch(self, batch_size: int) -> list[Phrase]:
        """Atomically claim a batch of phrases ready for variant generation

        :param:
            batch_size: maximum number of phrases to claim in one call

        :returns:
            batch: list of claimed Phrase objects (empty if nothing is ready)
        """
        async with self.uow_factory as uow:
            # 1. Pick the highest-priority phrase (DRAFT / FAILED / stuck)
            first = await uow.phrase_repository.get_first_for_generation()
            if not first:
                return []
            logger.info(
                f"[W2, generating] First chosen: id={first.id}, lang={first.lang}, status={first.status}"
            )
            # 2. Fill the rest of the batch with same-lang phrases
            rest = await uow.phrase_repository.get_batch_for_generation(
                lang=first.lang, exclude_id=first.id, limit=batch_size - 1
            )
            batch = [first, *rest]
            for i, member in enumerate(batch, start=1):
                logger.info(
                    f"[W2, generating] {i}st chosen: id={member.id}, lang={member.lang}, status={member.status}"
                )
            # 3. Mark all claimed phrases as in-progress (SKIP LOCKED prevents duplicates)
            await uow.phrase_repository.mark_generating_in_progress(
                ids=[p.id for p in batch]
            )
        return batch

    @log_decorator(level=logging.INFO)
    async def _call_mistral(self, batch: list[Phrase], lang: str) -> str | None:
        """Send a batch of phrases to Mistral and return the raw JSON response

        :param:
            batch: list of Phrase objects to generate variants for
            lang: language code used to select the system prompt ('ru' or 'en')

        :returns:
            raw: raw JSON string from Mistral, or None on failure
        """
        if not isinstance(self.ai_client, MistralClient):
            logger.error("ai_client is not MistralClient")
            return None
        try:
            system = PromptService.get("mistral_variants", lang)
        except ValueError as exc:
            logger.error("No prompt for lang=%s: %s", lang, exc)
            return None
        payload = json.dumps(
            [{"id": p.id, "phrase": p.original, "tag": p.tag} for p in batch],
            ensure_ascii=False,
        )
        return await self.ai_client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": payload},
            ],
            options={"response_format": {"type": "json_object"}},
            timeout=aiohttp.ClientTimeout(total=60),
        )

    def _parse_w2_response(self, raw: str) -> dict[int, dict]:
        """Validate and parse the Mistral JSON response into a phrase_id → variants mapping

        :param:
            raw: raw JSON string from Mistral

        :returns:
            matched: dict of phrase_id → tone variants dict (empty on validation error)
        """
        try:
            data = W2MistralResponse.model_validate_json(raw)
        except ValidationError as exc:
            logger.error("Failed to validate w2 response: %s", exc)
            return {}
        result = {}
        for item in data.results:
            variants = {
                k: v
                for k, v in item.model_dump(exclude={"id"}).items()
                if v is not None
            }
            if variants:
                result[item.id] = variants
        return result

    @log_decorator(level=logging.INFO)
    async def w2_generate(self, batch_size: int) -> dict[str, int]:
        """Run one W2 cycle: fetch → call Mistral → save variants → mark done/failed

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
        actual_lang = lang_raw.value if hasattr(lang_raw, "value") else str(lang_raw)

        # 2. Call Mistral for all phrases in one request
        raw = await self._call_mistral(batch, actual_lang)

        # 3. Parse response and compute failed IDs
        matched = self._parse_w2_response(raw) if raw else {}
        returned_ids = set(matched.keys())
        failed_ids = sent_ids - returned_ids

        # 4. Save variants and update phrase statuses
        async with self.uow_factory as uow:
            if matched:
                rows = [
                    {"phrase_id": pid, "variants": variants}
                    for pid, variants in matched.items()
                ]
                await uow.phrase_data_repository.bulk_upsert_variants(rows)
            if returned_ids:
                logger.info(f"[W2, generating]: returned ids {returned_ids}")
                await uow.phrase_repository.mark_generating_done(ids=list(returned_ids))
            if failed_ids:
                logger.info(f"[W2, generating]: failed ids {failed_ids}")
                await uow.phrase_repository.mark_generating_failed(ids=list(failed_ids))

        return {
            "processed": len(returned_ids),
            "failed": len(failed_ids),
            "skipped": 0,
        }
