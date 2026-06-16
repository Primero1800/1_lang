import ast
import base64
import logging
import re
from typing import Any

from app.adapters.ai_client import MistralClient
from app.common.enums import PhraseStatusEnum
from app.common.logging import log_decorator, logger
from app.services.base import BaseService
from app.services.prompt_service import PromptService


class PhraseService(BaseService):
    """Service for processing images through the vision pipeline and persisting phrases"""

    @log_decorator(level=logging.INFO)
    async def _recognize(self, images_raw: list[bytes], prompt: str) -> str | None:
        """Send images to the Mistral vision model and return raw text output

        :param:
            images_raw: list of raw image bytes
            prompt: the vision prompt to use

        :returns:
            raw_text: raw model response string, or None if ai_client is not MistralClient
        """
        if not isinstance(self.ai_client, MistralClient):
            logger.error("ai_client is not MistralClient")
            return None
        images_b64 = [base64.b64encode(img).decode() for img in images_raw]
        return await self.ai_client.vision_chat(images_b64=images_b64, prompt=prompt)

    @log_decorator(level=logging.INFO)
    async def _parse_pixtral_response(self, raw: str) -> list[dict[str, Any]]:
        """Parse the raw Pixtral response into a deduplicated list of phrase dicts

        :param:
            raw: raw text response from the vision model

        :returns:
            result: list of dicts with keys 'phrase' and 'tag', duplicates removed
        """
        # 1. Strip code fences if present
        text = raw.strip()
        match = re.search(r"```(?:python|json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        # 2. Parse Python literal
        try:
            data = ast.literal_eval(text)
        except Exception as exc:
            logger.error("Failed to parse pixtral output: %s", text[:300], exc_info=exc)
            return []

        # 3. Flatten nested structure and deduplicate
        seen: set[tuple[str, str]] = set()
        result: list[dict[str, Any]] = []
        for photo in data:
            for tag_dict in photo:
                for tag, variants in tag_dict.items():
                    for phrase in variants:
                        key = (phrase.strip(), str(tag))
                        if key not in seen:
                            seen.add(key)
                            result.append({"phrase": phrase.strip(), "tag": str(tag)})
        # 4. Return deduplicated phrase list
        return result

    @log_decorator(level=logging.INFO)
    async def _build_rows(
        self, parsed: list[dict[str, Any]], lang: str
    ) -> list[dict[str, Any]]:
        """Convert parsed phrase dicts into DB-ready row dicts

        :param:
            parsed: list of dicts with 'phrase' and 'tag' keys from the parser
            lang: language code to assign to each row

        :returns:
            rows: list of dicts ready for bulk insert (empty phrases skipped)
        """
        rows = []
        for item in parsed:
            cleaned = re.sub(r"[^\w\s\-]", "", item["phrase"], flags=re.UNICODE)
            original = " ".join(cleaned.split()).lower()
            if original:
                rows.append(
                    {
                        "original": original,
                        "tag": item["tag"],
                        "lang": lang,
                        "status": PhraseStatusEnum.DRAFT,
                    }
                )
        return rows

    @log_decorator(level=logging.INFO)
    async def _save_phrases(self, rows: list[dict[str, Any]]) -> int:
        """Persist phrase rows to the database via the UoW factory

        :param:
            rows: list of DB-ready phrase row dicts

        :returns:
            inserted_count: number of rows actually inserted
        """
        async with self.uow_factory as uow:
            return await uow.phrase_repository.bulk_create(rows)

    @log_decorator(level=logging.INFO)
    async def upload_images(self, images_raw: list[bytes], lang: str) -> dict[str, int]:
        """Run the full vision pipeline: recognise → parse → build → save

        :param:
            images_raw: list of raw image bytes from the upload
            lang: target language code for the phrases

        :returns:
            result: dict with 'phrases_found', 'inserted', and 'skipped' counts
        """
        # 1. Send images to vision model
        try:
            prompt = PromptService.get("pixtral_vision", lang)
        except ValueError as exc:
            logger.error("No prompt for lang=%s: %s", lang, exc)
            return {"phrases_found": 0, "inserted": 0, "skipped": 0}
        raw = await self._recognize(images_raw, prompt)
        if not raw:
            return {"phrases_found": 0, "inserted": 0, "skipped": 0}

        # 2. Parse raw model output into structured phrase list
        parsed = await self._parse_pixtral_response(raw)
        if not parsed:
            return {"phrases_found": 0, "inserted": 0, "skipped": 0}

        # 3. Normalise and convert to DB-ready rows
        rows = await self._build_rows(parsed, lang)
        if not rows:
            return {"phrases_found": 0, "inserted": 0, "skipped": 0}

        # 4. Persist to database
        inserted = await self._save_phrases(rows)
        return {
            "phrases_found": len(rows),
            "inserted": inserted,
            "skipped": len(rows) - inserted,
        }
