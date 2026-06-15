import ast
import base64
import logging
import re
from typing import Any

from app.adapters.ai_client import MistralClient
from app.common.enums import PhraseStatusEnum
from app.common.logging import log_decorator, logger
from app.services.base import BaseService


class PhraseService(BaseService):
    @log_decorator(level=logging.INFO)
    async def _recognize(self, images_raw: list[bytes], prompt: str) -> str | None:
        if not isinstance(self.ai_client, MistralClient):
            logger.error("ai_client is not MistralClient")
            return None
        images_b64 = [base64.b64encode(img).decode() for img in images_raw]
        return await self.ai_client.vision_chat(images_b64=images_b64, prompt=prompt)

    @log_decorator(level=logging.INFO)
    async def _parse_pixtral_response(self, raw: str) -> list[dict[str, Any]]:
        text = raw.strip()
        match = re.search(r"```(?:python|json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        try:
            data = ast.literal_eval(text)
        except Exception as exc:
            logger.error("Failed to parse pixtral output: %s", text[:300], exc_info=exc)
            return []

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
        return result

    @log_decorator(level=logging.INFO)
    async def _build_rows(
        self, parsed: list[dict[str, Any]], lang: str
    ) -> list[dict[str, Any]]:
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
        async with self.uow_factory as uow:
            return await uow.phrase_repository.bulk_create(rows)

    @log_decorator(level=logging.INFO)
    async def upload_images(
        self, images_raw: list[bytes], prompt: str, lang: str
    ) -> dict[str, int]:
        raw = await self._recognize(images_raw, prompt)
        if not raw:
            return {"phrases_found": 0, "inserted": 0, "skipped": 0}

        parsed = await self._parse_pixtral_response(raw)
        if not parsed:
            return {"phrases_found": 0, "inserted": 0, "skipped": 0}

        rows = await self._build_rows(parsed, lang)
        if not rows:
            return {"phrases_found": 0, "inserted": 0, "skipped": 0}

        inserted = await self._save_phrases(rows)
        return {
            "phrases_found": len(rows),
            "inserted": inserted,
            "skipped": len(rows) - inserted,
        }
