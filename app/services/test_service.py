import ast
import base64
import logging
import re

from app.common.logging import log_decorator, logger
from app.pyd.requests import SearchSettings, TagExclusionFilters
from app.services.base import BaseService
from app.services.prompt_service import PromptService


class TestService(BaseService):
    """T1 search test service: vision → embed → Qdrant search pipeline"""

    @log_decorator(level=logging.INFO)
    async def t1_search(
        self,
        image_raw: bytes,
        filters: TagExclusionFilters,
        search_settings: SearchSettings,
    ) -> list[str]:
        """Orchestrate the full T1 pipeline: vision → embed → search → filter by mood

        :param:
            image_raw: raw bytes of the uploaded image
            filters: tag exclusion flags
            search_settings: target language and mood tone

        :returns:
            phrases: flat list of matched variant strings
        """
        # Step 1: extract up to 3 observation phrases from the image
        phrases = await self._t1_get_phrases(image_raw, lang=search_settings.lang.value)
        logger.info(f"[T1] step 1 — extracted {len(phrases)} phrase(s): {phrases}")
        if not phrases:
            return []

        # Steps 2–5: embed → search → filter — coming soon
        raise NotImplementedError

    @log_decorator(level=logging.DEBUG)
    async def _t1_get_phrases(self, image_raw: bytes, lang: str) -> list[str]:
        """Send image to Mistral vision and return up to 3 observation phrases

        :param:
            image_raw: raw image bytes
            lang: target language code ('ru' or 'en')

        :returns:
            phrases: list of up to 3 observation phrase strings
        """
        if not self.ai_client.supports_vision:
            logger.error("[T1, step 1] ai_client does not support vision")
            return []
        try:
            prompt = PromptService.get("t1_vision", lang)
        except Exception as exc:
            logger.error("[T1, step 1] no prompt for lang=%s", lang, exc_info=exc)
            return []
        image_b64 = base64.b64encode(image_raw).decode()
        raw = await self.ai_client.vision_chat(images_b64=[image_b64], prompt=prompt)
        if not raw:
            logger.warning("[T1, step 1] vision returned empty response")
            return []
        phrases = self._parse_vision_phrases(raw)
        logger.info(f"[T1, step 1] parsed {len(phrases)} phrase(s) from vision output")
        return phrases[:3]

    @staticmethod
    def _parse_vision_phrases(raw: str) -> list[str]:
        """Parse raw vision output as a flat JSON list of phrase strings

        :param:
            raw: raw text response from the vision model

        :returns:
            phrases: list of phrase strings
        """
        import json
        text = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(p).strip() for p in data if p]
        except Exception as exc:
            logger.error("[T1, step 1] failed to parse vision output: %s", text[:300], exc_info=exc)
        return []
