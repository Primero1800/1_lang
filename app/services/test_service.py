import ast
import base64
import logging
import re

from qdrant_client.models import ScoredPoint

from app.common.logging import log_decorator, logger
from app.pyd.requests import SearchSettings, TagExclusionFilters
from app.repositories.phrase_vector_repository import PhraseVectorRepository
from app.services.base import BaseDeps, BaseService
from app.services.prompt_service import PromptService
from app.uow import UnitOfWork


class TestService(BaseService):
    """T1 search test service: vision → embed → Qdrant search pipeline"""

    def __init__(
        self,
        base_deps: BaseDeps,
        vector_repository: PhraseVectorRepository,
        uow: UnitOfWork | None = None,
    ) -> None:
        """Initialize with shared infrastructure deps and Qdrant vector repository

        :param:
            base_deps: shared infrastructure dependencies
            vector_repository: repository for Qdrant search and upsert operations
            uow: optional request-scoped UnitOfWork session

        :returns:
            None
        """
        super().__init__(base_deps, uow)
        self.vector_repository = vector_repository

    @log_decorator(level=logging.INFO)
    async def t1_search(
        self,
        image_raw: bytes,
        filters: TagExclusionFilters,
        search_settings: SearchSettings,
    ) -> dict[str, dict]:
        """Orchestrate the full T1 pipeline: vision → embed → search → filter by mood and gender

        :param:
            image_raw: raw bytes of the uploaded image
            filters: tag exclusion flags
            search_settings: target language and mood tone

        :returns:
            result: dict of {original: {tag: str, phrases: list[str]}}
        """
        # Step 1: extract gender and up to 3 observation phrases from the image
        gender, phrases = await self._t1_get_phrases(image_raw, lang=search_settings.lang.value)
        logger.info(f"[T1] step 1 — gender={gender}, extracted {len(phrases)} phrase(s): {phrases}")
        if not phrases:
            return {}

        # Step 2: embed phrases with search_query prefix
        vectors = await self._t1_embed_phrases(phrases)
        logger.info(f"[T1] step 2 — embedded {len(vectors)}/{len(phrases)} phrase(s)")
        if not vectors:
            return {}

        # Step 3: batch search in Qdrant filtered by lang and excluded tags
        excluded_tags = [
            tag for tag, excluded in {
                "behavior": filters.not_behavior,
                "appearance": filters.not_appearance,
                "age": filters.not_age,
                "mood": filters.not_mood,
                "posture": filters.not_posture,
                "hairstyle": filters.not_hairstyle,
            }.items() if excluded
        ]
        results = await self.vector_repository.search_batch(
            vectors=vectors,
            lang=search_settings.lang.value,
            excluded_tags=excluded_tags,
        )
        total = sum(len(r) for r in results)
        logger.info(f"[T1] step 3 — search_batch returned {total} point(s) across {len(results)} vector(s)")

        # Step 4: extract variants by mood and gender, group by original
        mood_key = search_settings.mood.value
        output = self._t1_extract_variants(results, mood_key=mood_key, gender=gender)
        logger.info(f"[T1] step 4 — {len(output)} unique original(s) in result")
        return output

    @log_decorator(level=logging.DEBUG)
    async def _t1_get_phrases(self, image_raw: bytes, lang: str) -> tuple[str, list[str]]:
        """Send image to Mistral vision and return detected gender and up to 3 observation phrases

        :param:
            image_raw: raw image bytes
            lang: target language code ('ru' or 'en')

        :returns:
            gender: 'male' or 'female' (defaults to 'male' if undetermined)
            phrases: list of up to 3 observation phrase strings
        """
        if not self.ai_client.supports_vision:
            logger.error("[T1, step 1] ai_client does not support vision")
            return "male", []
        try:
            prompt = PromptService.get("t1_vision", lang)
        except Exception as exc:
            logger.error("[T1, step 1] no prompt for lang=%s", lang, exc_info=exc)
            return "male", []
        image_b64 = base64.b64encode(image_raw).decode()
        raw = await self.ai_client.vision_chat(images_b64=[image_b64], prompt=prompt)
        if not raw:
            logger.warning("[T1, step 1] vision returned empty response")
            return "male", []
        gender, phrases = self._parse_vision_phrases(raw)
        logger.info(f"[T1, step 1] gender={gender}, parsed {len(phrases)} phrase(s)")
        return gender, phrases[:3]

    @staticmethod
    def _t1_extract_variants(
        results: list[list[ScoredPoint]],
        mood_key: str,
        gender: str,
    ) -> dict[str, dict]:
        """Extract mood+gender variants from Qdrant search results, grouped by original phrase

        :param:
            results: list of ScoredPoint lists from search_batch
            mood_key: mood letter key ('A'–'E')
            gender: 'male' or 'female'

        :returns:
            output: dict of {original: {tag: str, phrases: list[str]}}
        """
        output: dict[str, dict] = {}
        for points in results:
            for point in points:
                if not point.payload:
                    continue
                original = point.payload.get("original")
                tag = point.payload.get("tag")
                variants = point.payload.get("variants", {})
                if not original or original in output:
                    continue
                phrases = variants.get(mood_key, {}).get(gender, [])
                if phrases:
                    output[original] = {"tag": tag, "phrases": phrases}
        return output

    @log_decorator(level=logging.DEBUG)
    async def _t1_embed_phrases(self, phrases: list[str]) -> list[list[float]]:
        """Embed observation phrases using Mistral search_query strategy

        :param:
            phrases: list of observation phrase strings

        :returns:
            vectors: list of embedding vectors, one per phrase (failed phrases are skipped)
        """
        if not self.ai_client.supports_embed:
            logger.error("[T1, step 2] ai_client does not support embed")
            return []
        result = await self.ai_client.embed(phrases, task_type="query")
        if not result:
            logger.warning("[T1, step 2] embed returned empty result")
            return []
        vectors = result if isinstance(result, list) and isinstance(result[0], list) else [result]
        return vectors

    @staticmethod
    def _parse_vision_phrases(raw: str) -> tuple[str, list[str]]:
        """Parse raw vision output as JSON object with gender and phrases

        :param:
            raw: raw text response from the vision model

        :returns:
            gender: 'male' or 'female' (defaults to 'male' on parse failure)
            phrases: list of phrase strings
        """
        import json
        text = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        try:
            data = json.loads(text)
            gender = data.get("gender", "male")
            if gender not in ("male", "female"):
                gender = "male"
            phrases = [str(p).strip() for p in data.get("phrases", []) if p]
            return gender, phrases
        except Exception as exc:
            logger.error("[T1, step 1] failed to parse vision output: %s", text[:300], exc_info=exc)
        return "male", []
