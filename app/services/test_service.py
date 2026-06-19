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
        lang = search_settings.lang.value
        all_tags = ["behavior", "appearance", "age", "mood", "posture", "hairstyle"]
        filter_map = {
            "behavior": filters.not_behavior,
            "appearance": filters.not_appearance,
            "age": filters.not_age,
            "mood": filters.not_mood,
            "posture": filters.not_posture,
            "hairstyle": filters.not_hairstyle,
        }
        allowed_tags = [tag for tag in all_tags if not filter_map[tag]]
        if not allowed_tags:
            logger.warning("[T1] all tags restricted, returning message")
            return {"message": PromptService.get_restricted_message(lang)}

        # Step 1: extract gender and one phrase per allowed tag from the image
        gender, tag_phrases = await self._t1_get_phrases(image_raw, lang=lang, allowed_tags=allowed_tags)
        logger.info(f"[T1] step 1 — gender={gender}, extracted {len(tag_phrases)} phrase(s): {tag_phrases}")
        if not tag_phrases:
            return {}

        # Step 2: embed each phrase with search_query prefix
        tag_vectors = await self._t1_embed_phrases(tag_phrases)
        logger.info(f"[T1] step 2 — embedded {len(tag_vectors)}/{len(tag_phrases)} phrase(s)")
        if not tag_vectors:
            return {}

        # Step 3: batch search — each vector searches within its own tag
        tags = list(tag_vectors.keys())
        vectors = list(tag_vectors.values())
        results = await self.vector_repository.search_batch(
            vectors=vectors,
            tags=tags,
            lang=lang,
        )
        total = sum(len(r) for r in results)
        logger.info(f"[T1] step 3 — search_batch returned {total} point(s) across {len(results)} vector(s)")

        # Step 4: extract variants by mood and gender, group by original
        mood_key = search_settings.mood.value
        output = self._t1_extract_variants(results, mood_key=mood_key, gender=gender)
        logger.info(f"[T1] step 4 — {len(output)} unique original(s) in result")
        return output

    @log_decorator(level=logging.DEBUG)
    async def _t1_get_phrases(
        self,
        image_raw: bytes,
        lang: str,
        allowed_tags: list[str],
    ) -> tuple[str, dict[str, str]]:
        """Send image to Mistral vision and return detected gender and one phrase per allowed tag

        :param:
            image_raw: raw image bytes
            lang: target language code ('ru' or 'en')
            allowed_tags: tag keys to request observations for

        :returns:
            gender: 'male' or 'female' (defaults to 'male' if undetermined)
            tag_phrases: dict of {tag: observation phrase}
        """
        if not self.ai_client.supports_vision:
            logger.error("[T1, step 1] ai_client does not support vision")
            return "male", {}
        prompt = PromptService.get_t1_vision_prompt(lang=lang, allowed_tags=allowed_tags)
        image_b64 = base64.b64encode(image_raw).decode()
        raw = await self.ai_client.vision_chat(images_b64=[image_b64], prompt=prompt)
        if not raw:
            logger.warning("[T1, step 1] vision returned empty response")
            return "male", {}
        gender, tag_phrases = self._parse_vision_phrases(raw)
        logger.info(f"[T1, step 1] gender={gender}, parsed {len(tag_phrases)} phrase(s)")
        return gender, tag_phrases

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
                    output[original] = {"tag": tag, "score": round(point.score, 4), "phrases": phrases}
        return output

    @log_decorator(level=logging.DEBUG)
    async def _t1_embed_phrases(self, tag_phrases: dict[str, str]) -> dict[str, list[float]]:
        """Embed tagged observation phrases using Mistral search_query strategy

        :param:
            tag_phrases: dict of {tag: observation phrase}

        :returns:
            tag_vectors: dict of {tag: embedding vector}, preserving input order
        """
        if not self.ai_client.supports_embed:
            logger.error("[T1, step 2] ai_client does not support embed")
            return {}
        tags = list(tag_phrases.keys())
        phrases = list(tag_phrases.values())
        result = await self.ai_client.embed(phrases, task_type="query")
        if not result:
            logger.warning("[T1, step 2] embed returned empty result")
            return {}
        vectors = result if isinstance(result, list) and isinstance(result[0], list) else [result]
        return dict(zip(tags, vectors))

    @staticmethod
    def _parse_vision_phrases(raw: str) -> tuple[str, dict[str, str]]:
        """Parse raw vision output as JSON object with gender and per-tag phrases

        :param:
            raw: raw text response from the vision model

        :returns:
            gender: 'male' or 'female' (defaults to 'male' on parse failure)
            tag_phrases: dict of {tag: phrase} parsed from the 'phrases' object
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
            phrases = data.get("phrases", {})
            if isinstance(phrases, dict):
                return gender, {k: str(v).strip() for k, v in phrases.items() if v}
        except Exception as exc:
            logger.error("[T1, step 1] failed to parse vision output: %s", text[:300], exc_info=exc)
        return "male", {}
