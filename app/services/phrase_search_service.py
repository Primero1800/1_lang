import base64
import json
import logging
import re
from typing import Any

from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr
from qdrant_client.models import ScoredPoint

from app.adapters.embeddings_client import TrackedMistralEmbeddings
from app.common.exceptions import T1PipelineException
from app.common.logging import log_decorator, logger
from app.core.config import settings
from app.pyd.requests import SearchSettings, TagExclusionFilters
from app.repositories.phrase_vector_repository import PhraseVectorRepository
from app.services.base import BaseDeps, BaseService
from app.services.prompt_service import PromptService
from app.uow import UnitOfWork

_T1_EMBED_MODEL = settings.MISTRAL_EMBED_MODEL
_t1_embeddings = TrackedMistralEmbeddings(
    model=_T1_EMBED_MODEL,
    api_key=SecretStr(settings.MISTRAL_API_KEY),
)


class PhraseSearchService(BaseService):
    """T1 search test service: vision → embed → Qdrant search pipeline"""

    _t1_embeddings = _t1_embeddings
    _OPERATION = "t1_search"
    _EMBED_OPERATION = "t1_embed"
    _VISION_OPERATION = "t1_vision"

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
    ) -> dict[str, Any]:
        """Orchestrate the full T1 pipeline: vision → embed → search → filter by mood and gender

        :param:
            image_raw: raw bytes of the uploaded image
            filters: tag exclusion flags
            search_settings: target language and mood tone

        :returns:
            result: dict of {original: {tag: str, phrases: list[str]}}
        """
        lang = search_settings.lang.value
        mood_key = search_settings.mood.value
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

        # Step 1: vision — outside chain (uses ai_client, not structured output)
        gender, tag_phrases = await self._t1_get_phrases(
            image_raw, lang=lang, allowed_tags=allowed_tags
        )
        logger.info(
            f"[T1] step 1 — gender={gender}, extracted {len(tag_phrases)} phrase(s): {tag_phrases}"
        )
        if not tag_phrases:
            return {}

        # 1. Bind search context into the last chain step
        async def _search_and_extract(
            tag_vectors: dict[str, list[float]],
        ) -> dict[str, Any]:
            if not tag_vectors:
                return {}
            tags = list(tag_vectors.keys())
            vectors = list(tag_vectors.values())
            results = await self.vector_repository.search_batch(
                vectors=vectors, tags=tags, lang=lang
            )
            total = sum(len(r) for r in results)
            logger.info(
                f"[T1] step 3 — search_batch returned {total} point(s) across {len(results)} vector(s)"
            )
            output = self._t1_extract_variants(
                results, mood_key=mood_key, gender=gender
            )
            logger.info(f"[T1] step 4 — {len(output)} unique original(s) in result")
            return output

        # 2. Assemble chain: embed → search + extract
        chain = (
            RunnableLambda(self._t1_embed_phrases) | RunnableLambda(_search_and_extract)
        ).with_config(run_name=self._OPERATION)

        # 3. Invoke chain; wrap unexpected errors in T1PipelineException
        try:
            return await chain.ainvoke(tag_phrases)
        except Exception as exc:
            if not isinstance(exc, T1PipelineException):
                logger.error("[T1] pipeline failed: %s", exc, exc_info=exc)
            if isinstance(exc, T1PipelineException):
                raise
            raise T1PipelineException(detail=str(exc)) from exc

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
        # 1. Skip if vision model is unavailable
        if not self.ai_client.supports_vision:
            logger.error("[T1, step 1] ai_client does not support vision")
            return "male", {}
        # 2. Build the vision prompt for the allowed tags
        prompt = PromptService.get_t1_vision_prompt(
            lang=lang, allowed_tags=allowed_tags
        )
        # 3. Encode image and call the vision model
        image_b64 = base64.b64encode(image_raw).decode()
        raw = await self.ai_client.vision_chat(
            images_b64=[image_b64], prompt=prompt, operation=self._VISION_OPERATION
        )
        if not raw:
            logger.warning("[T1, step 1] vision returned empty response")
            return "male", {}
        # 4. Parse gender and per-tag phrases from the vision response
        gender, tag_phrases = self._parse_vision_phrases(raw)
        logger.info(
            f"[T1, step 1] gender={gender}, parsed {len(tag_phrases)} phrase(s)"
        )
        return gender, tag_phrases

    @log_decorator(level=logging.DEBUG)
    async def _t1_embed_phrases(
        self, tag_phrases: dict[str, str]
    ) -> dict[str, list[float]]:
        """Embed tagged observation phrases using search_query prefix strategy

        :param:
            tag_phrases: dict of {tag: observation phrase}

        :returns:
            tag_vectors: dict of {tag: embedding vector}, preserving input order
        """
        tags = list(tag_phrases.keys())
        phrases = list(tag_phrases.values())
        vectors, input_tokens = await self._t1_embeddings.aembed_with_usage(
            phrases, input_type="search_query"
        )
        logger.info(f"[T1] step 2 — embedded {len(vectors)}/{len(phrases)} phrase(s)")
        self._queue_token_usage(
            model=_T1_EMBED_MODEL,
            operation=self._EMBED_OPERATION,
            input_tokens=input_tokens,
        )
        return dict(zip(tags, vectors))

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
                    output[original] = {
                        "tag": tag,
                        "score": round(point.score, 4),
                        "gender": gender,
                        "phrases": phrases,
                    }
        return output

    @staticmethod
    def _parse_vision_phrases(raw: str) -> tuple[str, dict[str, str]]:
        """Parse raw vision output as JSON object with gender and per-tag phrases

        :param:
            raw: raw text response from the vision model

        :returns:
            gender: 'male' or 'female' (defaults to 'male' on parse failure)
            tag_phrases: dict of {tag: phrase} parsed from the 'phrases' object
        """
        # 1. Strip markdown code fence if present
        text = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        try:
            # 2. Parse JSON payload
            data = json.loads(text)
            # 3. Normalise gender — default to 'male' on unknown value
            gender = data.get("gender", "male")
            if gender not in ("male", "female"):
                gender = "male"
            # 4. Extract and filter the per-tag phrase dict
            phrases = data.get("phrases", {})
            if isinstance(phrases, dict):
                result = {}
                for k, v in phrases.items():
                    if isinstance(v, list):
                        joined = " ".join(s.strip() for s in v if s)
                    else:
                        joined = str(v).strip()
                    if joined:
                        result[k] = joined
                return gender, result
        except Exception as exc:
            logger.error(
                "[T1, step 1] failed to parse vision output: %s",
                text[:300],
                exc_info=exc,
            )
        return "male", {}
