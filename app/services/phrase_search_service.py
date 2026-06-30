import base64
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_mistralai import ChatMistralAI
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

_T1_VISION_MODEL = settings.MISTRAL_VISION_MODEL
_t1_vision_llm = ChatMistralAI(
    model_name=_T1_VISION_MODEL,
    api_key=SecretStr(settings.MISTRAL_API_KEY),
    timeout=settings.MISTRAL_VISION_TIMEOUT_SEC,
)

_T1_EMBED_MODEL = settings.MISTRAL_EMBED_MODEL
_t1_embeddings = TrackedMistralEmbeddings(
    model=_T1_EMBED_MODEL,
    api_key=SecretStr(settings.MISTRAL_API_KEY),
)


class PhraseSearchService(BaseService):
    """T1 search service: vision → embed → Qdrant search pipeline"""

    _vision_llm = _t1_vision_llm
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

        # 1. Closure: build vision message from captured image_raw, lang, allowed_tags
        async def _build_message(_: dict) -> list[HumanMessage]:
            prompt = PromptService.get_t1_vision_prompt(
                lang=lang, allowed_tags=allowed_tags
            )
            image_b64 = base64.b64encode(image_raw).decode()
            return [
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ]
                )
            ]

        # 2. Closure: parse vision response, queue tokens, pass context forward
        async def _process_vision(response: Any) -> dict[str, Any]:
            usage = response.usage_metadata or {}
            self._queue_token_usage(
                model=_T1_VISION_MODEL,
                operation=self._VISION_OPERATION,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
            raw = response.content if response.content else ""
            if not raw:
                logger.warning("[T1, step 1] vision returned empty response")
                return {
                    "gender": "male",
                    "tag_phrases": {},
                    "lang": lang,
                    "mood_key": mood_key,
                }
            gender, tag_phrases = self._parse_vision_phrases(raw)
            logger.info(
                f"[T1] step 1 — gender={gender}, extracted {len(tag_phrases)} phrase(s): {tag_phrases}"
            )
            return {
                "gender": gender,
                "tag_phrases": tag_phrases,
                "lang": lang,
                "mood_key": mood_key,
            }

        # 3. Assemble full pipeline chain
        chain = (
            RunnableLambda(_build_message)
            | self._vision_llm.with_config(
                run_name=self._VISION_OPERATION,
                metadata={"ls_hide_inputs": True},
            )
            | RunnableLambda(_process_vision)
            | RunnableLambda(self._t1_embed_phrases)
            | RunnableLambda(self._t1_search_extract)
        ).with_config(run_name=self._OPERATION)

        # 4. Invoke chain; wrap unexpected errors in T1PipelineException
        try:
            return await chain.ainvoke({"lang": lang, "mood_key": mood_key})
        except Exception as exc:
            if not isinstance(exc, T1PipelineException):
                logger.error("[T1] pipeline failed: %s", exc, exc_info=exc)
            if isinstance(exc, T1PipelineException):
                raise
            raise T1PipelineException(detail=str(exc)) from exc

    @log_decorator(level=logging.DEBUG)
    async def _t1_embed_phrases(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Embed tagged observation phrases using search_query prefix strategy

        :param:
            inputs: dict with 'gender', 'tag_phrases', 'lang', 'mood_key'

        :returns:
            context: dict with 'gender', 'tag_vectors', 'lang', 'mood_key'
                     tag_vectors is empty if tag_phrases was empty
        """
        tag_phrases: dict[str, str] = inputs["tag_phrases"]
        if not tag_phrases:
            return {
                "gender": inputs["gender"],
                "tag_vectors": {},
                "lang": inputs["lang"],
                "mood_key": inputs["mood_key"],
            }
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
        return {
            "gender": inputs["gender"],
            "tag_vectors": dict(zip(tags, vectors)),
            "lang": inputs["lang"],
            "mood_key": inputs["mood_key"],
        }

    @log_decorator(level=logging.DEBUG)
    async def _t1_search_extract(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Search Qdrant for matching phrases and filter by mood and gender

        :param:
            inputs: dict with 'gender', 'tag_vectors', 'lang', 'mood_key'

        :returns:
            output: dict of {original: {tag: str, phrases: list[str]}}, empty if no vectors
        """
        tag_vectors: dict[str, list[float]] = inputs["tag_vectors"]
        if not tag_vectors:
            return {}
        tags = list(tag_vectors.keys())
        vectors = list(tag_vectors.values())
        results = await self.vector_repository.search_batch(
            vectors=vectors, tags=tags, lang=inputs["lang"]
        )
        total = sum(len(r) for r in results)
        logger.info(
            f"[T1] step 3 — search_batch returned {total} point(s) across {len(results)} vector(s)"
        )
        output = self._t1_extract_variants(
            results, mood_key=inputs["mood_key"], gender=inputs["gender"]
        )
        logger.info(f"[T1] step 4 — {len(output)} unique original(s) in result")
        return output

    @staticmethod
    def _t1_extract_variants(
        results: list[list[ScoredPoint]],
        mood_key: str,
        gender: str,
    ) -> dict[str, dict[str, Any]]:
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
