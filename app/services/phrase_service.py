import base64
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr

from app.common.enums import PhraseStatusEnum
from app.common.exceptions import VisionPipelineException
from app.common.logging import log_decorator, logger
from app.core.config import settings
from app.pyd.ai_schemas import VisionBatchOutput, VisionOutput
from app.services.base import BaseService
from app.services.prompt_service import PromptService

_VISION_MODEL = settings.MISTRAL_VISION_MODEL
_vision_llm = ChatMistralAI(
    model_name=_VISION_MODEL,
    api_key=SecretStr(settings.MISTRAL_API_KEY),
    timeout=settings.MISTRAL_VISION_TIMEOUT_SEC,
)


class PhraseService(BaseService):
    """Service for processing images through the vision pipeline and persisting phrases"""

    _llm = _vision_llm
    _OPERATION = "w1_vision"

    @log_decorator(level=logging.INFO)
    async def _fire_token_task(self, data: dict[str, Any]) -> VisionOutput:
        """Publish token usage to Redis Streams and return the parsed VisionOutput

        :param:
            data: dict with 'raw' (AIMessage) and 'parsed' (VisionOutput)

        :returns:
            parsed: the structured VisionOutput from the LLM
        """
        parsed: VisionBatchOutput | None = data.get("parsed")
        if parsed is None:
            raise VisionPipelineException(
                detail="LLM returned invalid structured output"
            )
        all_phrases = [phrase for photo in parsed.photos for phrase in photo.phrases]
        usage = (data["raw"].usage_metadata or {}) if data.get("raw") else {}
        self._queue_token_usage(
            model=_VISION_MODEL,
            operation=self._OPERATION,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        return VisionOutput(phrases=all_phrases)

    @log_decorator(level=logging.INFO)
    async def _build_rows(
        self, vision_output: VisionOutput, lang: str
    ) -> list[dict[str, Any]]:
        """Normalise VisionOutput phrases into DB-ready row dicts, deduplicating by original text

        :param:
            vision_output: structured output from the vision LLM
            lang: language code to assign to each row

        :returns:
            rows: list of dicts ready for bulk insert
        """
        seen: set[str] = set()
        rows = []
        for item in vision_output.phrases:
            original = f"{item.concrete}. {item.abstract}".strip()
            if original and original not in seen:
                seen.add(original)
                rows.append(
                    {
                        "original": original,
                        "tag": item.tag,
                        "lang": lang,
                        "status": PhraseStatusEnum.DRAFT,
                    }
                )
        return rows

    @log_decorator(level=logging.INFO)
    async def _save_phrases(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        """Persist phrase rows to the database

        :param:
            rows: list of DB-ready phrase row dicts

        :returns:
            result: dict with 'phrases_found', 'inserted', and 'skipped' counts
        """
        if not rows:
            raise VisionPipelineException(
                detail="No valid phrases extracted from images"
            )
        async with self.uow_factory as uow:
            ids = await uow.phrase_repository.bulk_create(rows)
        inserted = len(ids)
        return {
            "phrases_found": len(rows),
            "inserted": inserted,
            "skipped": len(rows) - inserted,
        }

    @log_decorator(level=logging.INFO)
    async def upload_images(self, images_raw: list[bytes], lang: str) -> dict[str, int]:
        """Run the W1 vision chain once per image, merge rows, then save

        :param:
            images_raw: list of raw image bytes from the upload
            lang: target language code for the phrases

        :returns:
            result: dict with 'phrases_found', 'inserted', and 'skipped' counts
        """
        images_b64 = [base64.b64encode(img).decode() for img in images_raw]
        prompt = PromptService.get("pixtral_vision", lang)

        async def _build_message(_: Any) -> list[HumanMessage]:
            content: list[str | dict[Any, Any]] = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
                for b64 in images_b64
            ]
            content.append({"type": "text", "text": prompt})
            return [HumanMessage(content=content)]

        async def _build_rows_for_lang(vo: VisionOutput) -> list[dict[str, Any]]:
            return await self._build_rows(vo, lang)

        llm = self._llm.with_structured_output(
            VisionBatchOutput, include_raw=True
        ).with_config(metadata={"ls_hide_inputs": True})

        chain = (
            RunnableLambda(_build_message)
            | llm
            | RunnableLambda(self._fire_token_task)
            | RunnableLambda(_build_rows_for_lang)
            | RunnableLambda(self._save_phrases)
        ).with_config(run_name=self._OPERATION)

        try:
            return await chain.ainvoke({"lang": lang})
        except VisionPipelineException:
            raise
        except Exception as exc:
            logger.error("[W1] vision pipeline failed: %s", exc, exc_info=exc)
            raise VisionPipelineException(detail=str(exc)) from exc
