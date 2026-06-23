import asyncio
import base64
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr

from app.common.enums import PhraseStatusEnum
from app.common.exceptions import VisionPipelineException
from app.common.logging import log_decorator, logger
from app.core.config import settings
from app.pyd.ai_schemas import VisionOutput
from app.services.base import BaseService
from app.services.prompt_service import PromptService

_VISION_MODEL = "pixtral-12b-2409"
_vision_llm = ChatMistralAI(
    model_name=_VISION_MODEL,
    api_key=SecretStr(settings.MISTRAL_API_KEY),
    timeout=settings.MISTRAL_VISION_TIMEOUT_SEC,
)


class PhraseService(BaseService):
    """Service for processing images through the vision pipeline and persisting phrases"""

    _llm = _vision_llm

    @log_decorator(level=logging.INFO)
    async def _encode_images(self, data: dict) -> dict:
        """Encode raw image bytes to base64

        :param:
            data: dict with 'images_raw' (list[bytes]) and 'lang' (str)

        :returns:
            dict with 'images_b64' (list[str]) and 'lang'
        """
        images_b64 = [base64.b64encode(img).decode() for img in data["images_raw"]]
        return {"images_b64": images_b64, "lang": data["lang"]}

    @log_decorator(level=logging.INFO)
    async def _build_vision_message(self, data: dict) -> list[HumanMessage]:
        """Resolve the vision prompt and build a multimodal HumanMessage

        :param:
            data: dict with 'images_b64' (list[str]) and 'lang' (str)

        :returns:
            messages: single-element list ready for the vision LLM
        """
        prompt = PromptService.get("pixtral_vision", data["lang"])
        content: list[str | dict[Any, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in data["images_b64"]
        ]
        content.append({"type": "text", "text": prompt})
        return [HumanMessage(content=content)]

    @log_decorator(level=logging.INFO)
    async def _fire_token_task(self, data: dict) -> VisionOutput:
        """Publish token usage to Redis Streams and return the parsed VisionOutput

        :param:
            data: dict with 'raw' (AIMessage) and 'parsed' (VisionOutput)

        :returns:
            parsed: the structured VisionOutput from the LLM
        """
        parsed: VisionOutput | None = data.get("parsed")
        if parsed is None:
            raise VisionPipelineException(
                detail="LLM returned invalid structured output"
            )
        usage = (data["raw"].usage_metadata or {}) if data.get("raw") else {}
        asyncio.create_task(
            self.queue_client.xadd(
                settings.REDIS_TOKENS_STREAM,
                {
                    "model": _VISION_MODEL,
                    "operation": "w1_vision",
                    "input_tokens": str(usage.get("input_tokens", 0)),
                    "output_tokens": str(usage.get("output_tokens", 0)),
                },
            )
        )
        return parsed

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
            cleaned = re.sub(r"[^\w\s\-]", "", item.phrase, flags=re.UNICODE)
            original = " ".join(cleaned.split()).lower()
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
        """Run the full W1 vision chain: prepare → build message → LLM → track tokens → build rows → save

        :param:
            images_raw: list of raw image bytes from the upload
            lang: target language code for the phrases

        :returns:
            result: dict with 'phrases_found', 'inserted', and 'skipped' counts
        """
        # 1. Attach structured output schema to the LLM
        llm = self._llm.with_structured_output(VisionOutput, include_raw=True)

        # 2. Bind the target language into the _build_rows step
        async def _build_rows_for_lang(vo: VisionOutput) -> list:
            return await self._build_rows(vo, lang)

        # 3. Assemble the full LangChain processing pipeline
        chain = (
            RunnableLambda(self._encode_images)
            | RunnableLambda(self._build_vision_message)
            | llm
            | RunnableLambda(self._fire_token_task)
            | RunnableLambda(_build_rows_for_lang)
            | RunnableLambda(self._save_phrases)
        )

        # 4. Invoke the chain; map unexpected errors to VisionPipelineException
        try:
            return await chain.ainvoke({"images_raw": images_raw, "lang": lang})
        except VisionPipelineException:
            raise
        except Exception as exc:
            logger.error("[W1] vision pipeline failed: %s", exc, exc_info=exc)
            raise VisionPipelineException(detail=str(exc)) from exc
