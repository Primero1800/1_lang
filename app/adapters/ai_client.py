import abc
import asyncio
import logging
from typing import Any, Literal, TypedDict

import aiohttp
from aiohttp import ClientSession

from app.common.logging import log_decorator, logger
from app.core.aiohttp_exception_handler import external_request_exception_handler
from app.core.config import settings
from app.utils.token_utils import record_token_usage


class Message(TypedDict):
    """Single chat message with a role and text content"""

    role: Literal["system", "user", "assistant"]
    content: str


class VisionMessage(TypedDict):
    """Chat message with multimodal content (text + base64 images)"""

    role: Literal["user", "assistant"]
    content: list[dict[str, Any]]


class AIClientAbstract(abc.ABC):
    """Abstract base class for AI client implementations"""

    supports_vision: bool = False
    supports_embed: bool = False
    supports_chat: bool = True

    @abc.abstractmethod
    async def start(self) -> None:
        """Start the client and initialise any internal resources

        :returns:
            None
        """

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop the client and release any internal resources

        :returns:
            None
        """

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
        operation: str | None = None,
    ) -> str | None:
        """Generate a completion from a single user prompt

        :param:
            prompt: the user message text
            system: optional system instruction
            model: model identifier override
            temperature: sampling temperature override
            options: additional provider-specific payload fields

        :returns:
            text: generated text, or None on failure
        """

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
        operation: str | None = None,
    ) -> str | None:
        """Send a multi-turn chat request

        :param:
            messages: ordered list of role/content message dicts
            model: model identifier override
            temperature: sampling temperature override
            options: additional provider-specific payload fields

        :returns:
            text: assistant reply text, or None on failure
        """

    @abc.abstractmethod
    async def embed(
        self,
        text: str | list[str],
        model: str | None = None,
        task_type: Literal["query", "document"] | None = None,
        operation: str | None = None,
    ) -> list[float] | list[list[float]] | None:
        """Compute embeddings for one or more text inputs

        :param:
            text: a single string or list of strings to embed
            model: embedding model identifier override
            task_type: 'query' or 'document' prefix strategy

        :returns:
            embedding: a single vector (str input) or list of vectors (list input), or None on failure
        """

    @abc.abstractmethod
    async def vision_chat(
        self,
        images_b64: list[str],
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        operation: str | None = None,
    ) -> str | None:
        """Send images and a text prompt to a vision model

        :param:
            images_b64: list of base64-encoded image strings
            prompt: instruction text accompanying the images
            model: vision model identifier override
            temperature: sampling temperature override

        :returns:
            text: model response text, or None on failure
        """


class MistralClient(AIClientAbstract):
    """Mistral AI client using the shared aiohttp session"""

    supports_vision = True
    supports_embed = True

    _BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, aiohttp_session: aiohttp.ClientSession) -> None:
        """Initialize the Mistral client with a shared aiohttp session

        :param:
            aiohttp_session: the shared aiohttp ClientSession

        :returns:
            None
        """
        self.session: ClientSession = aiohttp_session
        self._model: str = settings.MISTRAL_MODEL
        self._embed_model: str = settings.MISTRAL_EMBED_MODEL
        self._timeout = aiohttp.ClientTimeout(total=settings.MISTRAL_TIMEOUT_SEC)
        self._vision_timeout = aiohttp.ClientTimeout(
            total=settings.MISTRAL_VISION_TIMEOUT_SEC
        )
        self._headers = {
            "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }

    async def start(self) -> None:
        """No-op: session is injected via constructor

        :returns:
            None
        """

    async def stop(self) -> None:
        """Release the aiohttp session reference

        :returns:
            None
        """
        self.session = None  # type: ignore[assignment]

    @staticmethod
    def _fire_token_task(
        result: Any,
        model: str,
        operation: str | None,
        is_embed: bool = False,
    ) -> None:
        if not result or not operation:
            return
        usage = result.get("usage", {})
        asyncio.create_task(
            record_token_usage(
                model=model,
                operation=operation,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=0 if is_embed else usage.get("completion_tokens", 0),
            )
        )

    @log_decorator(level=logging.DEBUG)
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
        operation: str | None = None,
    ) -> str | None:
        """Build a single-turn message list and delegate to chat

        :param:
            prompt: the user message text
            system: optional system instruction prepended to messages
            model: model identifier override
            temperature: sampling temperature override
            options: additional payload fields passed to chat
            operation: pipeline operation name for token tracking

        :returns:
            text: generated text, or None on failure
        """
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(
            messages=messages, model=model, temperature=temperature, operation=operation
        )

    @log_decorator(level=logging.DEBUG)
    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
        operation: str | None = None,
    ) -> str | None:
        """Send a multi-turn chat request to the Mistral API

        :param:
            messages: ordered list of role/content message dicts
            model: model identifier override
            temperature: sampling temperature override
            options: additional payload fields merged into request body
            operation: pipeline operation name for token tracking

        :returns:
            text: assistant reply text, or None on failure
        """
        used_model = model or self._model
        payload: dict[str, Any] = {"model": used_model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if options:
            payload.update(options)
        result = await self.__post("/chat/completions", payload)
        self._fire_token_task(result, used_model, operation)
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected error while ai_client request", exc_info=exc)
            return None

    @log_decorator(level=logging.DEBUG)
    async def embed(
        self,
        text: str | list[str],
        model: str | None = None,
        task_type: Literal["query", "document"] | None = None,
        operation: str | None = None,
    ) -> list[float] | list[list[float]] | None:
        """Compute Mistral embeddings for one or more text inputs

        :param:
            text: a single string or list of strings to embed
            model: embedding model identifier override
            task_type: optional 'query' or 'document' prefix strategy
            operation: pipeline operation name for token tracking

        :returns:
            embedding: a single vector (str input) or list of vectors (list input), or None on failure
        """
        is_single_string = isinstance(text, str)
        input_data = [text] if is_single_string else text
        if task_type == "query":
            input_data = [f"search_query: {t}" for t in input_data]
        elif task_type == "document":
            input_data = [f"search_document: {t}" for t in input_data]
        used_model = model or self._embed_model
        payload: dict[str, Any] = {"model": used_model, "input": input_data}
        result = await self.__post("/embeddings", payload)
        self._fire_token_task(result, used_model, operation, is_embed=True)
        try:
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings[0] if is_single_string else embeddings
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected error while ai_client embed", exc_info=exc)
            return None

    @log_decorator(level=logging.DEBUG)
    async def vision_chat(
        self,
        images_b64: list[str],
        prompt: str,
        model: str = "pixtral-12b-2409",
        temperature: float | None = None,
        operation: str | None = None,
    ) -> str | None:
        """Send images and a text prompt to the Pixtral vision model

        :param:
            images_b64: list of base64-encoded JPEG image strings
            prompt: instruction text accompanying the images
            model: vision model identifier (default pixtral-12b-2409)
            temperature: sampling temperature override
            operation: pipeline operation name for token tracking

        :returns:
            text: model response text, or None on failure
        """
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in images_b64
        ]
        content.append({"type": "text", "text": prompt})
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        result = await self.__post(
            "/chat/completions", payload, timeout=self._vision_timeout
        )
        self._fire_token_task(result, model, operation)
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected error while mistral vision_chat", exc_info=exc)
            return None

    @external_request_exception_handler(is_raise=False)
    async def __post(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> Any:
        url = f"{self._BASE_URL}{path}"
        async with self.session.post(
            url,
            json=payload,
            headers=self._headers,
            timeout=timeout or self._timeout,
        ) as response:
            response.raise_for_status()
            return await response.json()


class GroqClient(AIClientAbstract):
    """Groq AI client — LLM inference and vision analysis, no embeddings"""

    supports_vision = True
    supports_embed = False

    _BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, aiohttp_session: aiohttp.ClientSession) -> None:
        """Initialize the Groq client with a shared aiohttp session

        :param:
            aiohttp_session: the shared aiohttp ClientSession

        :returns:
            None
        """
        self.session: ClientSession = aiohttp_session
        self._model: str = settings.GROQ_MODEL
        self._vision_model: str = settings.GROQ_VISION_MODEL
        self._timeout = aiohttp.ClientTimeout(total=settings.GROQ_TIMEOUT_SEC)
        self._headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

    async def start(self) -> None:
        """No-op: session is injected via constructor

        :returns:
            None
        """

    async def stop(self) -> None:
        """Release the aiohttp session reference

        :returns:
            None
        """
        self.session = None  # type: ignore[assignment]

    @staticmethod
    def _fire_token_task(
        result: Any,
        model: str,
        operation: str | None,
        is_embed: bool = False,
    ) -> None:
        if not result or not operation:
            return
        usage = result.get("usage", {})
        asyncio.create_task(
            record_token_usage(
                model=model,
                operation=operation,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=0 if is_embed else usage.get("completion_tokens", 0),
            )
        )

    @log_decorator(level=logging.DEBUG)
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
        operation: str | None = None,
    ) -> str | None:
        """Build a single-turn message list and delegate to chat

        :param:
            prompt: the user message text
            system: optional system instruction prepended to messages
            model: model identifier override
            temperature: sampling temperature override
            options: additional payload fields passed to chat
            operation: pipeline operation name for token tracking

        :returns:
            text: generated text, or None on failure
        """
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(
            messages=messages, model=model, temperature=temperature, operation=operation
        )

    @log_decorator(level=logging.DEBUG)
    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
        operation: str | None = None,
    ) -> str | None:
        """Send a multi-turn chat request to the Groq API

        :param:
            messages: ordered list of role/content message dicts
            model: model identifier override
            temperature: sampling temperature override
            options: additional payload fields merged into request body
            operation: pipeline operation name for token tracking

        :returns:
            text: assistant reply text, or None on failure
        """
        used_model = model or self._model
        payload: dict[str, Any] = {"model": used_model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if options:
            payload.update(options)
        result = await self.__post("/chat/completions", payload)
        self._fire_token_task(result, used_model, operation)
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected error while groq_client chat", exc_info=exc)
            return None

    @log_decorator(level=logging.DEBUG)
    async def vision_chat(
        self,
        images_b64: list[str],
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        operation: str | None = None,
    ) -> str | None:
        """Send images and a text prompt to the Groq vision model

        :param:
            images_b64: list of base64-encoded JPEG image strings
            prompt: instruction text accompanying the images
            model: vision model identifier override
            temperature: sampling temperature override
            operation: pipeline operation name for token tracking

        :returns:
            text: model response text, or None on failure
        """
        used_model = model or self._vision_model
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in images_b64
        ]
        content.append({"type": "text", "text": prompt})
        messages: list[VisionMessage] = [{"role": "user", "content": content}]
        payload: dict[str, Any] = {"model": used_model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        result = await self.__post("/chat/completions", payload)
        self._fire_token_task(result, used_model, operation)
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected error while groq_client vision_chat", exc_info=exc)
            return None

    async def embed(
        self,
        text: str | list[str],
        model: str | None = None,
        task_type: Literal["query", "document"] | None = None,
        operation: str | None = None,
    ) -> list[float] | list[list[float]] | None:
        """Not implemented: Groq does not support embeddings

        :param:
            text: text input (unused)
            model: model identifier (unused)
            task_type: task type hint (unused)

        :raise:
            NotImplementedError: always

        :returns:
            None
        """
        return None

    @external_request_exception_handler(is_raise=False)
    async def __post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self._BASE_URL}{path}"
        retries = 3
        max_retry_wait = 60.0
        for attempt in range(retries + 1):
            async with self.session.post(
                url, json=payload, headers=self._headers, timeout=self._timeout
            ) as response:
                if response.status == 429 and attempt < retries:
                    retry_after = float(response.headers.get("Retry-After", 10))
                    if retry_after > max_retry_wait:
                        logger.warning(
                            f"[Groq] Rate limited, Retry-After={retry_after:.0f}s exceeds limit — giving up"
                        )
                        return None
                    logger.warning(f"[Groq] Rate limited, retry in {retry_after:.1f}s")
                    await asyncio.sleep(retry_after)
                    continue
                response.raise_for_status()
                return await response.json()
