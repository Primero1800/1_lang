import abc
import logging
from typing import Any, Literal, TypedDict

import aiohttp
from aiohttp import ClientSession

from app.common.logging import log_decorator, logger
from app.core.aiohttp_exception_handler import external_request_exception_handler
from app.core.config import settings


class Message(TypedDict):
    """Single chat message with a role and text content"""

    role: Literal["system", "user", "assistant"]
    content: str


class AIClientAbstract(abc.ABC):
    """Abstract base class for AI client implementations"""

    @abc.abstractmethod
    async def start(self) -> None:
        pass

    @abc.abstractmethod
    async def stop(self) -> None:
        pass

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> str | None:
        pass

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> str | None:
        pass

    @abc.abstractmethod
    async def embed(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float] | None:
        pass


class MistralClient(AIClientAbstract):
    """Mistral AI client using the shared aiohttp session"""

    _BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, aiohttp_session: aiohttp.ClientSession) -> None:
        self.session: ClientSession = aiohttp_session
        self._model: str = settings.MISTRAL_MODEL
        self._embed_model: str = settings.MISTRAL_EMBED_MODEL
        self._timeout = aiohttp.ClientTimeout(total=settings.MISTRAL_TIMEOUT_SEC)
        self._headers = {
            "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }

    async def start(self) -> None:
        """No-op: session is injected via constructor"""

    async def stop(self) -> None:
        self.session = None  # type: ignore[assignment]

    @log_decorator(level=logging.DEBUG)
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> str | None:
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(messages=messages, model=model, temperature=temperature)

    @log_decorator(level=logging.DEBUG)
    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> str | None:
        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        result = await self.__post("/chat/completions", payload)
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected error while ai_client request", exc_info=exc)
            return None

    @log_decorator(level=logging.DEBUG)
    async def embed(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float] | None:
        payload: dict[str, Any] = {
            "model": model or self._embed_model,
            "input": [text],
        }
        result = await self.__post("/embeddings", payload)
        try:
            return result["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected error while ai_client embed", exc_info=exc)
            return None

    @external_request_exception_handler(is_raise=False)
    async def __post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self._BASE_URL}{path}"
        async with self.session.post(
            url, json=payload, headers=self._headers, timeout=self._timeout
        ) as response:
            response.raise_for_status()
            return await response.json()
