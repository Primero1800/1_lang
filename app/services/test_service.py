import logging
from typing import Any

from app.common.logging import log_decorator
from app.services.base import BaseService


class TestService(BaseService):
    @log_decorator(level=logging.DEBUG)
    async def check(self, text: str) -> Any:

        embeddings = await self.ai_client.embed(text)
        return embeddings
