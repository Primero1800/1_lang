import logging

from app.common.logging import log_decorator
from app.services.base import BaseService


class PhraseTranslationService(BaseService):
    @log_decorator(level=logging.INFO)
    async def w3_translate(self, batch_size: int) -> dict[str, int]:
        raise NotImplementedError
