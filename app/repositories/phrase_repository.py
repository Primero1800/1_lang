import logging

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.logging import log_decorator
from app.models.phrases import Phrase
from app.repositories.repository_error_handler import repository_error_handler


@repository_error_handler
class PhraseRepository:
    _session: AsyncSession

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @log_decorator(level=logging.DEBUG)
    async def bulk_create(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        stmt = (
            pg_insert(Phrase)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_phrases_original_lang")
        )
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore
