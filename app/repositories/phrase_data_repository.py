import logging

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.common.logging import log_decorator
from app.models.phrase_data import PhraseData
from app.repositories.base_repository import BaseRepository
from app.repositories.repository_error_handler import repository_error_handler


@repository_error_handler
class PhraseDataRepository(BaseRepository):
    """Repository for creating and updating PhraseData variant records"""

    @log_decorator(level=logging.DEBUG)
    async def bulk_upsert_variants(self, rows: list[dict]) -> None:
        """Insert or update phrase_data rows by phrase_id

        :param:
            rows: list of dicts with 'phrase_id' and 'variants' keys

        :returns:
            None
        """
        if not rows:
            return
        stmt = (
            pg_insert(PhraseData)
            .values(rows)
            .on_conflict_do_update(
                index_elements=["phrase_id"],
                set_={
                    "variants": pg_insert(PhraseData).excluded.variants,
                    "updated_at": func.now(),
                },
            )
        )
        await self._session.execute(stmt)
