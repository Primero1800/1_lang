import logging

from sqlalchemy import func, select
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
        insert_stmt = pg_insert(PhraseData).values(rows)
        stmt = insert_stmt.on_conflict_do_update(
            index_elements=["phrase_id"],
            set_={
                "variants": insert_stmt.excluded.variants,
                "updated_at": func.now(),
            },
        )
        await self._session.execute(stmt)

    @log_decorator(level=logging.DEBUG)
    async def get_by_phrase_ids(self, phrase_ids: list[int]) -> list[PhraseData]:
        """Return PhraseData records for the given phrase IDs

        :param:
            phrase_ids: list of phrase IDs to fetch variants for

        :returns:
            records: list of PhraseData objects
        """
        if not phrase_ids:
            return []
        stmt = select(PhraseData).where(PhraseData.phrase_id.in_(phrase_ids))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
