import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.common.logging import log_decorator
from app.models.phrase_embeddings import PhraseEmbedding
from app.repositories.base_repository import BaseRepository
from app.repositories.repository_error_handler import repository_error_handler


@repository_error_handler
class PhraseEmbeddingRepository(BaseRepository):
    """Repository for creating and updating PhraseEmbedding vector records"""

    @log_decorator(level=logging.DEBUG)
    async def bulk_upsert_embeddings(self, rows: list[dict[str, Any]]) -> None:
        """Insert or update phrase_embeddings rows by phrase_id

        :param:
            rows: list of dicts with 'phrase_id' and 'embedding' keys

        :returns:
            None
        """
        if not rows:
            return
        insert_stmt = pg_insert(PhraseEmbedding).values(rows)
        stmt = insert_stmt.on_conflict_do_update(
            index_elements=["phrase_id"],
            set_={
                "embedding": insert_stmt.excluded.embedding,
                "updated_at": func.now(),
            },
        )
        await self._session.execute(stmt)

    @log_decorator(level=logging.DEBUG)
    async def get_by_phrase_ids(self, phrase_ids: list[int]) -> list[PhraseEmbedding]:
        """Return PhraseEmbedding records for the given phrase IDs

        :param:
            phrase_ids: list of phrase IDs to fetch embeddings for

        :returns:
            records: list of PhraseEmbedding objects
        """
        if not phrase_ids:
            return []
        stmt = select(PhraseEmbedding).where(PhraseEmbedding.phrase_id.in_(phrase_ids))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
