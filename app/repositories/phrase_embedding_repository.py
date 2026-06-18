import logging

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.common.logging import log_decorator
from app.models.phrase_embeddings import PhraseEmbedding
from app.repositories.base_repository import BaseRepository
from app.repositories.repository_error_handler import repository_error_handler


@repository_error_handler
class PhraseEmbeddingRepository(BaseRepository):
    """Repository for creating and updating PhraseEmbedding vector records"""

    @log_decorator(level=logging.DEBUG)
    async def bulk_upsert_embeddings(self, rows: list[dict]) -> None:
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
