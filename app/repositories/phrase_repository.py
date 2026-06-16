import logging
from datetime import timedelta

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.common.enums import PhraseStatusEnum
from app.common.logging import log_decorator
from app.core.config import settings
from app.models.phrases import Phrase
from app.repositories.base_repository import BaseRepository
from app.repositories.repository_error_handler import repository_error_handler


@repository_error_handler
class PhraseRepository(BaseRepository):
    """Repository for creating and managing Phrase records in PostgreSQL"""

    @log_decorator(level=logging.DEBUG)
    async def bulk_create(self, rows: list[dict]) -> int:
        """Insert multiple phrase rows, silently skipping duplicates

        :param:
            rows: list of phrase attribute dicts (original, tag, lang, status)

        :returns:
            inserted_count: number of rows actually inserted (duplicates excluded)
        """
        if not rows:
            return 0
        stmt = (
            pg_insert(Phrase)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_phrases_original_lang")
        )
        result = await self._session.execute(stmt)
        return result.rowcount  # type: ignore

    @log_decorator(level=logging.DEBUG)
    async def get_first_for_generation(self) -> Phrase | None:
        """Return the highest-priority phrase eligible for generation, locked for update

        Priority: stuck GENERATING_IN_PROGRESS → GENERATING_FAILED → DRAFT.
        Uses FOR UPDATE SKIP LOCKED so concurrent workers skip already-locked rows.

        :returns:
            phrase: the selected Phrase, or None if no eligible rows exist
        """
        stuck = and_(
            Phrase.status == PhraseStatusEnum.GENERATING_IN_PROGRESS,
            Phrase.updated_at
            < func.now() - timedelta(minutes=settings.STUCK_THRESHOLD),
        )
        stmt = (
            select(Phrase)
            .where(
                or_(
                    Phrase.status.in_(
                        [
                            PhraseStatusEnum.GENERATING_FAILED,
                            PhraseStatusEnum.DRAFT,
                        ]
                    ),
                    stuck,
                )
            )
            .order_by(
                case(
                    (stuck, 0),
                    (Phrase.status == PhraseStatusEnum.GENERATING_FAILED, 1),
                    else_=2,
                ),
                Phrase.id,
            )
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    @log_decorator(level=logging.DEBUG)
    async def get_batch_for_generation(
        self, lang: str, exclude_id: int, limit: int = 6
    ) -> list[Phrase]:
        """Return additional phrases for a batch, filtered by lang and excluding one ID

        :param:
            lang: language code to match ('ru' or 'en')
            exclude_id: phrase ID already claimed as the batch leader
            limit: maximum number of additional phrases to return

        :returns:
            phrases: list of additional Phrase objects, locked for update
        """
        stuck = and_(
            Phrase.status == PhraseStatusEnum.GENERATING_IN_PROGRESS,
            Phrase.updated_at
            < func.now() - timedelta(minutes=settings.STUCK_THRESHOLD),
        )
        stmt = (
            select(Phrase)
            .where(
                or_(
                    Phrase.status.in_(
                        [
                            PhraseStatusEnum.GENERATING_FAILED,
                            PhraseStatusEnum.DRAFT,
                        ]
                    ),
                    stuck,
                ),
                Phrase.lang == lang,
                Phrase.id != exclude_id,
            )
            .order_by(
                case(
                    (stuck, 0),
                    (Phrase.status == PhraseStatusEnum.GENERATING_FAILED, 1),
                    else_=2,
                ),
                Phrase.id,
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    @log_decorator(level=logging.DEBUG)
    async def mark_generating_in_progress(self, ids: list[int]) -> None:
        """Set status to GENERATING_IN_PROGRESS for the given phrase IDs

        :param:
            ids: list of phrase IDs to update

        :returns:
            None
        """
        stmt = (
            update(Phrase)
            .where(Phrase.id.in_(ids))
            .values(
                status=PhraseStatusEnum.GENERATING_IN_PROGRESS, updated_at=func.now()
            )
        )
        await self._session.execute(stmt)

    @log_decorator(level=logging.DEBUG)
    async def mark_generating_done(self, ids: list[int]) -> None:
        """Set status to GENERATING_DONE for the given phrase IDs

        :param:
            ids: list of phrase IDs to update

        :returns:
            None
        """
        stmt = (
            update(Phrase)
            .where(Phrase.id.in_(ids))
            .values(status=PhraseStatusEnum.GENERATING_DONE, updated_at=func.now())
        )
        await self._session.execute(stmt)

    @log_decorator(level=logging.DEBUG)
    async def mark_generating_failed(self, ids: list[int]) -> None:
        """Set status to GENERATING_FAILED for the given phrase IDs

        :param:
            ids: list of phrase IDs to update

        :returns:
            None
        """
        stmt = (
            update(Phrase)
            .where(Phrase.id.in_(ids))
            .values(status=PhraseStatusEnum.GENERATING_FAILED, updated_at=func.now())
        )
        await self._session.execute(stmt)
