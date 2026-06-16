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
    async def bulk_create(self, rows: list[dict]) -> list[int]:
        """Insert multiple phrase rows, silently skipping duplicates

        :param:
            rows: list of phrase attribute dicts (original, tag, lang, status)

        :returns:
            ids: list of IDs of actually inserted rows (duplicates excluded)
        """
        if not rows:
            return []
        stmt = (
            pg_insert(Phrase)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_phrases_original_lang")
            .returning(Phrase.id)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    @log_decorator(level=logging.DEBUG)
    async def get_first_for_processing(
        self,
        in_progress_status: PhraseStatusEnum,
        priority_status: PhraseStatusEnum,
        base_statuses: list[PhraseStatusEnum],
    ) -> Phrase | None:
        """Return the highest-priority phrase eligible for processing, locked for update

        Priority: stuck in_progress → priority_status → base_statuses.
        Uses FOR UPDATE SKIP LOCKED so concurrent workers skip already-locked rows.

        :param:
            in_progress_status: the in-progress status to detect stuck phrases
            priority_status: status processed first (e.g. failed)
            base_statuses: remaining eligible statuses (e.g. draft, generating_done)

        :returns:
            phrase: the selected Phrase, or None if no eligible rows exist
        """
        stuck = and_(
            Phrase.status == in_progress_status,
            Phrase.updated_at
            < func.now() - timedelta(minutes=settings.STUCK_THRESHOLD),
        )
        stmt = (
            select(Phrase)
            .where(
                or_(
                    Phrase.status.in_([priority_status, *base_statuses]),
                    stuck,
                )
            )
            .order_by(
                case(
                    (stuck, 0),
                    (Phrase.status == priority_status, 1),
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
    async def get_batch_for_processing(
        self,
        in_progress_status: PhraseStatusEnum,
        priority_status: PhraseStatusEnum,
        base_statuses: list[PhraseStatusEnum],
        lang: str,
        exclude_id: int,
        limit: int = 6,
    ) -> list[Phrase]:
        """Return additional phrases for a batch, filtered by lang and excluding one ID

        :param:
            in_progress_status: the in-progress status to detect stuck phrases
            priority_status: status processed first (e.g. failed)
            base_statuses: remaining eligible statuses
            lang: language code to match
            exclude_id: phrase ID already claimed as the batch leader
            limit: maximum number of additional phrases to return

        :returns:
            phrases: list of additional Phrase objects, locked for update
        """
        stuck = and_(
            Phrase.status == in_progress_status,
            Phrase.updated_at
            < func.now() - timedelta(minutes=settings.STUCK_THRESHOLD),
        )
        stmt = (
            select(Phrase)
            .where(
                or_(
                    Phrase.status.in_([priority_status, *base_statuses]),
                    stuck,
                ),
                Phrase.lang == lang,
                Phrase.id != exclude_id,
            )
            .order_by(
                case(
                    (stuck, 0),
                    (Phrase.status == priority_status, 1),
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
    async def get_ids_by_originals(self, originals: list[str], lang: str) -> list[int]:
        """Return IDs of phrases matching the given originals and lang

        Used after bulk_create (do_nothing) to retrieve IDs of both newly inserted
        and pre-existing rows that would have conflicted.

        :param:
            originals: list of normalised phrase strings to look up
            lang: language code to filter by

        :returns:
            ids: list of phrase IDs
        """
        stmt = select(Phrase.id).where(
            Phrase.original.in_(originals),
            Phrase.lang == lang,
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    @log_decorator(level=logging.DEBUG)
    async def update_status(self, ids: list[int], status: PhraseStatusEnum) -> None:
        """Set the given status for the specified phrase IDs

        :param:
            ids: list of phrase IDs to update
            status: target PhraseStatusEnum value

        :returns:
            None
        """
        stmt = (
            update(Phrase)
            .where(Phrase.id.in_(ids))
            .values(status=status, updated_at=func.now())
        )
        await self._session.execute(stmt)
