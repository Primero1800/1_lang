import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import joinedload

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
    async def bulk_create(self, rows: list[dict[str, Any]]) -> list[int]:
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
        lang: str | None = None,
        exclude_id: int | None = None,
        limit: int = 6,
    ) -> list[Phrase]:
        """Return phrases for a batch with optional lang and exclude_id filters

        :param:
            in_progress_status: the in-progress status to detect stuck phrases
            priority_status: status processed first (e.g. failed)
            base_statuses: remaining eligible statuses
            lang: language code to match; None to skip language filter
            exclude_id: phrase ID to exclude; None to skip exclusion filter
            limit: maximum number of phrases to return

        :returns:
            phrases: list of Phrase objects, locked for update
        """
        stuck = and_(
            Phrase.status == in_progress_status,
            Phrase.updated_at
            < func.now() - timedelta(minutes=settings.STUCK_THRESHOLD),
        )
        conditions = [
            or_(
                Phrase.status.in_([priority_status, *base_statuses]),
                stuck,
            ),
        ]
        if lang is not None:
            conditions.append(Phrase.lang == lang)
        if exclude_id is not None:
            conditions.append(Phrase.id != exclude_id)
        stmt = (
            select(Phrase)
            .where(*conditions)
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
    async def get_ids_by_originals(
        self, originals: list[str], lang: str
    ) -> dict[str, int]:
        """Return a mapping of original text → ID for phrases matching the given originals and lang

        Used after bulk_create (do_nothing) to retrieve IDs of both newly inserted
        and pre-existing rows that would have conflicted.

        :param:
            originals: list of normalised phrase strings to look up
            lang: language code to filter by

        :returns:
            mapping: dict of {original_text: phrase_id}
        """
        stmt = select(Phrase.original, Phrase.id).where(
            Phrase.original.in_(originals),
            Phrase.lang == lang,
        )
        result = await self._session.execute(stmt)
        return {row.original: row.id for row in result}

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

    @log_decorator(level=logging.DEBUG)
    async def get_pipeline_status_counts(
        self, stuck_threshold_sec: int
    ) -> dict[str, int]:
        """Return ready phrase counts per status for the pipeline scheduler

        For IN_PROGRESS statuses: counts only phrases stuck longer than threshold.
        For all other statuses: counts all phrases.
        Excludes LOADING_DONE. Single query, one round-trip.

        :param:
            stuck_threshold_sec: seconds after which an IN_PROGRESS phrase is considered stuck

        :returns:
            counts: {status_value: ready_count}
        """
        threshold_dt = datetime.now(settings.default_timezone) - timedelta(
            seconds=stuck_threshold_sec
        )

        _in_progress = [
            PhraseStatusEnum.GENERATING_IN_PROGRESS,
            PhraseStatusEnum.TRANSLATING_IN_PROGRESS,
            PhraseStatusEnum.EMBEDDING_IN_PROGRESS,
            PhraseStatusEnum.LOADING_IN_PROGRESS,
        ]

        stmt = (
            select(
                Phrase.status,
                case(
                    (
                        Phrase.status.in_(_in_progress),
                        func.count().filter(Phrase.updated_at < threshold_dt),
                    ),
                    else_=func.count(),
                ).label("ready"),
            )
            .where(Phrase.status != PhraseStatusEnum.LOADING_DONE)
            .group_by(Phrase.status)
        )

        result = await self._session.execute(stmt)
        return {row.status.value: row.ready for row in result}

    @log_decorator(level=logging.DEBUG)
    async def get_sample_per_tag(
        self, sample_size: int, load_data: bool = False, lang: str | None = None
    ) -> list[Phrase]:
        """Return a representative sample of LOADING_DONE phrases.

        Designed for evaluation dataset assembly (W1/W2/W3 LLM-as-judge pipeline).

        :param:
            sample_size: total number of phrases to sample
            load_data: if True, eagerly loads phrase_data via JOIN
            lang: if set, filters by language code

        :returns:
            phrases: list of Phrase objects ordered by id
        """
        conditions = [Phrase.status == PhraseStatusEnum.LOADING_DONE]
        if lang is not None:
            conditions.append(Phrase.lang == lang)
        total: int = (
            await self._session.execute(
                select(func.count(Phrase.id)).where(*conditions)
            )
        ).scalar_one()

        step = max(total // sample_size, 1)
        target_ids = [i * step + 1 for i in range(sample_size + 1)]

        stmt = (
            select(Phrase)
            .where(*conditions, Phrase.id.in_(target_ids))
            .order_by(Phrase.id)
            .limit(sample_size)
        )
        if load_data:
            stmt = stmt.options(joinedload(Phrase.phrase_data))

        result = await self._session.execute(stmt)
        return list(result.scalars().all())
