import logging
from datetime import date
from typing import Any

from sqlalchemy import func, literal, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.common.logging import log_decorator
from app.models.ai_token_usage import AiTokenUsage
from app.pyd.requests import AITokenFilter, Pagination
from app.repositories.base_repository import BaseRepository
from app.repositories.repository_error_handler import repository_error_handler


@repository_error_handler
class AiTokenUsageRepository(BaseRepository):
    """Repository for daily-aggregated AI token usage upserts"""

    @log_decorator(level=logging.DEBUG)
    async def accumulate(
        self,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        name: str = "system",
        usage_date: date | None = None,
    ) -> None:
        """Upsert a token usage record, accumulating tokens on conflict

        :param:
            model: AI model identifier
            operation: pipeline operation name (e.g. 'w2_generate')
            input_tokens: number of input tokens to add
            output_tokens: number of output tokens to add
            name: actor name, defaults to 'system'
            usage_date: date to record against, defaults to today

        :returns:
            None
        """
        row = {
            "model": model,
            "date": usage_date or date.today(),
            "name": name,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        stmt = pg_insert(AiTokenUsage).values(row)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_ai_token_usage",
            set_={
                "input_tokens": AiTokenUsage.input_tokens + stmt.excluded.input_tokens,
                "output_tokens": AiTokenUsage.output_tokens
                + stmt.excluded.output_tokens,
                "updated_at": func.now(),
            },
        )
        await self._session.execute(stmt)

    def _build_conditions(self, filters: AITokenFilter) -> list[Any]:
        """Build SQLAlchemy WHERE conditions from AITokenFilter

        :param:
            filters: filter model with optional fields

        :returns:
            conditions: list of SQLAlchemy column expressions
        """
        conditions = []
        if filters.date_from is not None:
            conditions.append(AiTokenUsage.date >= filters.date_from)
        if filters.date_to is not None:
            conditions.append(AiTokenUsage.date <= filters.date_to)
        if filters.model is not None:
            conditions.append(AiTokenUsage.model.ilike(f"%{filters.model}%"))
        if filters.name is not None:
            conditions.append(AiTokenUsage.name.ilike(f"%{filters.name}%"))
        if filters.operation is not None:
            conditions.append(AiTokenUsage.operation.ilike(f"{filters.operation}%"))
        return conditions

    @log_decorator(level=logging.DEBUG)
    async def list_usage(
        self,
        filters: AITokenFilter,
        pagination: Pagination,
    ) -> tuple[list[AiTokenUsage], int]:
        """Return paginated token usage rows matching filters and total count

        :param:
            filters: filter conditions
            pagination: page and per_page values

        :returns:
            rows: matching AiTokenUsage ORM instances for the requested page
            total_count: total number of matching rows across all pages
        """
        conditions = self._build_conditions(filters)
        total_count: int = (
            await self._session.execute(
                select(func.count()).select_from(AiTokenUsage).where(*conditions)
            )
        ).scalar_one()
        offset = (pagination.page - 1) * pagination.per_page

        rows = (
            await self._session.execute(
                select(AiTokenUsage)
                .where(*conditions)
                .order_by(AiTokenUsage.date.desc())
                .limit(pagination.per_page)
                .offset(offset)
            )
        ).scalars().all()
        return list(rows), total_count

    @log_decorator(level=logging.DEBUG)
    async def bulk_accumulate(self, rows: list[dict]) -> None:
        """Upsert multiple pre-aggregated token usage rows in a single statement

        :param:
            rows: list of dicts with keys model, operation, name, date, input_tokens, output_tokens

        :returns:
            None
        """
        if not rows:
            return
        stmt = pg_insert(AiTokenUsage).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_ai_token_usage",
            set_={
                "input_tokens": AiTokenUsage.input_tokens + stmt.excluded.input_tokens,
                "output_tokens": AiTokenUsage.output_tokens
                + stmt.excluded.output_tokens,
                "updated_at": func.now(),
            },
        )
        await self._session.execute(stmt)
