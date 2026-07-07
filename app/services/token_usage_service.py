import logging

from app.common.logging import log_decorator
from app.pyd.requests import AITokenFilter, Pagination
from app.services.base import BaseService


class TokenUsageService(BaseService):
    """Service for reading AI token usage records"""

    @log_decorator(level=logging.INFO)
    async def list_usage(
        self,
        filters: AITokenFilter,
        pagination: Pagination,
    ) -> dict:
        """Return paginated token usage records matching filters

        :param:
            filters: query filters (date range, model, name, operation, exclusions)
            pagination: page and per_page values

        :returns:
            result: dict with per_page, page, total_count and items list
        """
        rows, total_count = await self.uow.ai_token_usage_repository.list_usage(
            filters=filters,
            pagination=pagination,
        )
        items = [
            {
                "model": row.model,
                "date": row.date,
                "name": row.name,
                "operation": row.operation,
                "input_tokens": 0 if filters.exclude_input else row.input_tokens,
                "output_tokens": 0 if filters.exclude_output else row.output_tokens,
                "total_tokens": (0 if filters.exclude_input else row.input_tokens)
                + (0 if filters.exclude_output else row.output_tokens),
                "updated_at": row.updated_at,
            }
            for row in rows
        ]
        return {
            "per_page": pagination.per_page,
            "page": pagination.page,
            "total_count": total_count,
            "items": items,
        }

    @log_decorator(level=logging.INFO)
    async def aggregate_usage(self, filters: AITokenFilter) -> dict:
        """Return summed token usage for all rows matching filters

        :param:
            filters: query filters; model/name/operation reflect filter values in response

        :returns:
            result: flat dict with filter context fields and token sums
        """
        row = await self.uow.ai_token_usage_repository.aggregate_usage(filters=filters)
        input_tokens = 0 if filters.exclude_input else row["input_tokens"]
        output_tokens = 0 if filters.exclude_output else row["output_tokens"]
        return {
            "model": filters.model,
            "name": filters.name,
            "operation": filters.operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
