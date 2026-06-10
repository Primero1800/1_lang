import logging
from typing import Any

from app.common.logging import log_decorator, logger
from app.services.base import BaseService


class TestService(BaseService):
    @log_decorator(level=logging.DEBUG)
    async def check(self, text: str) -> Any:
        text_embedding = await self.ai_client.embed(text, task_type="query")
        if not text_embedding:
            return None

        try:
            points = await self.vector_client.search(
                query_vector=text_embedding,
                raise_exception=True,
                limit=10,
                with_payload=True,
            )

            logger.info(points)

            res = []
            for point in points:
                temp = {
                    "score": point.score,
                    "message_id": (
                        point.payload.get("message_id") if point.payload else None
                    ),
                    "chunk_id": (
                        point.payload.get("chunk_id") if point.payload else None
                    ),
                    "total_chunks": (
                        point.payload.get("total_chunks") if point.payload else None
                    ),
                    "text": point.payload.get("text") if point.payload else None,
                }
                res.append(temp)
            return res

        except Exception as exc:
            logger.error("ERROR!!!!!", exc_info=exc)
            return False
