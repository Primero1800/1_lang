import logging

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import PointStruct

from app.adapters.vector_client import VectorClientAbstract
from app.common.logging import log_decorator, logger
from app.core.config import settings


class PhraseLoadingRepository:
    """Repository for upserting phrase vectors and payload into Qdrant"""

    def __init__(self, vector_client: VectorClientAbstract) -> None:
        """Initialize with a vector database client

        :param:
            vector_client: the Qdrant client to use for upsert operations

        :returns:
            None
        """
        self._client = vector_client

    @log_decorator(level=logging.DEBUG)
    async def bulk_upsert(self, points: list[PointStruct]) -> int:
        """Upsert a batch of points into the default Qdrant collection

        :param:
            points: list of PointStruct objects with id, vector, and payload

        :returns:
            count: number of points submitted (0 on failure)
        """
        if not points:
            return 0
        upserted = 0
        for i in range(0, len(points), settings.QDRANT_MAIN_UPSERT_CHUNK_SIZE):
            chunk = points[i : i + settings.QDRANT_MAIN_UPSERT_CHUNK_SIZE]
            try:
                result = await self._client.upsert(
                    collection_name=settings.VECTOR_DB_COLLECTION,
                    points=chunk,
                )
            except (UnexpectedResponse, ResponseHandlingException) as e:
                logger.error(
                    f"[W5, loading] Qdrant upsert chunk {i}–{i + len(chunk)} failed: {e}"
                )
                continue
            if result is not None:
                upserted += len(chunk)
        return upserted
