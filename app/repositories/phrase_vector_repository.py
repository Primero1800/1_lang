import logging

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    QueryRequest,
    ScoredPoint,
)

from app.adapters.vector_client import VectorClientAbstract
from app.common.logging import log_decorator, logger
from app.core.config import settings


class PhraseVectorRepository:
    """Repository for Qdrant operations on the phrases collection: upsert and search"""

    def __init__(self, vector_client: VectorClientAbstract) -> None:
        """Initialize with a vector database client

        :param:
            vector_client: the Qdrant client to use for upsert and search operations

        :returns:
            None
        """
        self._client = vector_client

    @log_decorator(level=logging.DEBUG)
    async def bulk_upsert(self, points: list[PointStruct]) -> tuple[int, set[int]]:
        """Upsert a batch of points into the default Qdrant collection

        :param:
            points: list of PointStruct objects with id, vector, and payload

        :returns:
            upserted: number of points successfully upserted
            failed_ids: set of point IDs from chunks that failed
        """
        if not points:
            return 0, set()
        upserted = 0
        failed_ids: set[int] = set()
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
                failed_ids.update(p.id for p in chunk)  # type: ignore[misc]
                continue
            if result is not None:
                upserted += len(chunk)
            else:
                failed_ids.update(p.id for p in chunk)  # type: ignore[misc]
        return upserted, failed_ids

    @log_decorator(level=logging.DEBUG)
    async def search_batch(
        self,
        vectors: list[list[float]],
        tags: list[str],
        lang: str,
        limit: int = 3,
    ) -> list[list[ScoredPoint]]:
        """Search Qdrant for nearest neighbours for each vector, each filtered by lang and its own tag

        :param:
            vectors: list of query embedding vectors
            tags: tag to search within for each vector (must match payload 'tag' field)
            lang: language to filter by (must match payload 'lang' field)
            limit: maximum results per vector

        :returns:
            results: list of ScoredPoint lists, one per input vector
        """
        requests = [
            QueryRequest(
                query=vector,
                filter=Filter(
                    must=[
                        FieldCondition(key="lang", match=MatchValue(value=lang)),
                        FieldCondition(key="tag", match=MatchValue(value=tag)),
                    ]
                ),
                limit=limit,
                score_threshold=settings.T1_SEARCH_MIN_SCORE,
                with_payload=True,
                with_vector=False,
            )
            for vector, tag in zip(vectors, tags)
        ]
        return await self._client.search_batch(requests=requests)
