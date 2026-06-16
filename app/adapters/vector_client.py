import abc
import logging
from typing import Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    PointStruct,
    ScoredPoint,
    Filter,
    Distance,
    UpdateResult,
)

from app.common.logging import log_decorator, logger
from app.core.config import settings


class VectorClientAbstract(abc.ABC):
    """Abstract base class for vector database client implementations"""

    @abc.abstractmethod
    async def start(self) -> None:
        """Start the client and ensure the default collection exists

        :returns:
            None
        """

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop the client and release any held connections

        :returns:
            None
        """

    @abc.abstractmethod
    async def collection_exists(
        self, collection_name: str | None = None, raise_exception: bool = False
    ) -> bool:
        """Check whether a collection exists in the vector database

        :param:
            collection_name: target collection name; defaults to configured collection
            raise_exception: whether to re-raise exceptions instead of returning False

        :returns:
            exists: True if the collection exists, False otherwise
        """

    @abc.abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams | None = None,
        raise_exception: bool = False,
    ) -> bool:
        """Create a new collection with the given vector configuration

        :param:
            collection_name: name for the new collection
            vectors_config: vector size and distance metric; defaults to configured values
            raise_exception: whether to re-raise exceptions instead of returning False

        :returns:
            success: True if the collection was created successfully
        """

    @abc.abstractmethod
    async def upsert(
        self,
        collection_name: str,
        points: Sequence[PointStruct],
        raise_exception: bool = False,
    ) -> UpdateResult | None:
        """Upsert a sequence of points into a collection

        :param:
            collection_name: target collection name
            points: sequence of PointStruct objects to upsert
            raise_exception: whether to re-raise exceptions instead of returning None

        :returns:
            result: Qdrant UpdateResult, or None on failure
        """

    @abc.abstractmethod
    async def search(
        self,
        query_vector: list[float],
        collection_name: str | None = None,
        query_filter: Filter | None = None,
        limit: int = 10,
        raise_exception: bool = False,
        with_payload: bool = True,
    ) -> list[ScoredPoint]:
        """Search for nearest neighbours by vector similarity

        :param:
            query_vector: the query embedding vector
            collection_name: target collection name; defaults to configured collection
            query_filter: optional Qdrant filter to narrow results
            limit: maximum number of results to return
            raise_exception: whether to re-raise exceptions instead of returning []
            with_payload: whether to include payload fields in results

        :returns:
            points: list of ScoredPoint results ordered by similarity
        """


class QdrantVectorClient(VectorClientAbstract):
    """Qdrant vector database client using the async gRPC/HTTP AsyncQdrantClient"""

    def __init__(self) -> None:
        """Initialize the Qdrant client using settings from the environment

        :returns:
            None
        """
        self._client: AsyncQdrantClient = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            grpc_port=settings.QDRANT_GRPC_PORT,
            prefer_grpc=settings.QDRANT_PREFER_GRPC,
            api_key=settings.QDRANT_API_KEY,
            check_compatibility=False,
            https=settings.QDRANT_HTTPS,
        )

    async def start(self) -> None:
        """Ensure the default collection exists, creating it if necessary

        :raise:
            Exception: re-raised if collection creation fails unexpectedly

        :returns:
            None
        """
        try:
            if not await self.collection_exists(settings.VECTOR_DB_COLLECTION):
                await self.create_collection(settings.VECTOR_DB_COLLECTION)
        except Exception as exc:
            logger.error("Unexpected error while collection creating", exc_info=exc)
            raise exc
        logger.debug("Qdrant vector client wrapper ready.")

    async def stop(self) -> None:
        """Close the underlying Qdrant connection pool

        :returns:
            None
        """
        await self._client.close()
        logger.debug("Qdrant vector client connection pool closed.")

    @log_decorator(level=logging.DEBUG)
    async def collection_exists(
        self, collection_name: str | None = None, raise_exception: bool = False
    ) -> bool:
        """Check whether a collection exists in Qdrant

        :param:
            collection_name: target collection name; defaults to configured collection
            raise_exception: whether to re-raise exceptions instead of returning False

        :returns:
            exists: True if the collection exists, False otherwise
        """
        target_collection = collection_name or settings.VECTOR_DB_COLLECTION
        try:
            return await self._client.collection_exists(
                collection_name=target_collection
            )
        except Exception as exc:
            logger.warning(
                f"Error checking collection '{target_collection}' existence",
                exc_info=exc,
            )
            if raise_exception:
                raise
            return False

    @log_decorator(level=logging.DEBUG)
    async def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams | None = None,
        raise_exception: bool = False,
    ) -> bool:
        """Create a new Qdrant collection with cosine distance by default

        :param:
            collection_name: name for the new collection
            vectors_config: vector params override; defaults to configured size + cosine
            raise_exception: whether to re-raise exceptions instead of returning False

        :returns:
            success: True if collection was created successfully
        """
        config = vectors_config or VectorParams(
            size=settings.VECTOR_DB_VECTOR_SIZE, distance=Distance.COSINE
        )
        try:
            return await self._client.create_collection(
                collection_name=collection_name, vectors_config=config
            )
        except Exception as exc:
            logger.critical(
                f"Failed to create collection '{collection_name}'", exc_info=exc
            )
            if raise_exception:
                raise
            return False

    @log_decorator(level=logging.DEBUG)
    async def upsert(
        self,
        collection_name: str,
        points: Sequence[PointStruct],
        raise_exception: bool = False,
    ) -> UpdateResult | None:
        """Upsert a batch of points into the specified Qdrant collection

        :param:
            collection_name: target collection name
            points: sequence of PointStruct objects to upsert
            raise_exception: whether to re-raise exceptions instead of returning None

        :returns:
            result: Qdrant UpdateResult, or None on failure
        """
        try:
            return await self._client.upsert(
                collection_name=collection_name, points=points
            )
        except Exception as exc:
            logger.error(f"Error upserting points to '{collection_name}'", exc_info=exc)
            if raise_exception:
                raise
            return None

    @log_decorator(level=logging.DEBUG)
    async def search(
        self,
        query_vector: list[float],
        collection_name: str | None = None,
        query_filter: Filter | None = None,
        limit: int = 10,
        with_payload: bool = True,
        raise_exception: bool = False,
    ) -> list[ScoredPoint]:
        """Search for nearest neighbours in a Qdrant collection

        :param:
            query_vector: the query embedding vector
            collection_name: target collection name; defaults to configured collection
            query_filter: optional Qdrant filter to narrow results
            limit: maximum number of results to return
            with_payload: whether to include payload fields in results
            raise_exception: whether to re-raise exceptions instead of returning []

        :returns:
            points: list of ScoredPoint results ordered by similarity
        """
        try:
            response = await self._client.query_points(
                collection_name=collection_name or settings.VECTOR_DB_COLLECTION,
                query=query_vector,  # type: ignore
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=False,
            )
            return response.points
        except Exception as exc:
            logger.error(
                f"Error during vector search in '{collection_name}'", exc_info=exc
            )
            if raise_exception:
                raise
            return []
