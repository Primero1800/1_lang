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
    @abc.abstractmethod
    async def start(self) -> None:
        pass

    @abc.abstractmethod
    async def stop(self) -> None:
        pass

    @abc.abstractmethod
    async def collection_exists(
        self, collection_name: str | None = None, raise_exception: bool = False
    ) -> bool:
        pass

    @abc.abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams | None = None,
        raise_exception: bool = False,
    ) -> bool:
        pass

    @abc.abstractmethod
    async def upsert(
        self,
        collection_name: str,
        points: Sequence[PointStruct],
        raise_exception: bool = False,
    ) -> UpdateResult | None:
        pass

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
        pass


class QdrantVectorClient(VectorClientAbstract):
    def __init__(self) -> None:
        self._client: AsyncQdrantClient = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            grpc_port=settings.QDRANT_GRPC_PORT,
            prefer_grpc=True,
            api_key=settings.QDRANT_API_KEY,
            check_compatibility=False,
            https=False,
        )

    async def start(self) -> None:
        try:
            if not await self.collection_exists(settings.VECTOR_DB_COLLECTION):
                await self.create_collection(settings.VECTOR_DB_COLLECTION)
        except Exception as exc:
            logger.error("Unexpected error while collection creating", exc_info=exc)
            raise exc
        logger.debug("Qdrant vector client wrapper ready.")

    async def stop(self) -> None:
        await self._client.close()
        logger.debug("Qdrant vector client connection pool closed.")

    @log_decorator(level=logging.DEBUG)
    async def collection_exists(
        self, collection_name: str | None = None, raise_exception: bool = False
    ) -> bool:
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
        config = vectors_config or VectorParams(size=1024, distance=Distance.COSINE)
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
