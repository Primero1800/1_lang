import logging
from typing import Any
from uuid import uuid4

from qdrant_client.models import PointStruct

from app.common.enums import PhraseStatusEnum
from app.common.logging import log_decorator, logger
from app.models.phrase_data import PhraseData
from app.models.phrase_embeddings import PhraseEmbedding
from app.models.phrases import Phrase
from app.repositories.phrase_vector_repository import PhraseVectorRepository
from app.services.base import BaseDeps, BaseService
from app.uow import UnitOfWork


class PhraseLoadingService(BaseService):
    """W5 worker service: loads phrase embeddings and variants into Qdrant"""

    def __init__(
        self,
        base_deps: BaseDeps,
        loading_repository: PhraseVectorRepository,
        uow: UnitOfWork | None = None,
    ) -> None:
        """Initialize with shared infrastructure deps and a Qdrant loading repository

        :param:
            base_deps: shared infrastructure dependencies
            loading_repository: repository for Qdrant upsert operations
            uow: optional request-scoped UnitOfWork session

        :returns:
            None
        """
        super().__init__(base_deps, uow)
        self.loading_repository = loading_repository

    @log_decorator(level=logging.INFO)
    async def _fetch_batch(
        self, batch_size: int
    ) -> tuple[list[Phrase], dict[int, list[float]], dict[int, dict[str, Any]]]:
        """Atomically claim a batch of phrases ready for loading and fetch their embeddings and variants

        :param:
            batch_size: maximum number of phrases to claim in one call

        :returns:
            batch: list of claimed Phrase objects
            embeddings_map: dict of {phrase_id: embedding vector}
            variants_map: dict of {phrase_id: variants dict}
        """
        async with self.uow_factory as uow:
            # 1. Claim a batch of phrases eligible for loading
            batch = await uow.phrase_repository.get_batch_for_processing(
                in_progress_status=PhraseStatusEnum.LOADING_IN_PROGRESS,
                priority_status=PhraseStatusEnum.LOADING_FAILED,
                base_statuses=[PhraseStatusEnum.EMBEDDING_DONE],
                limit=batch_size,
            )
            if not batch:
                return [], {}, {}
            # 2. Log each chosen phrase
            for i, member in enumerate(batch, start=1):
                logger.info(
                    f"[W5, loading] {i} chosen: id={member.id}, lang={member.lang}, status={member.status}"
                )
            ids = [p.id for p in batch]
            # 3. Fetch embeddings and variants for the batch
            embeddings: list[
                PhraseEmbedding
            ] = await uow.phrase_embedding_repository.get_by_phrase_ids(ids)
            variants: list[
                PhraseData
            ] = await uow.phrase_data_repository.get_by_phrase_ids(ids)
            # 4. Mark phrases as in-progress to prevent double-claiming
            await uow.phrase_repository.update_status(
                ids=ids,
                status=PhraseStatusEnum.LOADING_IN_PROGRESS,
            )
        # 5. Build lookup maps by phrase_id
        embeddings_map = {e.phrase_id: e.embedding for e in embeddings}
        variants_map = {v.phrase_id: v.variants for v in variants}
        return batch, embeddings_map, variants_map

    def _build_points(
        self,
        batch: list[Phrase],
        embeddings_map: dict[int, list[float]],
        variants_map: dict[int, dict[str, Any]],
    ) -> tuple[list[PointStruct], set[int]]:
        """Build Qdrant PointStruct objects for phrases that have both embedding and variants

        :param:
            batch: list of Phrase objects
            embeddings_map: dict of {phrase_id: embedding vector}
            variants_map: dict of {phrase_id: variants dict}

        :returns:
            points: list of PointStruct ready for upsert
            failed_ids: set of phrase IDs missing embedding or variants
        """
        points = []
        failed_ids = set()
        for phrase in batch:
            embedding = embeddings_map.get(phrase.id)
            variants = variants_map.get(phrase.id)
            if not embedding or not variants:
                logger.warning(
                    f"[W5, loading] id={phrase.id} missing {'embedding' if not embedding else 'variants'} — skipping"
                )
                failed_ids.add(phrase.id)
                continue
            points.append(
                PointStruct(
                    id=phrase.id,
                    vector=embedding,
                    payload={
                        "id": phrase.id,
                        "uuid": str(uuid4()),
                        "original": phrase.original,
                        "tag": phrase.tag,
                        "lang": phrase.lang,
                        "variants": variants,
                    },
                )
            )
        return points, failed_ids

    @log_decorator(level=logging.INFO)
    async def w5_load(self, batch_size: int) -> dict[str, int]:
        """Run one W5 cycle: fetch → build Qdrant points → upsert → mark done/failed

        :param:
            batch_size: number of phrases per Qdrant upsert call

        :returns:
            result: dict with 'processed', 'failed', and 'skipped' counts
        """
        # 1. Fetch batch with embeddings and variants
        batch, embeddings_map, variants_map = await self._fetch_batch(batch_size)
        if not batch:
            return {"processed": 0, "failed": 0, "skipped": 1}

        # 2. Build Qdrant points
        points, build_failed_ids = self._build_points(
            batch, embeddings_map, variants_map
        )

        # 3. Upsert to Qdrant
        upsert_failed_ids: set[int] = set()
        upserted_count = 0
        if points:
            (
                upserted_count,
                upsert_failed_ids,
            ) = await self.loading_repository.bulk_upsert(points)
            logger.info(
                f"[W5, loading] upserted to Qdrant: {upserted_count}/{len(points)}"
            )

        failed_ids = build_failed_ids | upsert_failed_ids
        done_ids = {p.id for p in batch} - failed_ids

        # 4. Update statuses in DB
        async with self.uow_factory as uow:
            if done_ids:
                logger.info(f"[W5, loading] done ids: {done_ids}")
                await uow.phrase_repository.update_status(
                    ids=list(done_ids), status=PhraseStatusEnum.LOADING_DONE
                )
            if failed_ids:
                logger.info(f"[W5, loading] failed ids: {failed_ids}")
                await uow.phrase_repository.update_status(
                    ids=list(failed_ids), status=PhraseStatusEnum.LOADING_FAILED
                )

        return {
            "processed": len(done_ids),
            "failed": len(failed_ids),
            "skipped": 0,
            "upserted": upserted_count,
        }
