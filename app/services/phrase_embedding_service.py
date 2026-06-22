import logging

from app.common.enums import PhraseStatusEnum
from app.common.logging import log_decorator, logger
from app.models.phrases import Phrase
from app.services.base import BaseService


class PhraseEmbeddingService(BaseService):
    """W4 worker service: fetches translated phrases, embeds them via Mistral, saves vectors"""

    @log_decorator(level=logging.INFO)
    async def _fetch_batch(self, batch_size: int) -> list[Phrase]:
        """Atomically claim a batch of phrases ready for embedding

        :param:
            batch_size: maximum number of phrases to claim in one call

        :returns:
            batch: list of claimed Phrase objects (empty if nothing is ready)
        """
        async with self.uow_factory as uow:
            batch = await uow.phrase_repository.get_batch_for_processing(
                in_progress_status=PhraseStatusEnum.EMBEDDING_IN_PROGRESS,
                priority_status=PhraseStatusEnum.EMBEDDING_FAILED,
                base_statuses=[PhraseStatusEnum.TRANSLATING_DONE],
                limit=batch_size,
            )
            if not batch:
                return []
            for i, member in enumerate(batch, start=1):
                logger.info(
                    f"[W4, embedding] {i} chosen: id={member.id}, lang={member.lang}, status={member.status}"
                )
            await uow.phrase_repository.update_status(
                ids=[p.id for p in batch],
                status=PhraseStatusEnum.EMBEDDING_IN_PROGRESS,
            )
        return batch

    @log_decorator(level=logging.INFO)
    async def _call_embed(self, batch: list[Phrase]) -> dict[int, list[float]]:
        """Embed all phrases in the batch in a single API call

        :param:
            batch: list of Phrase objects to embed

        :returns:
            matched: dict of {phrase_id: embedding_vector}, empty on failure
        """
        if not self.ai_client.supports_embed:
            logger.error("ai_client does not support embed")
            return {}
        texts = [p.original for p in batch]
        result = await self.ai_client.embed(
            texts, task_type="document", operation="w4_embed"
        )
        if not result:
            return {}
        vectors: list[list[float]] = result  # type: ignore[assignment]
        if len(vectors) != len(batch):
            logger.error(
                "[W4, embedding] vector count mismatch: got %d, expected %d",
                len(vectors),
                len(batch),
            )
            return {}
        return {batch[i].id: vectors[i] for i in range(len(batch))}

    @log_decorator(level=logging.INFO)
    async def w4_embed(self, batch_size: int) -> dict[str, int]:
        """Run one W4 cycle: fetch → embed via Mistral → save vectors → mark done/failed

        :param:
            batch_size: number of phrases per embedding call

        :returns:
            result: dict with 'processed', 'failed', and 'skipped' counts
        """
        # 1. Fetch a batch of eligible phrases
        batch = await self._fetch_batch(batch_size)
        if not batch:
            return {"processed": 0, "failed": 0, "skipped": 1}

        sent_ids = {p.id for p in batch}

        # 2. Embed all phrases in one API call
        matched = await self._call_embed(batch)
        returned_ids = set(matched.keys())
        failed_ids = sent_ids - returned_ids

        # 3. Persist vectors and update phrase statuses
        async with self.uow_factory as uow:
            if matched:
                rows = [
                    {"phrase_id": pid, "embedding": vector}
                    for pid, vector in matched.items()
                ]
                await uow.phrase_embedding_repository.bulk_upsert_embeddings(rows)
            if returned_ids:
                logger.info(f"[W4, embedding]: returned ids {returned_ids}")
                await uow.phrase_repository.update_status(
                    ids=list(returned_ids), status=PhraseStatusEnum.EMBEDDING_DONE
                )
            if failed_ids:
                logger.info(f"[W4, embedding]: failed ids {failed_ids}")
                await uow.phrase_repository.update_status(
                    ids=list(failed_ids), status=PhraseStatusEnum.EMBEDDING_FAILED
                )

        return {
            "processed": len(returned_ids),
            "failed": len(failed_ids),
            "skipped": 0,
        }
