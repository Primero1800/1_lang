import logging
from typing import Any

from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from app.adapters.embeddings_client import TrackedMistralEmbeddings
from app.common.enums import PhraseStatusEnum
from app.common.exceptions import EmbeddingPipelineException
from app.common.logging import log_decorator, logger
from app.core.config import settings
from app.models.phrases import Phrase
from app.services.base import BaseService

_W4_MODEL = settings.MISTRAL_EMBED_MODEL
_w4_embeddings = TrackedMistralEmbeddings(
    model=_W4_MODEL,
    api_key=SecretStr(settings.MISTRAL_API_KEY),
)


class PhraseEmbeddingService(BaseService):
    """W4 worker service: fetches translated phrases, embeds them via Mistral, saves vectors"""

    _embeddings = _w4_embeddings
    _OPERATION = "w4_embed"

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
    async def _embed(self, batch: list[Phrase]) -> dict[int, list[float]]:
        """Embed all phrases in the batch in a single API call

        :param:
            batch: list of Phrase objects to embed

        :returns:
            matched: dict of {phrase_id: embedding_vector}
        """
        texts = [p.original for p in batch]
        vectors, input_tokens = await self._embeddings.aembed_with_usage(texts)
        if len(vectors) != len(batch):
            raise EmbeddingPipelineException(
                detail=f"Vector count mismatch: got {len(vectors)}, expected {len(batch)}"
            )
        self._queue_token_usage(
            model=_W4_MODEL,
            operation=self._OPERATION,
            input_tokens=input_tokens,
        )
        return {batch[i].id: vectors[i] for i in range(len(batch))}

    @log_decorator(level=logging.INFO)
    async def _save_vectors(
        self, matched: dict[int, list[float]], sent_ids: set[int]
    ) -> dict[str, int]:
        """Persist embedding vectors and update phrase statuses

        :param:
            matched: dict of {phrase_id: embedding_vector}
            sent_ids: set of all phrase IDs sent to the embedding API

        :returns:
            result: dict with 'processed', 'failed', and 'skipped' counts
        """
        returned_ids = set(matched.keys())
        failed_ids = sent_ids - returned_ids

        async with self.uow_factory as uow:
            # 1. Persist embedding vectors
            if matched:
                rows = [
                    {"phrase_id": pid, "embedding": vector}
                    for pid, vector in matched.items()
                ]
                await uow.phrase_embedding_repository.bulk_upsert_embeddings(rows)
            # 2. Update phrase statuses (DONE / FAILED)
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

    @log_decorator(level=logging.INFO)
    async def w4_embed(self, batch_size: int) -> dict[str, int]:
        """Run one W4 cycle: fetch → embed via Mistral → save vectors → mark done/failed

        :param:
            batch_size: number of phrases per embedding call

        :returns:
            result: dict with 'processed', 'failed', and 'skipped' counts
        """
        batch = await self._fetch_batch(batch_size)
        if not batch:
            return {"processed": 0, "failed": 0, "skipped": 0}

        sent_ids = {p.id for p in batch}

        # 1. Bind sent_ids into the _save_vectors step
        async def _save_for_batch(matched: dict[int, list[float]]) -> dict[str, Any]:
            return await self._save_vectors(matched, sent_ids)

        # 2. Assemble the embedding chain
        chain = (
            RunnableLambda(self._embed) | RunnableLambda(_save_for_batch)
        ).with_config(run_name=self._OPERATION)

        # 3. Invoke the chain; mark all phrases as failed on any error
        try:
            return await chain.ainvoke(batch)
        except Exception as exc:
            if not isinstance(exc, EmbeddingPipelineException):
                logger.error("[W4] embedding failed: %s", exc, exc_info=exc)
            async with self.uow_factory as uow:
                await uow.phrase_repository.update_status(
                    ids=list(sent_ids), status=PhraseStatusEnum.EMBEDDING_FAILED
                )
            if isinstance(exc, EmbeddingPipelineException):
                raise
            raise EmbeddingPipelineException(detail=str(exc)) from exc
