from typing import Any

from app.commands.base import BaseCommand
from app.common.enums import WorkerRoleEnum
from app.services.phrase_embedding_service import PhraseEmbeddingService


class CommandW4(BaseCommand):
    """Dispatch command for W4: generate embeddings via Mistral"""

    _ROLE = WorkerRoleEnum.W4

    async def _do_execute(self) -> dict[str, Any]:
        return await PhraseEmbeddingService(self._base_deps).w4_embed(
            batch_size=self._batch_size
        )
