from typing import Any

from app.commands.base import BaseCommand
from app.common.enums import WorkerRoleEnum
from app.dependencies.infrastructure import get_phrase_vector_repository
from app.services.phrase_loading_service import PhraseLoadingService


class CommandW5(BaseCommand):
    """Dispatch command for W5: load embedded phrases into Qdrant"""

    _ROLE = WorkerRoleEnum.W5

    async def _do_execute(self) -> dict[str, Any]:
        loading_repository = await get_phrase_vector_repository(
            local_client=self._base_deps.vector_client,
            main_client=self._base_deps.vector_client_main,
        )
        return await PhraseLoadingService(self._base_deps, loading_repository).w5_load(
            batch_size=self._batch_size
        )
