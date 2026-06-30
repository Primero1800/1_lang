from app.commands.base import BaseCommand
from app.common.enums import WorkerRoleEnum
from app.core.config import settings
from app.repositories.phrase_vector_repository import PhraseVectorRepository
from app.services.phrase_loading_service import PhraseLoadingService


class CommandW5(BaseCommand):
    """Dispatch command for W5: load embedded phrases into Qdrant"""

    _ROLE = WorkerRoleEnum.W5

    async def _do_execute(self) -> dict:
        client = (
            self._base_deps.vector_client_main
            if settings.QDRANT_MAIN_ENABLED
            else self._base_deps.vector_client
        )
        loading_repository = PhraseVectorRepository(client)
        return await PhraseLoadingService(self._base_deps, loading_repository).w5_load(batch_size=self._batch_size)
