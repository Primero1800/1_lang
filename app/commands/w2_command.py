from typing import Any

from app.commands.base import BaseCommand
from app.common.enums import WorkerRoleEnum
from app.services.phrase_data_service import PhraseDataService


class CommandW2(BaseCommand):
    """Dispatch command for W2: generate phrase variants via Mistral"""

    _ROLE = WorkerRoleEnum.W2

    async def _do_execute(self) -> dict[str, Any]:
        return await PhraseDataService(self._base_deps).w2_generate(
            batch_size=self._batch_size
        )
