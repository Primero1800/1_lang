from typing import Any

from app.commands.base import BaseCommand
from app.common.enums import WorkerRoleEnum
from app.services.phrase_translation_service import PhraseTranslationService


class CommandW3(BaseCommand):
    """Dispatch command for W3: translate phrases via Mistral"""

    _ROLE = WorkerRoleEnum.W3

    async def _do_execute(self) -> dict[str, Any]:
        return await PhraseTranslationService(self._base_deps).w3_translate(
            batch_size=self._batch_size
        )
