import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from app.common.logging import log_decorator
from app.dependencies.services import (
    get_phrase_service_without_session,
    get_prompt_service,
)
from app.pyd.responses import UploadImagesResponse
from app.services.phrase_service import PhraseService
from app.services.prompt_service import PromptService

router = APIRouter(
    prefix="/pipeline",
    tags=["Pipeline"],
)


@router.post(
    "/w1_upload",
    response_model=UploadImagesResponse,
    status_code=status.HTTP_201_CREATED,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["images"],
                        "properties": {
                            "images": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            },
                        },
                    }
                }
            },
        }
    },
)
@log_decorator(level=logging.INFO)
async def w1_upload_images(
    phrase_service: Annotated[
        PhraseService, Depends(get_phrase_service_without_session)
    ],
    prompt_service: Annotated[PromptService, Depends(get_prompt_service)],
    images: Annotated[list[UploadFile], File()],
    lang: Annotated[Literal["ru", "en"], Query()] = "ru",
) -> Any:
    """Upload a batch of images and trigger the phrase extraction pipeline

    :role:
        user

    :param:
        phrase_service: service responsible for the vision pipeline
        prompt_service: service that provides the vision prompt
        images: one or more uploaded image files
        lang: target language for extracted phrases ('ru' or 'en')

    :returns:
        result: UploadImagesResponse with phrases_found, inserted, and skipped counts
    """
    images_raw: list[bytes] = []
    try:
        for image in images:
            images_raw.append(await image.read())
    finally:
        for image in images:
            await image.close()

    prompt = prompt_service.get("pixtral_vision", lang)
    return await phrase_service.upload_images(
        images_raw=images_raw, prompt=prompt, lang=lang
    )
