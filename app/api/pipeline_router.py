import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from app.common.logging import log_decorator
from app.dependencies.services import (
    get_phrase_data_service_without_session,
    get_phrase_service_without_session,
)
from app.pyd.responses import UploadImagesResponse, W2GenerateResponse
from app.services.phrase_data_service import PhraseDataService
from app.services.phrase_service import PhraseService

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
    images: Annotated[list[UploadFile], File()],
    lang: Annotated[Literal["ru", "en"], Query()] = "ru",
) -> Any:
    """Upload a batch of images and trigger the phrase extraction pipeline

    :role:
        user

    :param:
        phrase_service: service responsible for the vision pipeline
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

    return await phrase_service.upload_images(images_raw=images_raw, lang=lang)


@router.post(
    "/w2_generate",
    response_model=W2GenerateResponse,
    status_code=status.HTTP_200_OK,
)
@log_decorator(level=logging.INFO)
async def w2_generate(
    phrase_data_service: Annotated[
        PhraseDataService, Depends(get_phrase_data_service_without_session)
    ],
    batch_size: Annotated[int, Query(ge=1, le=50)] = 7,
) -> Any:
    """Trigger W2: pick a batch of draft phrases and generate tone variants via Mistral

    :role:
        user

    :param:
        phrase_data_service: service responsible for variant generation
        batch_size: number of phrases per Mistral call

    :returns:
        result: W2GenerateResponse with processed, failed, and skipped counts
    """
    return await phrase_data_service.w2_generate(batch_size=batch_size)
