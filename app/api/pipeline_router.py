import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.common.exceptions import (
    GenerationPipelineException,
    IntegrityDataException,
    VisionPipelineException,
)
from app.common.logging import log_decorator
from app.dependencies.services import (
    get_phrase_loading_service_without_session,
    get_phrase_data_service_without_session,
    get_phrase_embedding_service_without_session,
    get_phrase_service_without_session,
    get_phrase_translation_service_without_session,
)
from app.pyd.responses import (
    UploadImagesResponse,
    W2GenerateResponse,
    W3TranslateResponse,
    W4EmbedResponse,
    W5LoadResponse,
)
from app.services.phrase_data_service import PhraseDataService
from app.services.phrase_embedding_service import PhraseEmbeddingService
from app.services.phrase_loading_service import PhraseLoadingService
from app.services.phrase_service import PhraseService
from app.services.phrase_translation_service import PhraseTranslationService

router = APIRouter(
    prefix="/pipeline",
    tags=["Pipeline"],
)


@router.post(
    "/w1_upload",
    response_model=UploadImagesResponse,
    status_code=status.HTTP_200_OK,
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

    try:
        return await phrase_service.upload_images(images_raw=images_raw, lang=lang)
    except VisionPipelineException as e:
        return {"phrases_found": 0, "inserted": 0, "skipped": 0, "error": str(e.detail)}


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
    batch_size: Annotated[int, Query(ge=1, le=50)] = 5,
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
    try:
        return await phrase_data_service.w2_generate(batch_size=batch_size)
    except GenerationPipelineException as e:
        return {"processed": 0, "failed": 0, "skipped": 0, "error": str(e.detail)}


@router.post(
    "/w3_translate",
    response_model=W3TranslateResponse,
    status_code=status.HTTP_200_OK,
)
@log_decorator(level=logging.INFO)
async def w3_translate(
    phrase_translation_service: Annotated[
        PhraseTranslationService,
        Depends(get_phrase_translation_service_without_session),
    ],
    batch_size: Annotated[int, Query(ge=1, le=50)] = 5,
) -> Any:
    """Trigger W3: translate a batch of generated phrases and their variants via Mistral

    :role:
        user

    :param:
        phrase_translation_service: service responsible for translation
        batch_size: number of phrases per Mistral call

    :returns:
        result: W3TranslateResponse with processed, failed, and skipped counts
    """
    try:
        return await phrase_translation_service.w3_translate(batch_size=batch_size)
    except IntegrityDataException as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Integrity constraint violation during translation",
        ) from e


@router.post(
    "/w4_embed",
    response_model=W4EmbedResponse,
    status_code=status.HTTP_200_OK,
)
@log_decorator(level=logging.INFO)
async def w4_embed(
    phrase_embedding_service: Annotated[
        PhraseEmbeddingService,
        Depends(get_phrase_embedding_service_without_session),
    ],
    batch_size: Annotated[int, Query(ge=1, le=500)] = 200,
) -> Any:
    """Trigger W4: embed a batch of translated phrases via Mistral and store vectors

    :role:
        user

    :param:
        phrase_embedding_service: service responsible for embedding generation
        batch_size: number of phrases per embedding call

    :returns:
        result: W4EmbedResponse with processed, failed, and skipped counts
    """
    try:
        return await phrase_embedding_service.w4_embed(batch_size=batch_size)
    except IntegrityDataException as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Integrity constraint violation during embedding",
        ) from e


@router.post(
    "/w5_load",
    response_model=W5LoadResponse,
    status_code=status.HTTP_200_OK,
)
@log_decorator(level=logging.INFO)
async def w5_load(
    phrase_loading_service: Annotated[
        PhraseLoadingService,
        Depends(get_phrase_loading_service_without_session),
    ],
    batch_size: Annotated[int, Query(ge=1, le=2000)] = 400,
) -> Any:
    """Trigger W5: load a batch of embedded phrases into Qdrant

    :role:
        user

    :param:
        phrase_loading_service: service responsible for Qdrant upsert
        batch_size: number of phrases per upsert call

    :returns:
        result: W5LoadResponse with processed, failed, and skipped counts
    """
    try:
        return await phrase_loading_service.w5_load(batch_size=batch_size)
    except IntegrityDataException as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Integrity constraint violation during loading",
        ) from e
