import logging
from typing import Annotated, Any, Union

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.common.exceptions import VectorDBException
from app.common.logging import log_decorator
from app.dependencies.services import get_test_service_without_session
from app.pyd.requests import SearchSettings, TagExclusionFilters
from app.services.test_service import TestService

router = APIRouter(
    prefix="/test_routes",
    tags=["Test"],
)


@router.post(
    "/t1_search",
    status_code=200,
    response_model=dict[str, Any],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["image"],
                        "properties": {
                            "image": {"type": "string", "format": "binary"},
                        },
                    }
                }
            },
        }
    },
)
@log_decorator(level=logging.INFO)
async def t1_search(
    test_service: Annotated[TestService, Depends(get_test_service_without_session)],
    image: Annotated[UploadFile, File()],
    filters: Annotated[TagExclusionFilters, Depends()],
    search_settings: Annotated[SearchSettings, Depends()],
) -> Any:
    """T1: upload a photo, extract phrases via vision, embed and search Qdrant, return mood-filtered variants

    :role:
        admin

    :param:
        test_service: service orchestrating the vision → embed → search pipeline
        image: single photo to analyse
        filters: tag exclusion flags (not_behavior, not_appearance, etc.)
        search_settings: target language and mood tone (A–E)

    :returns:
        phrases: flat list of matched variant strings
    """
    try:
        image_raw = await image.read()
    finally:
        await image.close()
    try:
        return await test_service.t1_search(
            image_raw=image_raw,
            filters=filters,
            search_settings=search_settings,
        )
    except VectorDBException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database search failed",
        ) from e
