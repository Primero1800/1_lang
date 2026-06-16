import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, UploadFile

from app.common.logging import log_decorator
from app.dependencies.services import (
    get_test_service_without_session,
)
from app.pyd.responses import PhraseVariantsRequest, PhraseVariantsResponse
from app.services.base import BaseServiceAbstract
from app.services.test_service import TestService

router = APIRouter(
    prefix="/test_routes",
    tags=["Test"],
)

PROMPT = """Опиши подробно, ПОДРОБНО(!) что происходит на изображениях: 
    чем занимается человек, что открыто на экране, какое его поведение по плану:
    конкретно нужно четкое описание в одном расширенном предложении по тэгам:
    1. поведение (как он себя ведет, что делает, может засыпает или наоборот слишком бодрый) - предложение из 5-6 слов, 
    2. внешность (опрятный, причесанный, ухоженный, лысый, косой, хромой, больной) - предложение из 5-6 слов, 
    3. возраст (старый, молодой, сопляк, пердун, сосунок) - предложение из 5-6 слов, 
    4. настроение (приуныл, ржет, веселый, в петлю полезть готов - предложение из 5-6 слов, 
    5. поза (сидит, раком, как царь, забитый) и тп. по такому принципу - предложение из 5-6 слов.
    6. прическа (сама прическа или головной убор по-простому, если прически не видно) - предложение из 5-6 слов
    Очень подробно надо, придирчиво. При этом вариант может быть в том числе вопросительным или восклицательным.

    для каждого фото надо выдать по 5 оригинальных, отличных друг от друга вариантов пунктов от 1 до 5 тэгов в виде
    1. поведение: [Вариант 1_1. Вариант 1_2. Вариант 1_3. Вариант 1_4. Вариант 1_5]
    2. внешность: [Вариант 2_1. Вариант 2_2. Вариант 2_3. Вариант 2_4. Вариант 2_5]
    ...
    6 прическа: [Вариант 6_1. Вариант 6_2. Вариант 6_3. Вариант 6_4. Вариант 6_5]

    Обращаю внимание, что каждый вариант (Вариант 1_1 и д.р) - это не одно слово, а целое предложение из 5-6 слов - этот пункт
    очень важен, не игнорируй его.

    И очень важное дополнительное условие - варианты в рамках одного тега даже в разных фотках не должны повторяться,
    а оставаться уникальными. Т.е. если в первой фотографии в прическе есть например "Нет волос на голове",
    то такого варианта не должно быть в описании прически остальных фотографий. Т.е. в рамках одного тега все
    варианты для всех фотографий должны быть уникальны.

    Все варианты возвращает в нижнем регистре lower(). Ответ должен быть списком list из списков list словарей dict
    Никаких дополнительных фраз - только list[list[dict[int, str]]]
    [
        [
            {1: ["Вариант 1_1", "Вариант 1_2", .. "Вариант 1_5"]},
            {2: ["Вариант 2_1", "Вариант 2_2", .. "Вариант 2_5"]},
            ...
            {6: ["Вариант 6_1", "Вариант 6_2", .. "Вариант 6_5"]},
        ],
        [
            {1: ["Вариант 1_1", "Вариант 1_2", .. "Вариант 1_5"]},
            {2: ["Вариант 2_1", "Вариант 2_2", .. "Вариант 2_5"]},
            ...
            {6: ["Вариант 6_1", "Вариант 6_2", .. "Вариант 6_5"]},
        ],
        ..
        [
            {1: ["Вариант 1_1", "Вариант 1_2", .. "Вариант 1_5"]},
            {2: ["Вариант 2_1", "Вариант 2_2", .. "Вариант 2_5"]},
            ...
            {6: ["Вариант 6_1", "Вариант 6_2", .. "Вариант 6_5"]},
        ],
    ]
    где ключи - номера тэгов из описания
    """


@router.post(
    "/pixtral-vision",
    status_code=200,
    response_model=list[PhraseVariantsRequest],
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
async def pixtral_vision(
    test_service: Annotated[TestService, Depends(get_test_service_without_session)],
    images: Annotated[list[UploadFile], File()],
) -> Any:
    """Run Pixtral vision on uploaded images and return structured phrase list

    :role:
        admin

    :param:
        test_service: service wrapping the Pixtral vision call
        images: one or more uploaded image files

    :returns:
        batch: list of phrase dicts extracted from all images
    """
    images_raw: list[bytes] = []
    try:
        for image in images:
            images_raw.append(await image.read())
    finally:
        for image in images:
            await image.close()
    batch = await test_service.pixtral_vision(images_raw=images_raw, prompt=PROMPT)
    return batch


@router.post(
    "/generate-variants-batch",
    status_code=200,
    response_model=list[PhraseVariantsResponse | None],
)
@log_decorator(level=logging.INFO)
async def generate_variants_batch(
    body: list[PhraseVariantsRequest],
    test_service: Annotated[TestService, Depends(get_test_service_without_session)],
) -> Any:
    """Generate mood/gender variants for a batch of phrases using Groq

    :role:
        admin

    :param:
        body: list of phrase+tag+count request items
        test_service: service handling batch variant generation

    :returns:
        results: list of PhraseVariantsResponse or None per input phrase
    """
    phrases = [(item.phrase, item.tag) for item in body]
    count = body[0].count if body else 5
    results = await test_service.generate_variants_batch(phrases=phrases, count=count)
    return results


@router.post(
    "/generate-variants",
    status_code=200,
    response_model=PhraseVariantsResponse | None,
)
@log_decorator(level=logging.INFO)
async def generate_variants(
    body: PhraseVariantsRequest,
    test_service: Annotated[TestService, Depends(get_test_service_without_session)],
) -> Any:
    """Generate mood/gender variants for a single phrase using Groq

    :role:
        admin

    :param:
        body: phrase, tag, and count request payload
        test_service: service handling single phrase variant generation

    :returns:
        response: PhraseVariantsResponse or None if generation failed
    """
    variants = await test_service.generate_variants(  # type: ignore
        phrase=body.phrase,
        tag=body.tag,
        count=body.count,
    )
    if variants is None:
        return None
    return PhraseVariantsResponse(original=body.phrase, tag=body.tag, variants=variants)  # type: ignore


@router.post(
    "/generate-variants-mistral",
    status_code=200,
    response_model=PhraseVariantsResponse | None,
)
@log_decorator(level=logging.INFO)
async def generate_variants_mistral(
    body: PhraseVariantsRequest,
    test_service: Annotated[TestService, Depends(get_test_service_without_session)],
) -> PhraseVariantsResponse | None:
    """Generate mood/gender variants for a single phrase using Mistral

    :role:
        admin

    :param:
        body: phrase, tag, and count request payload
        test_service: service handling Mistral-based variant generation

    :returns:
        response: PhraseVariantsResponse or None if generation failed
    """
    variants = await test_service.generate_variants_mistral(
        phrase=body.phrase,
        tag=body.tag,
        count=body.count,
    )
    if variants is None:
        return None
    return PhraseVariantsResponse(original=body.phrase, tag=body.tag, variants=variants)


@router.post(
    "/generate-variants-batch-mistral",
    status_code=200,
    response_model=list[PhraseVariantsResponse | None],
)
@log_decorator(level=logging.INFO)
async def generate_variants_batch_mistral(
    body: list[PhraseVariantsRequest],
    test_service: Annotated[TestService, Depends(get_test_service_without_session)],
) -> Any:
    """Generate mood/gender variants for a batch of phrases using Mistral

    :role:
        admin

    :param:
        body: list of phrase+tag+count request items
        test_service: service handling Mistral-based batch variant generation

    :returns:
        results: list of PhraseVariantsResponse or None per input phrase
    """
    phrases = [(item.phrase, item.tag) for item in body]
    count = body[0].count if body else 5
    return await test_service.generate_variants_batch_mistral(
        phrases=phrases, count=count
    )


@router.get(
    "",
    status_code=200,
)
@log_decorator(level=logging.INFO)
async def test(
    text: str,
    test_service: Annotated[
        BaseServiceAbstract, Depends(get_test_service_without_session)
    ],
) -> None:
    """Embed a query text and search similar vectors in Qdrant

    :role:
        admin

    :param:
        text: query string to embed and search
        test_service: service wrapping vector similarity search

    :returns:
        results: list of matched vector payloads or None
    """
    return await test_service.check(text)


@router.post(
    "/vision",
    status_code=200,
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
                            "prompt": {"type": "string"},
                        },
                    }
                }
            },
        }
    },
)
@log_decorator(level=logging.INFO)
async def test_vision(
    test_service: Annotated[TestService, Depends(get_test_service_without_session)],
    images: Annotated[list[UploadFile], File()],
) -> str | None:
    """Run raw vision inference on uploaded images and return the model's text output

    :role:
        admin

    :param:
        test_service: service wrapping the Groq vision call
        images: one or more uploaded image files

    :returns:
        raw_text: raw model response string, or None on failure
    """
    images_raw: list[bytes] = []

    try:
        for image in images:
            images_raw.append(await image.read())
    finally:
        for image in images:
            await image.close()
    return await test_service.vision(images_raw=images_raw, prompt=PROMPT)
