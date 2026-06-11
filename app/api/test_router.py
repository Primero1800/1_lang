import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.common.logging import log_decorator
from app.dependencies.services import (
    get_test_service_without_session,
)
from app.services.base import BaseServiceAbstract
from app.services.test_service import TestService

router = APIRouter(
    prefix="/test_routes",
    tags=["Test"],
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
    prompt: Annotated[
        str, Form()
    ] = "Опиши подробно что происходит на изображениях: чем занимается человек, что открыто на экране, какое его поведение. Очень подробно надо, придирчиво",
) -> str | None:
    images_raw: list[bytes] = []
    prompt = """Опиши подробно что происходит на изображениях: 
    чем занимается человек, что открыто на экране, какое его поведение. 
    конкретно нужно четкое описание в трех-четырех предложениях по тегам
    1. поведение (как он ебя ведет, что делает, может засыпает или наоборот слишком бодрый), 
    2. внешность (опрятный, причесанный, ухоженный, лысый, косой, хромой, больной), 
    3. возраст (старый, молодой, сопляк, пердун, сосунок), 
    4. настроение (приуныл, ржет, веселый, в петлю полезть готов, 
    5. поза (сидит, раком, как царь, забитый) и тп. по такому принципу.
    Очень подробно надо, придирчиво. 
    При этом каждая фотка должна быть проанализирована по всем этим и иным параметрам в вариациях - 
    A. крайне грубо (аморально), B. очень грубо, 
    C. сносно, D.нейтрально, E.нормально, 
    F.приятно, G.очень  доброжелательно, H.крайне доброжелательно (до тошноты)
    Соотвественно ответы должны быть по каждому фото в виде
    1 фото A1 - "текст на два-три предложения", A2 - ... G5 - ""..
    2 фото A1 - "текст на два-три предложения", A2 - ... G5 - ""..
    т.е для каждого фото должны быть описаны все сценарии - (A1, A2, A3, A4, A5, B1, B2, B3, B4, B5, .... H1, H2, H3, H4, H5)
    пояснение: D.нейтрально - это примерно то, что ты и видишь. Остальные настроения от A до H должны описывать
    то же самое, но согласно своему видению: если настроение A - значит крайне грубо все описать в том чиле аморально можно,
    если настроение G - то максимально возвышенно и хвалебно.
    """
    try:
        for image in images:
            images_raw.append(await image.read())
    finally:
        for image in images:
            await image.close()
    return await test_service.vision(images_raw=images_raw, prompt=prompt)
