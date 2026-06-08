import pytest
from httpx import AsyncClient

from app.core.config import settings


@pytest.mark.asyncio
async def test_get_version(async_client: AsyncClient) -> None:
    response = await async_client.get("/version")
    assert response.status_code == 200
    assert settings.APP_NAME in response.json()
    assert settings.APP_VERSION in response.json()
