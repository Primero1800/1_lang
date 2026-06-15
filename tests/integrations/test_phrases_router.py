import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_images_success(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import (
        get_phrase_service_without_session,
        get_prompt_service,
    )

    mock_phrase_service = AsyncMock()
    mock_phrase_service.upload_images.return_value = {
        "phrases_found": 5,
        "inserted": 4,
        "skipped": 1,
    }
    mock_prompt_service = MagicMock()
    mock_prompt_service.get.return_value = "test prompt"

    app.dependency_overrides[get_phrase_service_without_session] = lambda: (
        mock_phrase_service
    )
    app.dependency_overrides[get_prompt_service] = lambda: mock_prompt_service

    response = await async_client.post(
        "/pipeline/w1_upload",
        files=[("images", ("test.jpg", b"fake image data", "image/jpeg"))],
    )

    assert response.status_code == 201
    data = response.json()
    assert data["phrases_found"] == 5
    assert data["inserted"] == 4
    assert data["skipped"] == 1


@pytest.mark.asyncio
async def test_upload_images_no_phrases_found(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import (
        get_phrase_service_without_session,
        get_prompt_service,
    )

    mock_phrase_service = AsyncMock()
    mock_phrase_service.upload_images.return_value = {
        "phrases_found": 0,
        "inserted": 0,
        "skipped": 0,
    }
    mock_prompt_service = MagicMock()
    mock_prompt_service.get.return_value = "test prompt"

    app.dependency_overrides[get_phrase_service_without_session] = lambda: (
        mock_phrase_service
    )
    app.dependency_overrides[get_prompt_service] = lambda: mock_prompt_service

    response = await async_client.post(
        "/pipeline/w1_upload",
        files=[("images", ("test.jpg", b"fake image data", "image/jpeg"))],
    )

    assert response.status_code == 201
    data = response.json()
    assert data["phrases_found"] == 0
    assert data["inserted"] == 0
    assert data["skipped"] == 0


@pytest.mark.asyncio
async def test_upload_images_with_lang_en(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import (
        get_phrase_service_without_session,
        get_prompt_service,
    )

    mock_phrase_service = AsyncMock()
    mock_phrase_service.upload_images.return_value = {
        "phrases_found": 3,
        "inserted": 3,
        "skipped": 0,
    }
    mock_prompt_service = MagicMock()
    mock_prompt_service.get.return_value = "english test prompt"

    app.dependency_overrides[get_phrase_service_without_session] = lambda: (
        mock_phrase_service
    )
    app.dependency_overrides[get_prompt_service] = lambda: mock_prompt_service

    response = await async_client.post(
        "/pipeline/w1_upload?lang=en",
        files=[("images", ("test.jpg", b"fake image data", "image/jpeg"))],
    )

    assert response.status_code == 201
    mock_prompt_service.get.assert_called_once_with("pixtral_vision", "en")
    mock_phrase_service.upload_images.assert_called_once_with(
        images_raw=[b"fake image data"],
        prompt="english test prompt",
        lang="en",
    )


@pytest.mark.asyncio
async def test_upload_images_no_files_returns_422(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import (
        get_phrase_service_without_session,
        get_prompt_service,
    )

    app.dependency_overrides[get_phrase_service_without_session] = lambda: AsyncMock()
    app.dependency_overrides[get_prompt_service] = lambda: MagicMock()

    response = await async_client.post("/pipeline/w1_upload")
    assert response.status_code == 422
