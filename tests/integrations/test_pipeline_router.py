import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient

# --- /pipeline/w1_upload ---


@pytest.mark.asyncio
async def test_upload_images_success(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_phrase_service_without_session

    mock_phrase_service = AsyncMock()
    mock_phrase_service.upload_images.return_value = {
        "phrases_found": 5,
        "inserted": 4,
        "skipped": 1,
    }
    app.dependency_overrides[get_phrase_service_without_session] = lambda: (
        mock_phrase_service
    )

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
    from app.dependencies.services import get_phrase_service_without_session

    mock_phrase_service = AsyncMock()
    mock_phrase_service.upload_images.return_value = {
        "phrases_found": 0,
        "inserted": 0,
        "skipped": 0,
    }
    app.dependency_overrides[get_phrase_service_without_session] = lambda: (
        mock_phrase_service
    )

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
    from app.dependencies.services import get_phrase_service_without_session

    mock_phrase_service = AsyncMock()
    mock_phrase_service.upload_images.return_value = {
        "phrases_found": 3,
        "inserted": 3,
        "skipped": 0,
    }
    app.dependency_overrides[get_phrase_service_without_session] = lambda: (
        mock_phrase_service
    )

    response = await async_client.post(
        "/pipeline/w1_upload?lang=en",
        files=[("images", ("test.jpg", b"fake image data", "image/jpeg"))],
    )

    assert response.status_code == 201
    mock_phrase_service.upload_images.assert_called_once_with(
        images_raw=[b"fake image data"],
        lang="en",
    )


@pytest.mark.asyncio
async def test_upload_images_no_files_returns_422(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_phrase_service_without_session

    app.dependency_overrides[get_phrase_service_without_session] = lambda: AsyncMock()

    response = await async_client.post("/pipeline/w1_upload")
    assert response.status_code == 422


# --- /pipeline/w2_generate ---


@pytest.mark.asyncio
async def test_w2_generate_success(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_phrase_data_service_without_session

    mock_service = AsyncMock()
    mock_service.w2_generate.return_value = {"processed": 5, "failed": 1, "skipped": 0}
    app.dependency_overrides[get_phrase_data_service_without_session] = lambda: (
        mock_service
    )

    response = await async_client.post("/pipeline/w2_generate")
    assert response.status_code == 200
    data = response.json()
    assert data["processed"] == 5
    assert data["failed"] == 1
    assert data["skipped"] == 0


@pytest.mark.asyncio
async def test_w2_generate_skipped_when_no_batch(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_phrase_data_service_without_session

    mock_service = AsyncMock()
    mock_service.w2_generate.return_value = {"processed": 0, "failed": 0, "skipped": 1}
    app.dependency_overrides[get_phrase_data_service_without_session] = lambda: (
        mock_service
    )

    response = await async_client.post("/pipeline/w2_generate")
    assert response.status_code == 200
    assert response.json()["skipped"] == 1


@pytest.mark.asyncio
async def test_w2_generate_custom_batch_size(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_phrase_data_service_without_session

    mock_service = AsyncMock()
    mock_service.w2_generate.return_value = {"processed": 3, "failed": 0, "skipped": 0}
    app.dependency_overrides[get_phrase_data_service_without_session] = lambda: (
        mock_service
    )

    response = await async_client.post("/pipeline/w2_generate?batch_size=3")
    assert response.status_code == 200
    mock_service.w2_generate.assert_called_once_with(batch_size=3)


# --- /pipeline/w3_translate ---


@pytest.mark.asyncio
async def test_w3_translate_success(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_phrase_translation_service_without_session

    mock_service = AsyncMock()
    mock_service.w3_translate.return_value = {"processed": 4, "failed": 1, "skipped": 0}
    app.dependency_overrides[get_phrase_translation_service_without_session] = lambda: (
        mock_service
    )

    response = await async_client.post("/pipeline/w3_translate")
    assert response.status_code == 200
    data = response.json()
    assert data["processed"] == 4
    assert data["failed"] == 1
    assert data["skipped"] == 0


@pytest.mark.asyncio
async def test_w3_translate_skipped_when_no_batch(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_phrase_translation_service_without_session

    mock_service = AsyncMock()
    mock_service.w3_translate.return_value = {"processed": 0, "failed": 0, "skipped": 1}
    app.dependency_overrides[get_phrase_translation_service_without_session] = lambda: (
        mock_service
    )

    response = await async_client.post("/pipeline/w3_translate")
    assert response.status_code == 200
    assert response.json()["skipped"] == 1


@pytest.mark.asyncio
async def test_w3_translate_custom_batch_size(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_phrase_translation_service_without_session

    mock_service = AsyncMock()
    mock_service.w3_translate.return_value = {"processed": 3, "failed": 0, "skipped": 0}
    app.dependency_overrides[get_phrase_translation_service_without_session] = lambda: (
        mock_service
    )

    response = await async_client.post("/pipeline/w3_translate?batch_size=3")
    assert response.status_code == 200
    mock_service.w3_translate.assert_called_once_with(batch_size=3)


# --- /pipeline/w4_embed ---


@pytest.mark.asyncio
async def test_w4_embed_success(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_phrase_embedding_service_without_session

    mock_service = AsyncMock()
    mock_service.w4_embed.return_value = {"processed": 10, "failed": 1, "skipped": 0}
    app.dependency_overrides[get_phrase_embedding_service_without_session] = lambda: (
        mock_service
    )

    response = await async_client.post("/pipeline/w4_embed")
    assert response.status_code == 200
    data = response.json()
    assert data["processed"] == 10
    assert data["failed"] == 1
    assert data["skipped"] == 0


@pytest.mark.asyncio
async def test_w4_embed_skipped_when_no_batch(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_phrase_embedding_service_without_session

    mock_service = AsyncMock()
    mock_service.w4_embed.return_value = {"processed": 0, "failed": 0, "skipped": 1}
    app.dependency_overrides[get_phrase_embedding_service_without_session] = lambda: (
        mock_service
    )

    response = await async_client.post("/pipeline/w4_embed")
    assert response.status_code == 200
    assert response.json()["skipped"] == 1


@pytest.mark.asyncio
async def test_w4_embed_custom_batch_size(async_client: AsyncClient) -> None:
    from app.main import app
    from app.dependencies.services import get_phrase_embedding_service_without_session

    mock_service = AsyncMock()
    mock_service.w4_embed.return_value = {"processed": 50, "failed": 0, "skipped": 0}
    app.dependency_overrides[get_phrase_embedding_service_without_session] = lambda: (
        mock_service
    )

    response = await async_client.post("/pipeline/w4_embed?batch_size=100")
    assert response.status_code == 200
    mock_service.w4_embed.assert_called_once_with(batch_size=100)
