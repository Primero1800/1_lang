from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router as main_router
from app.core.config import settings
from app.lifecycle import AppLifecycle
from app.middleware import MonitoringMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage FastAPI application lifecycle"""
    lifecycle = AppLifecycle(app)
    await lifecycle.on_startup()
    try:
        yield
    finally:
        await lifecycle.on_shutdown()


app = FastAPI(
    title=settings.APP_NAME,
    summary=f"API for {settings.APP_NAME}",
    version=settings.APP_VERSION,
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(MonitoringMiddleware)
app.include_router(main_router)
