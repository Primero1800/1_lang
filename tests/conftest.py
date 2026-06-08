import asyncio
import os
from asyncio import AbstractEventLoop
from collections.abc import AsyncGenerator
from typing import Any, Generator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    with PostgresContainer("postgres:15") as postgres:
        url = make_url(postgres.get_connection_url())
        os.environ["POSTGRES_USER"] = str(url.username)
        os.environ["POSTGRES_PASSWORD"] = str(url.password)
        os.environ["POSTGRES_HOST"] = str(url.host)
        os.environ["POSTGRES_PORT"] = str(url.port)
        os.environ["POSTGRES_DB"] = str(url.database)
        yield postgres


@pytest.fixture(scope="session")
def test_engine(postgres_container: PostgresContainer):
    from sqlalchemy.pool import NullPool

    url = make_url(postgres_container.get_connection_url())
    async_url = url.set(drivername="postgresql+asyncpg")
    return create_async_engine(async_url, poolclass=NullPool)


@pytest.fixture(scope="session")
def test_session_maker(test_engine):
    return async_sessionmaker(bind=test_engine, expire_on_commit=False)


@pytest.fixture(scope="session")
def event_loop() -> Generator[AbstractEventLoop, Any, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def empty_db(test_engine) -> AsyncGenerator[None, None]:
    from app.models import Base

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def async_client(test_session_maker) -> AsyncGenerator[AsyncClient, None]:
    from app.main import app
    from app.core.database import get_session
    from app.uow import get_uow, get_uow_factory, UnitOfWork

    async def _override_get_uow():
        async with UnitOfWork(session_factory=test_session_maker) as uow:
            yield uow

    async def _override_get_session():
        async with test_session_maker() as session:
            yield session

    def _override_get_uow_factory():
        return UnitOfWork(session_factory=test_session_maker)

    app.dependency_overrides[get_uow] = _override_get_uow
    app.dependency_overrides[get_session] = _override_get_session
    app.dependency_overrides[get_uow_factory] = _override_get_uow_factory

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
