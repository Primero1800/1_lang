from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.logging import logger
from app.core.config import settings

engine = create_async_engine(
    url=str(settings.database_url),
    pool_size=settings.POOL_SIZE,
    max_overflow=settings.MAX_OVERFLOW,
    pool_timeout=settings.POOL_TIMEOUT,
    pool_recycle=settings.POOL_RECYCLE,
    pool_pre_ping=True,
    pool_use_lifo=True,
    connect_args={
        "server_settings": {
            "application_name": settings.APP_NAME,
            "tcp_keepalives_idle": "30",
            "tcp_keepalives_interval": "10",
            "tcp_keepalives_count": "5",
        },
    },
    echo=False,
    echo_pool=False,
)

async_database_session_maker = async_sessionmaker(expire_on_commit=False, bind=engine)


async def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get the global async session maker"""
    return async_database_session_maker


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a new async database session"""
    async with async_database_session_maker() as session:
        try:
            yield session
        except SQLAlchemyError as exc:
            logger.error("Unexpected error during session:", exc_info=exc)
            await session.rollback()
            raise exc
        finally:
            await session.close()


async def initialize_db() -> None:
    """Initialize DB connection pool and verify connectivity"""
    logger.info("Initialize db pool")
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info(
        f"DB initialized: SIZE={settings.POOL_SIZE}, "
        f"OVERFLOW={settings.MAX_OVERFLOW}, TIMEOUT={settings.POOL_TIMEOUT}"
    )


async def shutdown_db() -> None:
    """Shut down the database engine"""
    logger.info("Shutting down db pool")
    await engine.dispose()
