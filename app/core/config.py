import logging
from typing import Annotated
from pydantic import Field

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    APP_NAME: str
    APP_VERSION: str

    POSTGRES_HOST: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    POOL_SIZE: int
    MAX_OVERFLOW: int
    POOL_TIMEOUT: int = 30
    POOL_RECYCLE: int = 1800

    LOG_LEVEL: str

    HEALTH_CHECK_TIMEOUT_SEC: int = 5

    AIOHTTP_KEEPALIVE_TIMEOUT: int = 30
    AIOHTTP_TIMEOUT_SECONDS: int = 30

    # [AI] Mistral
    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL: str = "mistral-small-latest"
    MISTRAL_EMBED_MODEL: str = "mistral-embed"
    MISTRAL_TIMEOUT_SEC: int = 30

    QDRANT_HOST: str
    QDRANT_PORT: int
    QDRANT_GRPC_PORT: int
    QDRANT_API_KEY: Annotated[str, Field(validation_alias="QDRANT__SERVICE__API_KEY")]
    VECTOR_DB_COLLECTION: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def database_url(self) -> str:
        """Generate the asyncpg connection string for PostgreSQL"""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def log_level(self) -> int:
        """Convert string log level to numeric logging level"""
        level = self.LOG_LEVEL.upper()
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if level not in levels:
            raise ValueError(f"Invalid log level: {level}")
        return levels[level]


settings = Settings()  # type: ignore[call-arg]
