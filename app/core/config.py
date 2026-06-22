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
    MISTRAL_TIMEOUT_SEC: int = 60
    MISTRAL_VISION_TIMEOUT_SEC: int = 120

    # [AI] Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    GROQ_TIMEOUT_SEC: int = 30

    QDRANT_HOST: str
    QDRANT_PORT: int
    QDRANT_GRPC_PORT: int
    QDRANT_API_KEY: Annotated[str, Field(validation_alias="QDRANT__SERVICE__API_KEY")]
    QDRANT_PREFER_GRPC: bool = True
    QDRANT_HTTPS: bool = False
    QDRANT_TIMEOUT: int = 60

    QDRANT_MAIN_ENABLED: bool = True
    QDRANT_MAIN_URL: str = ""
    QDRANT_MAIN_API_KEY: str = ""
    QDRANT_MAIN_UPSERT_CHUNK_SIZE: int = 100
    T1_SEARCH_MIN_SCORE: float = 0.85

    VECTOR_DB_COLLECTION: str
    VECTOR_DB_VECTOR_SIZE: int = 1024

    STUCK_THRESHOLD: int = 10

    # [Redis]
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TOKENS_STREAM: str = "stream:tokens"
    REDIS_TOKENS_GROUP: str = "token_workers"
    REDIS_TOKENS_WORKER: str = "main"
    REDIS_TOKENS_BATCH_SIZE: int = 100
    REDIS_TOKENS_POLL_INTERVAL: int = 60

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def database_url(self) -> str:
        """Generate the asyncpg connection string for PostgreSQL

        :returns:
            url: asyncpg-compatible DSN string
        """
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def log_level(self) -> int:
        """Convert string log level to numeric logging level

        :raise:
            ValueError: if LOG_LEVEL is not a recognised level name

        :returns:
            level: numeric logging constant (e.g. logging.INFO)
        """
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
