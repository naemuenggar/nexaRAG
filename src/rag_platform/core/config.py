"""Application settings, loaded from environment variables (and an optional .env file).

Secrets have no defaults: the app refuses to start if they are missing.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "enterprise-rag-platform"
    environment: Environment = Environment.LOCAL
    log_level: str = "INFO"
    log_json: bool = True

    database_url: PostgresDsn
    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_echo: bool = False

    redis_url: RedisDsn

    qdrant_url: str
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(default="document_chunks", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    qdrant_upsert_batch_size: int = Field(default=128, ge=1, le=2048)

    # Chunk sizes are counted in whitespace-separated words, which keeps chunking independent of
    # the embedding model. 300 words stays under the 512-token limit of BERT-style encoders.
    chunk_size_words: int = Field(default=300, ge=20, le=2000)
    chunk_overlap_words: int = Field(default=50, ge=0, le=1000)
    # False keeps every chunk on a single page, so each citation points at exactly one page.
    chunk_cross_pages: bool = False

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1, le=1024)
    # BGE models expect this instruction on queries (not on passages).
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    jwt_secret: SecretStr = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = Field(default=60, ge=1, le=24 * 60)

    max_upload_size_mb: int = Field(default=50, ge=1, le=500)
    cors_allow_origins: list[str] = Field(default_factory=list)

    storage_dir: Path = Path("data/uploads")
    max_pdf_pages: int = Field(default=2000, ge=1, le=100_000)

    ingestion_poll_interval_seconds: float = Field(default=1.0, gt=0, le=60)
    ingestion_max_attempts: int = Field(default=3, ge=1, le=20)
    # A RUNNING job whose worker has not finished within this window is assumed dead and retried.
    ingestion_lease_seconds: int = Field(default=600, ge=10, le=24 * 3600)

    health_check_timeout_seconds: float = Field(default=2.0, gt=0, le=30)

    @field_validator("database_url")
    @classmethod
    def _require_asyncpg(cls, value: PostgresDsn) -> PostgresDsn:
        if value.scheme != "postgresql+asyncpg":
            raise ValueError("database_url must use the 'postgresql+asyncpg' scheme")
        return value

    @field_validator("qdrant_api_key", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        # `QDRANT_API_KEY=` in an env file means "no key", not "an empty key".
        return None if value == "" else value

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        return self

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        upper = value.upper()
        if upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level: {value}")
        return upper

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION


@lru_cache
def get_settings() -> Settings:
    return Settings()
