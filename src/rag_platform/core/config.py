"""Application settings, loaded from environment variables (and an optional .env file).

Secrets have no defaults: the app refuses to start if they are missing.
"""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator
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

    jwt_secret: SecretStr = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = Field(default=60, ge=1, le=24 * 60)

    max_upload_size_mb: int = Field(default=50, ge=1, le=500)
    cors_allow_origins: list[str] = Field(default_factory=list)

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
