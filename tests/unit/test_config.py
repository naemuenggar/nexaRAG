import pytest
from pydantic import ValidationError

from rag_platform.core.config import Environment, Settings
from tests.conftest import make_settings


def test_valid_settings_load() -> None:
    settings = make_settings()
    assert settings.environment is Environment.TEST
    assert settings.max_upload_size_bytes == 50 * 1024 * 1024
    assert not settings.is_production


def test_jwt_secret_is_required() -> None:
    with pytest.raises(ValidationError, match="jwt_secret"):
        make_settings(jwt_secret=None)


def test_short_jwt_secret_is_rejected() -> None:
    with pytest.raises(ValidationError, match="jwt_secret"):
        make_settings(jwt_secret="too-short")


def test_secrets_are_masked_in_repr() -> None:
    settings = make_settings(jwt_secret="s" * 40, qdrant_api_key="qdrant-key-value")
    assert "s" * 40 not in repr(settings)
    assert "qdrant-key-value" not in repr(settings)


def test_database_url_must_use_asyncpg() -> None:
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        make_settings(database_url="postgresql://user:pass@localhost:5432/rag")


def test_blank_qdrant_api_key_means_none() -> None:
    assert make_settings(qdrant_api_key="").qdrant_api_key is None


def test_log_level_is_normalised() -> None:
    assert make_settings(log_level="debug").log_level == "DEBUG"


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="log level"):
        make_settings(log_level="verbose")


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/rag")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("JWT_SECRET", "e" * 40)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "7")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", '["https://app.example.com"]')

    settings = Settings(_env_file=None)

    assert settings.is_production
    assert settings.max_upload_size_mb == 7
    assert settings.cors_allow_origins == ["https://app.example.com"]
    assert settings.jwt_secret.get_secret_value() == "e" * 40


def test_upload_limit_bounds() -> None:
    with pytest.raises(ValidationError):
        make_settings(max_upload_size_mb=0)


def test_env_example_loads_as_valid_settings() -> None:
    """The shipped .env.example must produce valid settings (blank optional values included)."""
    settings = Settings(_env_file=".env.example")
    assert settings.llm_provider.value == "none"
    assert settings.llm_price_input_per_1k is None
    assert settings.ask_min_rerank_score is None
