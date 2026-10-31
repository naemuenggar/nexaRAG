"""Application settings, loaded from environment variables (and an optional .env file).

Secrets have no defaults: the app refuses to start if they are missing.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_platform.retrieval.fusion import FusionMethod
from rag_platform.retrieval.types import RetrievalMode


class LLMProviderName(StrEnum):
    NONE = "none"
    OPENAI = "openai"  # any OpenAI-compatible /chat/completions endpoint
    ANTHROPIC = "anthropic"


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
    # Load the model in the background at API startup instead of on the first search.
    embedding_warm_up: bool = True

    bm25_k1: float = Field(default=1.5, gt=0, le=10)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    # Per-process cache of per-user BM25 indexes (least recently used owners are evicted).
    bm25_cache_max_owners: int = Field(default=64, ge=1, le=10_000)
    search_default_top_k: int = Field(default=5, ge=1, le=100)
    search_max_top_k: int = Field(default=50, ge=1, le=200)

    # Hybrid: each retriever contributes this many candidates before fusion.
    hybrid_candidates: int = Field(default=30, ge=1, le=500)
    fusion_method: FusionMethod = FusionMethod.RRF
    rrf_k: int = Field(default=60, ge=1, le=1000)
    # Weighted fusion only: share of the dense score (the rest goes to BM25).
    fusion_dense_weight: float = Field(default=0.5, ge=0, le=1)

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"
    reranker_batch_size: int = Field(default=16, ge=1, le=512)
    # Fused candidates passed to the cross-encoder; its cost grows linearly with this.
    rerank_candidates: int = Field(default=20, ge=1, le=200)
    reranker_warm_up: bool = True

    # "none" disables /ask (it answers 503) until a provider is configured.
    llm_provider: LLMProviderName = LLMProviderName.NONE
    llm_base_url: str = ""
    llm_api_key: SecretStr | None = None
    llm_model: str = ""
    llm_timeout_seconds: float = Field(default=60, gt=0, le=600)
    llm_max_tokens: int = Field(default=700, ge=16, le=8192)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    # USD per 1,000 tokens, for cost reporting only. Leave unset if unknown.
    llm_price_input_per_1k: float | None = Field(default=None, ge=0)
    llm_price_output_per_1k: float | None = Field(default=None, ge=0)

    ask_retrieval_mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    ask_top_k: int = Field(default=6, ge=1, le=30)
    # With reranking: skip the LLM and answer "I don't know" when the best passage's
    # cross-encoder score is below this. Unset = always ask the model.
    ask_min_rerank_score: float | None = None
    # Treat an answer that cites nothing as unsupported and return the refusal instead.
    ask_require_citations: bool = True

    # Earlier messages of a conversation sent to the model with each new question.
    conversation_history_messages: int = Field(default=6, ge=0, le=50)

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
    # The worker touches this file on every loop; container health checks read its age.
    worker_heartbeat_file: Path | None = None

    health_check_timeout_seconds: float = Field(default=2.0, gt=0, le=30)

    rate_limit_enabled: bool = True
    # Let requests through when Redis is down (availability) instead of refusing them.
    rate_limit_fail_open: bool = True
    # Per client IP: brute-force protection for credentials.
    rate_limit_auth_per_minute: int = Field(default=10, ge=1)
    # Per user.
    rate_limit_search_per_minute: int = Field(default=60, ge=1)
    rate_limit_ask_per_minute: int = Field(default=20, ge=1)
    rate_limit_upload_per_hour: int = Field(default=100, ge=1)

    # OpenTelemetry. Unset endpoint = spans are created (for log correlation) but not exported.
    # Standard OTEL_EXPORTER_OTLP_HEADERS is honoured, e.g. for Langfuse's OTLP endpoint.
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "enterprise-rag-platform"
    metrics_enabled: bool = True
    # When set, /metrics requires `Authorization: Bearer <token>`.
    metrics_token: SecretStr | None = None

    @field_validator("database_url")
    @classmethod
    def _require_asyncpg(cls, value: PostgresDsn) -> PostgresDsn:
        if value.scheme != "postgresql+asyncpg":
            raise ValueError("database_url must use the 'postgresql+asyncpg' scheme")
        return value

    @field_validator(
        "qdrant_api_key",
        "llm_api_key",
        "llm_price_input_per_1k",
        "llm_price_output_per_1k",
        "ask_min_rerank_score",
        "metrics_token",
        "otel_exporter_otlp_endpoint",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        # `QDRANT_API_KEY=` in an env file means "not set", not "an empty value".
        return None if value == "" else value

    @model_validator(mode="after")
    def _llm_settings_complete(self) -> "Settings":
        if self.llm_provider is not LLMProviderName.NONE and not (
            self.llm_base_url and self.llm_model
        ):
            raise ValueError("llm_base_url and llm_model are required when llm_provider is set")
        return self

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
