"""Request and response models. Responses never include storage paths or password hashes."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, EmailStr, Field

from rag_platform.db.models import DocumentStatus, JobStatus
from rag_platform.retrieval.types import RetrievalMode


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 - OAuth2 token type, not a secret
    expires_in: int


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    is_active: bool
    created_at: datetime


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: JobStatus
    attempts: int
    started_at: datetime | None
    finished_at: datetime | None


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    size_bytes: int
    content_sha256: str
    page_count: int | None
    status: DocumentStatus
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class DocumentDetail(DocumentOut):
    latest_job: JobOut | None


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class PageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page_number: int
    text: str
    char_count: int


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_ids: list[uuid.UUID] | None = Field(default=None, min_length=1, max_length=500)
    filename_contains: str | None = Field(default=None, min_length=1, max_length=255)
    uploaded_after: AwareDatetime | None = None
    uploaded_before: AwareDatetime | None = None


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    top_k: int | None = Field(default=None, ge=1)
    filters: SearchFilters | None = None


class SearchHitOut(BaseModel):
    rank: int
    score: float
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_start: int
    page_end: int
    text: str
    scores: dict[str, float]


class SearchResponseOut(BaseModel):
    mode: RetrievalMode
    results: list[SearchHitOut]
    latency_ms: float


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=30)
    filters: SearchFilters | None = None


class CitationOut(BaseModel):
    number: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_start: int
    page_end: int
    snippet: str


class UsageOut(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None


class LatencyOut(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class AskResponse(BaseModel):
    answer: str
    answered: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    sources_considered: int
    suspicious_sources: int
    usage: UsageOut | None
    latency: LatencyOut
