import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_platform.core.errors import RateLimitedError, ServiceUnavailableError
from rag_platform.core.rate_limit import Limit, RateLimiter
from rag_platform.main import create_app
from rag_platform.observability.tracing import add_trace_context
from tests.conftest import make_settings
from tests.fakes import ScriptedLLM
from tests.unit.test_generation import HITS, service

# --- rate limiting -------------------------------------------------------------------------


class FakeRedis:
    """Implements just the INCR-with-expiry script the limiter sends."""

    def __init__(self, fail: bool = False) -> None:
        self.counts: dict[str, int] = {}
        self.fail = fail

    async def eval(self, script: str, numkeys: int, *args: Any) -> int:
        if self.fail:
            raise ConnectionError("redis down")
        key = args[0]
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]


async def test_limiter_allows_budget_then_rejects_with_retry_after() -> None:
    limiter = RateLimiter(FakeRedis())
    limit = Limit(requests=3, window_seconds=60)
    for _ in range(3):
        await limiter.check("ask", "u1", limit, now=1000.0)
    with pytest.raises(RateLimitedError) as info:
        await limiter.check("ask", "u1", limit, now=1010.0)
    assert info.value.headers == {"Retry-After": "11"}  # window [960, 1020) ends in 10 s, +1
    # Other subjects and scopes have their own budget.
    await limiter.check("ask", "u2", limit, now=1010.0)
    await limiter.check("search", "u1", limit, now=1010.0)


async def test_limiter_resets_in_the_next_window() -> None:
    limiter = RateLimiter(FakeRedis())
    limit = Limit(requests=1, window_seconds=60)
    await limiter.check("ask", "u1", limit, now=0.0)
    with pytest.raises(RateLimitedError):
        await limiter.check("ask", "u1", limit, now=59.0)
    await limiter.check("ask", "u1", limit, now=60.0)


async def test_limiter_fails_open_or_closed_as_configured() -> None:
    limit = Limit(1, 60)
    await RateLimiter(FakeRedis(fail=True)).check("ask", "u", limit)
    with pytest.raises(ServiceUnavailableError):
        await RateLimiter(FakeRedis(fail=True), fail_open=False).check("ask", "u", limit)


async def test_disabled_limiter_never_counts() -> None:
    redis = FakeRedis()
    await RateLimiter(redis, enabled=False).check("ask", "u", Limit(1, 60))
    assert redis.counts == {}


def test_auth_endpoint_returns_429_with_retry_after(client: TestClient) -> None:
    container = client.app.state.container  # type: ignore[attr-defined]
    container.rate_limiter = RateLimiter(FakeRedis())
    container.settings = container.settings.model_copy(update={"rate_limit_auth_per_minute": 2})
    body = {"email": "x@example.com", "password": "whatever-password"}
    statuses = [client.post("/auth/login", json=body).status_code for _ in range(3)]
    # The first two reach the endpoint (and fail on the database stub); the third is limited.
    assert statuses[2] == 429
    limited = client.post("/auth/login", json=body)
    assert limited.json()["error"]["code"] == "rate_limited"
    assert int(limited.headers["retry-after"]) > 0


# --- metrics -------------------------------------------------------------------------------


def test_metrics_endpoint_exposes_request_counters(client: TestClient) -> None:
    client.get("/health/live")
    body = client.get("/metrics").text
    assert "rag_http_requests_total" in body
    assert 'route="/health/live"' in body
    assert "rag_llm_tokens_total" in body


def test_unmatched_paths_share_one_metric_label(client: TestClient) -> None:
    client.get(f"/scan/{uuid.uuid4()}")
    body = client.get("/metrics").text
    assert 'route="unmatched"' in body
    assert "/scan/" not in body


def test_metrics_token_is_enforced() -> None:
    app = create_app(make_settings(metrics_token="m" * 20))
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        ok = client.get("/metrics", headers={"Authorization": "Bearer " + "m" * 20})
        assert ok.status_code == 200


def test_metrics_can_be_disabled() -> None:
    with TestClient(create_app(make_settings(metrics_enabled=False))) as client:
        assert client.get("/metrics").status_code == 404


# --- security headers ----------------------------------------------------------------------


def test_api_responses_carry_csp_but_docs_do_not(client: TestClient) -> None:
    api = client.get("/health/live").headers
    assert api["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert api["permissions-policy"].startswith("camera=()")
    assert "strict-transport-security" not in api
    assert "content-security-policy" not in client.get("/docs").headers


def test_hsts_in_production() -> None:
    with TestClient(create_app(make_settings(environment="production"))) as client:
        assert "max-age=31536000" in client.get("/health/live").headers["strict-transport-security"]


# --- tracing -------------------------------------------------------------------------------


@pytest.fixture
def spans() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):  # first test in the process
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


async def test_llm_span_has_genai_usage_and_no_user_content(spans: InMemorySpanExporter) -> None:
    question = "What was revenue in 2024 for project zebra-secret?"
    await service(HITS, ScriptedLLM("Revenue was 4.2 million dollars in 2024 [2].")).ask(
        uuid.uuid4(), question
    )
    llm_span = next(s for s in spans.get_finished_spans() if s.name == "llm.generate")
    attributes = dict(llm_span.attributes or {})
    assert attributes["gen_ai.request.model"] == "scripted-llm"
    assert int(attributes["gen_ai.usage.input_tokens"]) > 0  # type: ignore[arg-type]
    assert int(attributes["gen_ai.usage.output_tokens"]) > 0  # type: ignore[arg-type]
    assert attributes["rag.sources"] == 2
    # Questions, passages and answers never go into span attributes.
    assert not any("zebra-secret" in str(v) or "4.2 million" in str(v) for v in attributes.values())


def test_log_entries_get_trace_ids_inside_a_span(spans: InMemorySpanExporter) -> None:
    with trace.get_tracer("test").start_as_current_span("unit"):
        event = add_trace_context(None, "info", {"event": "x"})
    assert len(event["trace_id"]) == 32
    assert len(event["span_id"]) == 16
    assert "trace_id" not in add_trace_context(None, "info", {"event": "outside"})


def test_private_network_preflight_is_allowed_for_cors_origins() -> None:
    app = create_app(make_settings(cors_allow_origins=["https://site.example"]))
    with TestClient(app) as client:
        headers = {
            "Origin": "https://site.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
            "Access-Control-Request-Private-Network": "true",
        }
        allowed = client.options("/auth/login", headers=headers)
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-private-network"] == "true"
        other = client.options("/auth/login", headers={**headers, "Origin": "https://evil.example"})
        assert other.status_code == 400  # origin not allowed: the browser blocks the request
        assert "access-control-allow-origin" not in other.headers
