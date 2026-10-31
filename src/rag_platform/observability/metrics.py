"""Prometheus metrics, served at ``/metrics``.

Label values are bounded (route templates, modes, model names, refusal reasons), never ids or
free text, so cardinality stays small.
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)

HTTP_REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=_LATENCY_BUCKETS,
)
SEARCH_LATENCY = Histogram(
    "rag_search_duration_seconds",
    "Retrieval latency by mode (scope resolution to hydrated results)",
    ["mode"],
    buckets=_LATENCY_BUCKETS,
)
LLM_TOKENS = Counter("rag_llm_tokens_total", "LLM tokens used", ["model", "kind"])
LLM_LATENCY = Histogram(
    "rag_llm_duration_seconds", "LLM call latency", ["model"], buckets=_LATENCY_BUCKETS
)
LLM_COST = Counter("rag_llm_cost_usd_total", "Estimated LLM cost (when prices are set)", ["model"])
ASK_OUTCOMES = Counter(
    "rag_ask_outcomes_total", "Answers by outcome", ["outcome"]
)  # outcome = answered | <refusal reason>
SUSPICIOUS_SOURCES = Counter(
    "rag_suspicious_sources_total", "Retrieved passages flagged as possible prompt injection"
)
RATE_LIMITED = Counter("rag_rate_limited_total", "Requests rejected by rate limiting", ["scope"])


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
