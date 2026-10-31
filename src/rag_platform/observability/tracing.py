"""OpenTelemetry tracing.

Spans always exist (so every log line can carry a trace id); they are exported only when
``OTEL_EXPORTER_OTLP_ENDPOINT`` is set. Any OTLP/HTTP backend works: Jaeger, Tempo, Honeycomb,
or Langfuse, which ingests OTLP and shows LLM spans with the ``gen_ai.*`` attributes set here.

Span attributes never include document text, questions or answers, only ids, counts, timings
and model names, so traces can go to a third party without leaking user content.
"""

from collections.abc import MutableMapping
from typing import Any

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from rag_platform import __version__
from rag_platform.core.config import Settings

_configured = False


def configure_tracing(settings: Settings) -> None:
    """Install the global tracer provider once per process."""
    global _configured
    if _configured:
        return
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": settings.otel_service_name,
                "service.version": __version__,
                "deployment.environment": settings.environment.value,
            }
        )
    )
    if settings.otel_exporter_otlp_endpoint:
        endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint += "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _configured = True


def instrument_app(app: FastAPI) -> None:
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health/.*,metrics")


def tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name, __version__)


def add_trace_context(
    _: Any, __: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: attach the current trace and span ids to every log entry."""
    context = trace.get_current_span().get_span_context()
    if context.is_valid:
        event_dict["trace_id"] = format(context.trace_id, "032x")
        event_dict["span_id"] = format(context.span_id, "016x")
    return event_dict
