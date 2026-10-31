"""Request-scoped middleware: request ids, structured access logs, security headers and a
safe 500 response for unhandled exceptions.

Written as pure ASGI middleware (not BaseHTTPMiddleware) so streaming responses and
contextvars behave correctly.
"""

import re
import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_platform.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS

REQUEST_ID_HEADER = "x-request-id"
# Accept a caller-supplied id only if it is short and harmless to log.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
    "cross-origin-opener-policy": "same-origin",
    "permissions-policy": "camera=(), microphone=(), geolocation=()",
}
# The API only returns JSON: nothing may be loaded or framed. The interactive docs page loads
# Swagger UI from a CDN, so it is exempt (and is disabled in production anyway).
_API_CSP = "default-src 'none'; frame-ancestors 'none'"
_DOCS_PATHS = ("/docs", "/openapi.json")
_HSTS = "max-age=31536000; includeSubDomains"

logger = structlog.stdlib.get_logger("rag_platform.access")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, hsts: bool = False) -> None:
        self.app = app
        # Only behind TLS (production): browsers then refuse plain HTTP for a year.
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        started = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
                for name, value in _SECURITY_HEADERS.items():
                    headers.setdefault(name, value)
                if not scope["path"].startswith(_DOCS_PATHS):
                    headers.setdefault("content-security-policy", _API_CSP)
                if self.hsts:
                    headers.setdefault("strict-transport-security", _HSTS)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            logger.exception("unhandled_error", path=scope["path"], error_type=type(exc).__name__)
            if response_started:
                raise  # Too late for a clean error response; let the server close the connection.
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "internal_error",
                        "message": "An internal error occurred",
                        "request_id": request_id,
                    }
                },
            )
            await response(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - started
            # The route template ("/documents/{document_id}"), not the raw path, keeps metric
            # labels bounded; unmatched paths (404 scans) share one label.
            route = getattr(scope.get("route"), "path", "unmatched")
            HTTP_REQUESTS.labels(scope["method"], route, str(status_code)).inc()
            HTTP_LATENCY.labels(scope["method"], route).observe(duration)
            logger.info(
                "request",
                method=scope["method"],
                path=scope["path"],
                status=status_code,
                duration_ms=round(duration * 1000, 2),
            )
            structlog.contextvars.clear_contextvars()


def _incoming_request_id(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name == REQUEST_ID_HEADER.encode():
            candidate = value.decode("latin-1")
            return candidate if _VALID_REQUEST_ID.match(candidate) else None
    return None
