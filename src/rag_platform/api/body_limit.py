"""Reject oversized request bodies while they stream in, before anything parses them.

Starlette's multipart parser spools file uploads of any size to disk, so without this a client
could fill the disk with one request. Both a declared ``Content-Length`` and the bytes actually
received are checked, so omitting the header (chunked encoding) does not bypass the limit.
"""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length" and value.isdigit() and int(value) > self.max_body_bytes:
                await self._too_large(scope)(scope, receive, send)
                return

        received = 0
        response_started = False
        rejected = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes and not response_started:
                    # Answer now, then tell the app the client went away so it stops reading.
                    rejected = True
                    await self._too_large(scope)(scope, receive, send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if rejected:
                return  # the 413 has already been sent; drop whatever the app produces
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, limited_receive, guarded_send)

    def _too_large(self, scope: Scope) -> JSONResponse:
        state = scope.get("state", {})
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "code": "payload_too_large",
                    "message": f"Request body exceeds {self.max_body_bytes} bytes",
                    "request_id": state.get("request_id"),
                }
            },
            headers={"connection": "close"},
        )
