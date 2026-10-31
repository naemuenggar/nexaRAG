"""Fixed-window rate limiting in Redis.

Each (scope, subject, window) gets a counter that expires with the window. INCR and the
first EXPIRE run in one Lua script, so concurrent requests can't leave a counter without expiry.
Subjects are user ids for authenticated endpoints and client IPs for login and registration.

If Redis is unreachable the limiter lets requests through and logs it: an outage of the rate
limiter should not take the whole API down. Set ``RATE_LIMIT_FAIL_OPEN=false`` to refuse instead.
"""

import time
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from rag_platform.core.errors import RateLimitedError, ServiceUnavailableError
from rag_platform.observability.metrics import RATE_LIMITED

logger = structlog.stdlib.get_logger(__name__)

_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


class _RedisLike(Protocol):
    # redis-py's async client returns an awaitable from a plain method.
    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class Limit:
    requests: int
    window_seconds: int


class RateLimiter:
    def __init__(self, redis: _RedisLike, *, enabled: bool = True, fail_open: bool = True) -> None:
        self._redis = redis
        self._enabled = enabled
        self._fail_open = fail_open

    async def check(
        self, scope: str, subject: str, limit: Limit, *, now: float | None = None
    ) -> None:
        """Count one request; raise ``RateLimitedError`` once the window's budget is spent."""
        if not self._enabled:
            return
        current = time.time() if now is None else now
        window = int(current // limit.window_seconds)
        key = f"ratelimit:{scope}:{subject}:{window}"
        try:
            count = int(await self._redis.eval(_SCRIPT, 1, key, limit.window_seconds))
        except Exception as exc:
            logger.warning("rate_limiter_unavailable", scope=scope, error=type(exc).__name__)
            if self._fail_open:
                return
            raise ServiceUnavailableError("Service temporarily unavailable") from exc
        if count > limit.requests:
            RATE_LIMITED.labels(scope).inc()
            retry_after = int((window + 1) * limit.window_seconds - current) + 1
            raise RateLimitedError(retry_after)
