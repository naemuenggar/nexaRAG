"""Dependency health probes used by the readiness endpoint."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

HealthProbe = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    name: str
    healthy: bool
    latency_ms: float
    error: str | None = None


def postgres_probe(engine: AsyncEngine) -> HealthProbe:
    async def probe() -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    return probe


def redis_probe(client: Redis) -> HealthProbe:
    async def probe() -> None:
        if not await client.ping():
            raise RuntimeError("PING returned false")

    return probe


def qdrant_probe(client: AsyncQdrantClient) -> HealthProbe:
    async def probe() -> None:
        await client.get_collections()

    return probe


async def _run_probe(name: str, probe: HealthProbe, timeout_seconds: float) -> ComponentHealth:
    started = time.perf_counter()
    try:
        await asyncio.wait_for(probe(), timeout=timeout_seconds)
    except Exception as exc:  # any failure means "unhealthy"
        # Only the exception type is exposed; messages can contain hostnames or credentials.
        error = "timeout" if isinstance(exc, TimeoutError) else type(exc).__name__
        return ComponentHealth(name, False, _elapsed_ms(started), error)
    return ComponentHealth(name, True, _elapsed_ms(started))


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


async def check_all(
    probes: Mapping[str, HealthProbe], timeout_seconds: float
) -> list[ComponentHealth]:
    return list(
        await asyncio.gather(
            *(_run_probe(name, probe, timeout_seconds) for name, probe in probes.items())
        )
    )
