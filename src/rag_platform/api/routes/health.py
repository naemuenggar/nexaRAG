from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from rag_platform import __version__
from rag_platform.api.deps import ContainerDep
from rag_platform.infrastructure.health import check_all

router = APIRouter(prefix="/health", tags=["health"])


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ComponentStatus(BaseModel):
    healthy: bool
    latency_ms: float
    error: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    components: dict[str, ComponentStatus]


@router.get("/live", response_model=LivenessResponse)
async def live() -> LivenessResponse:
    """The process is up. Never touches dependencies, so orchestrators don't restart on outages."""
    return LivenessResponse(version=__version__)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def ready(container: ContainerDep, response: Response) -> ReadinessResponse:
    """All dependencies answer within the timeout; otherwise 503 so traffic is withheld."""
    results = await check_all(
        container.health_probes, container.settings.health_check_timeout_seconds
    )
    all_healthy = all(r.healthy for r in results)
    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if all_healthy else "unavailable",
        components={
            r.name: ComponentStatus(healthy=r.healthy, latency_ms=r.latency_ms, error=r.error)
            for r in results
        },
    )
