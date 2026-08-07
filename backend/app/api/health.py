from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import HealthResponse


router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = getattr(request.app.state, "inference_service", None)
    error = getattr(request.app.state, "startup_error", None)
    ready = service is not None
    return HealthResponse(
        status="ok" if ready else "degraded",
        ready=ready,
        model_ready=ready,
        error=error,
    )

