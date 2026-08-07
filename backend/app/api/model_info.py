from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import get_settings
from app.schemas import ModelInfoResponse


router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    service = getattr(request.app.state, "inference_service", None)
    settings = getattr(request.app.state, "settings", get_settings())
    error = getattr(request.app.state, "startup_error", None)
    if service is None:
        return ModelInfoResponse(
            model="Model 6",
            architecture="Lightweight Region-Motion Depthwise Conv-Transformer + Word-Level CTC",
            parameters=None,
            vocab_size=None,
            blank_index=None,
            weight_source=None,
            device=None,
            max_sequence_length=settings.max_sequence_length,
            target_fps=settings.target_fps,
            ready=False,
            error=error,
        )
    return ModelInfoResponse(**service.model_info())

