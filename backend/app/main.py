from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health, history, inference, model_info
from app.config import get_settings
from app.schemas import ErrorResponse
from app.services.history_service import HistoryService


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bdsl_vision")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.history_service = HistoryService(settings)
    app.state.inference_service = None
    app.state.startup_error = None
    try:
        from app.services.inference_service import InferenceService

        logger.info("Loading Model 6 runtime and MediaPipe services.")
        app.state.inference_service = InferenceService(settings, app.state.history_service)
        logger.info("BdSL Vision backend is ready.")
    except Exception as exc:
        logger.exception("Backend started without a ready inference service.")
        app.state.startup_error = f"{type(exc).__name__}: {exc}"

    try:
        yield
    finally:
        service = getattr(app.state, "inference_service", None)
        if service is not None:
            service.close()


app = FastAPI(
    title="BdSL Vision API",
    description="Bangla Sign Language sentence recognition with Model 6 and word-level CTC decoding.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def max_upload_size_middleware(request: Request, call_next):
    if request.method == "POST" and request.url.path.endswith("/api/inference/video"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_upload_bytes:
            payload = ErrorResponse(
                error="upload_too_large",
                message="The recording is too large. Record a shorter sentence and try again.",
                detail=f"Maximum upload size is {settings.max_upload_mb} MB.",
            )
            data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
            return JSONResponse(status_code=413, content=data)
    return await call_next(request)


app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(model_info.router, prefix=settings.api_prefix)
app.include_router(inference.router, prefix=settings.api_prefix)
app.include_router(history.router, prefix=settings.api_prefix)


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "api": settings.api_prefix,
        "docs": "/docs",
        "model_info": f"{settings.api_prefix}/model/info",
    }
