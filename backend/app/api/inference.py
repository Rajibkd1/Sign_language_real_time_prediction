from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.errors import BdSLVisionError, ModelNotReadyError
from app.schemas import ErrorResponse, InferenceResponse
from app.services.video_service import delete_temp_files, store_upload_to_temp


router = APIRouter(prefix="/inference", tags=["inference"])


@router.post(
    "/video",
    response_model=InferenceResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def infer_video(request: Request, file: UploadFile = File(...)) -> InferenceResponse:
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise _http_error(ModelNotReadyError(getattr(request.app.state, "startup_error", None)))

    settings = request.app.state.settings
    stored = None
    try:
        stored = await store_upload_to_temp(file, settings)
        return await run_in_threadpool(service.predict_video, stored.path)
    except BdSLVisionError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        raise _http_error(BdSLVisionError("Recognition failed.", detail=str(exc))) from exc
    finally:
        if stored is not None:
            delete_temp_files([stored.path])


def _http_error(exc: BdSLVisionError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "success": False,
            "error": exc.error_code,
            "message": exc.user_message,
            "detail": exc.detail or str(exc),
        },
    )

