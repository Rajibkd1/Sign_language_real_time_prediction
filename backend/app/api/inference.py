from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.errors import BdSLVisionError, InvalidUploadError, ModelNotReadyError
from app.schemas import ErrorResponse, FrameInferenceRequest, InferenceResponse, LiveInferenceRequest, LiveInferenceResponse
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
async def infer_video(
    request: Request,
    file: UploadFile = File(...),
    save_history: bool = Form(True),
) -> InferenceResponse:
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise _http_error(ModelNotReadyError(getattr(request.app.state, "startup_error", None)))

    settings = request.app.state.settings
    stored = None
    try:
        stored = await store_upload_to_temp(file, settings)
        return await run_in_threadpool(service.predict_video, stored.path, save_history=save_history)
    except BdSLVisionError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        raise _http_error(BdSLVisionError("Recognition failed.", detail=str(exc))) from exc
    finally:
        if stored is not None:
            delete_temp_files([stored.path])


@router.post(
    "/frames",
    response_model=InferenceResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def infer_frames(request: Request, payload: FrameInferenceRequest) -> InferenceResponse:
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise _http_error(ModelNotReadyError(getattr(request.app.state, "startup_error", None)))

    try:
        frames = _decode_live_frames(payload, require_minimum=True)
        return await run_in_threadpool(service.predict_frames, frames, save_history=payload.save_history)
    except BdSLVisionError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        raise _http_error(BdSLVisionError("Recognition failed.", detail=str(exc))) from exc


@router.post(
    "/live",
    response_model=LiveInferenceResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def infer_live(request: Request, payload: LiveInferenceRequest) -> LiveInferenceResponse:
    service = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise _http_error(ModelNotReadyError(getattr(request.app.state, "startup_error", None)))

    try:
        frames = _decode_live_frames(payload, require_minimum=False)
        live = await run_in_threadpool(
            service.predict_live_session,
            session_id=payload.session_id,
            frames=frames,
            reset=payload.reset,
            predict=payload.predict,
            save_history=payload.save_history,
        )
        return LiveInferenceResponse(
            session_id=live.session_id,
            segment_id=live.segment_id,
            state=live.state,
            motion_score=live.motion_score,
            capture_fps=getattr(live, "capture_fps", 0.0),
            detections=getattr(live, "detections", None)
            or {"pose_rate": 0.0, "face_rate": 0.0, "left_hand_rate": 0.0, "right_hand_rate": 0.0},
            is_final=live.is_final,
            buffered_frames=live.buffered_frames,
            duration_seconds=live.duration_seconds,
            result=live.result,
        )
    except BdSLVisionError as exc:
        raise _http_error(exc) from exc
    except Exception as exc:
        raise _http_error(BdSLVisionError("Recognition failed.", detail=str(exc))) from exc


def _decode_live_frames(payload: FrameInferenceRequest, *, require_minimum: bool = True):
    frames = []
    total_bytes = 0
    for item in payload.frames:
        raw = item.image
        if "," in raw and raw[:64].lower().startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise InvalidUploadError("Invalid live frame encoding.", detail="Frame image must be base64.") from exc

        total_bytes += len(image_bytes)
        if total_bytes > 8 * 1024 * 1024:
            raise InvalidUploadError("Live frame payload is too large.")

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise InvalidUploadError("Live frame could not be decoded.")
        frames.append((bgr, item.timestamp_ms / 1000.0))

    if require_minimum and len(frames) < 2:
        raise InvalidUploadError("At least two live frames are required.")
    return frames


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
