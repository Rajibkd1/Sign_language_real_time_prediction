from __future__ import annotations

import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ConfidenceInfo, DetectionRates, InferenceResponse, TimingInfo


def _fake_response() -> InferenceResponse:
    return InferenceResponse(
        success=True,
        id="live-partial",
        prediction="ma vaat khabe",
        token_ids=[1, 2, 3],
        input_frames=24,
        ctc_steps=12,
        duration_seconds=1.2,
        timings=TimingInfo(video_decode_ms=1.0, mediapipe_ms=2.0, model_ms=3.0, total_ms=6.0),
        detections=DetectionRates(pose_rate=1.0, face_rate=1.0, left_hand_rate=1.0, right_hand_rate=1.0),
        confidence=ConfidenceInfo(value=0.7),
    )


def _tiny_jpeg_b64() -> str:
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def test_health_endpoint_returns_status():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in {"ok", "degraded"}
        assert "ready" in payload


def test_bundled_assets_initialize_a_ready_backend():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["ready"] is True, client.app.state.startup_error


@pytest.mark.parametrize("filename,content_type,status,error", [
    ("bad.txt", "text/plain", 400, "invalid_upload"),
    ("clip.webm", "video/webm", 503, "model_not_ready"),
])
def test_upload_validation_when_model_is_unavailable(monkeypatch, filename, content_type, status, error):
    async def unexpected_store(*args, **kwargs):
        raise AssertionError("Unavailable inference must not store an upload")

    monkeypatch.setattr("app.api.inference.store_upload_to_temp", unexpected_store)
    with TestClient(app) as client:
        original = client.app.state.inference_service
        client.app.state.inference_service = None
        try:
            response = client.post("/api/inference/video", files={"file": (filename, b"input", content_type)})
        finally:
            client.app.state.inference_service = original
    assert response.status_code == status
    assert response.json()["detail"]["error"] == error


def test_model_info_endpoint_returns_runtime_shape_contract():
    with TestClient(app) as client:
        response = client.get("/api/model/info")
        assert response.status_code == 200
        payload = response.json()
        assert payload["model"] == "Model 6"
        assert payload["input_shape"] == "[T, 1662]"
        assert payload["max_sequence_length"] == 384
        if payload["ready"]:
            assert payload["parameters"] == 4_924_163
            assert payload["vocab_size"] == 39
            assert payload["blank_index"] == 0
            assert payload["weight_source"] in {"ema_state", "model_state"}


def test_invalid_upload_is_rejected_cleanly():
    with TestClient(app) as client:
        response = client.post(
            "/api/inference/video",
            files={"file": ("bad.txt", b"not a video", "text/plain")},
        )
        assert response.status_code == 400
        payload = response.json()
        assert payload["detail"]["error"] == "invalid_upload"


def test_inference_can_skip_history_for_live_partial_updates():
    seen_save_history = []

    class FakeInferenceService:
        def predict_video(self, path, *, save_history=True):
            seen_save_history.append(save_history)
            return _fake_response()

    with TestClient(app) as client:
        original = client.app.state.inference_service
        client.app.state.inference_service = FakeInferenceService()
        try:
            response = client.post(
                "/api/inference/video",
                data={"save_history": "false"},
                files={"file": ("partial.webm", b"video bytes", "video/webm")},
            )
        finally:
            client.app.state.inference_service = original

    assert response.status_code == 200
    assert seen_save_history == [False]


def test_frame_inference_decodes_live_frames_and_skips_history():
    seen = []

    class FakeInferenceService:
        def predict_frames(self, frames, *, save_history=False):
            seen.append((len(frames), save_history))
            return _fake_response()

    frame = _tiny_jpeg_b64()
    with TestClient(app) as client:
        original = client.app.state.inference_service
        client.app.state.inference_service = FakeInferenceService()
        try:
            response = client.post(
                "/api/inference/frames",
                json={
                    "save_history": False,
                    "frames": [
                        {"image": frame, "timestamp_ms": 0},
                        {"image": frame, "timestamp_ms": 180},
                    ],
                },
            )
        finally:
            client.app.state.inference_service = original

    assert response.status_code == 200
    assert seen == [(2, False)]


def test_live_inference_returns_session_for_partial_batches():
    seen = []

    class FakeInferenceService:
        def predict_live_session(self, *, session_id, frames, reset=False, predict=True, save_history=False):
            seen.append(
                {
                    "session_id": session_id,
                    "frames": len(frames),
                    "reset": reset,
                    "predict": predict,
                    "save_history": save_history,
                }
            )
            return SimpleNamespace(
                session_id="live-session-1",
                segment_id=0,
                state="idle",
                motion_score=0.0,
                is_final=False,
                buffered_frames=len(frames),
                duration_seconds=0.0,
                result=None,
            )

    frame = _tiny_jpeg_b64()
    with TestClient(app) as client:
        original = client.app.state.inference_service
        client.app.state.inference_service = FakeInferenceService()
        try:
            response = client.post(
                "/api/inference/live",
                json={
                    "session_id": None,
                    "reset": True,
                    "predict": True,
                    "save_history": False,
                    "frames": [{"image": frame, "timestamp_ms": 0}],
                },
            )
        finally:
            client.app.state.inference_service = original

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "live-session-1"
    assert payload["segment_id"] == 0
    assert payload["state"] == "idle"
    assert payload["is_final"] is False
    assert payload["result"] is None
    assert seen == [
        {
            "session_id": None,
            "frames": 1,
            "reset": True,
            "predict": True,
            "save_history": False,
        }
    ]
