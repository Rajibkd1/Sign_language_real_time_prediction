from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_returns_status():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in {"ok", "degraded"}
        assert "ready" in payload


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
