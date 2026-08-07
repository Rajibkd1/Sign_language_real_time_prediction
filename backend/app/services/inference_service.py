from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import numpy as np

from app.config import Settings
from app.errors import MediaPipeFailureError, NoCTCWordsError, RecordingTooShortError
from app.model.checkpoint import ModelRuntime
from app.schemas import ConfidenceInfo, DetectionRates, InferenceResponse, TimingInfo
from app.services.history_service import HistoryService
from app.services.mediapipe_service import MediaPipeService
from app.utils.sequence import resample_landmarks_vectorized


class InferenceService:
    def __init__(self, settings: Settings, history: HistoryService):
        self.settings = settings
        self.history = history
        self.runtime = ModelRuntime.load(settings)
        self.mediapipe = MediaPipeService(settings)

    def close(self) -> None:
        self.mediapipe.close()

    def model_info(self) -> Dict[str, Any]:
        return self.runtime.info(self.settings.max_sequence_length, self.settings.target_fps)

    def predict_video(self, path: Path) -> InferenceResponse:
        total_started = time.perf_counter()
        landmarks = self.mediapipe.extract_from_video(path)

        if landmarks.features.shape[0] < 2:
            raise MediaPipeFailureError("Too few landmark frames were extracted.")

        model_features = resample_landmarks_vectorized(
            landmarks.features,
            landmarks.times_s,
            target_fps=self.settings.target_fps,
            max_seq_len=self.settings.max_sequence_length,
        )

        if len(model_features) < self.settings.min_model_frames:
            raise RecordingTooShortError(detail=f"Only {len(model_features)} model frames were available.")

        prediction = self.runtime.predict(model_features, self.settings.min_model_frames)
        if not prediction.sentence:
            raise NoCTCWordsError()

        total_ms = (time.perf_counter() - total_started) * 1000.0
        result_id = str(uuid4())
        response = InferenceResponse(
            success=True,
            id=result_id,
            prediction=prediction.sentence,
            token_ids=prediction.token_ids,
            input_frames=prediction.input_frames,
            ctc_steps=prediction.output_steps,
            duration_seconds=float(landmarks.duration_seconds),
            timings=TimingInfo(
                video_decode_ms=float(landmarks.video_decode_ms),
                mediapipe_ms=float(landmarks.mediapipe_ms),
                model_ms=float(prediction.model_ms),
                total_ms=float(total_ms),
            ),
            detections=DetectionRates(**landmarks.detections),
            confidence=ConfidenceInfo(value=float(prediction.confidence_value)),
            warnings=[],
        )

        self.history.add(
            {
                "id": result_id,
                "prediction": response.prediction,
                "token_ids": response.token_ids,
                "duration_seconds": response.duration_seconds,
                "total_latency_ms": response.timings.total_ms,
                "detections": response.detections.model_dump()
                if hasattr(response.detections, "model_dump")
                else response.detections.dict(),
                "confidence": response.confidence.model_dump()
                if hasattr(response.confidence, "model_dump")
                else response.confidence.dict(),
            }
        )
        return response

