from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    ready: bool
    model_ready: bool
    version: str = "1.0.0"
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model: str
    architecture: str
    parameters: Optional[int]
    vocab_size: Optional[int]
    blank_index: Optional[int]
    weight_source: Optional[str]
    device: Optional[str]
    max_sequence_length: int
    target_fps: float
    input_shape: str = "[T, 1662]"
    output_temporal_formula: str = "floor((T - 1) / 2) + 1"
    ready: bool
    error: Optional[str] = None


class TimingInfo(BaseModel):
    video_decode_ms: float
    mediapipe_ms: float
    model_ms: float
    total_ms: float


class DetectionRates(BaseModel):
    pose_rate: float
    face_rate: float
    left_hand_rate: float
    right_hand_rate: float


class ConfidenceInfo(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    type: str = "uncalibrated_greedy_heuristic"
    label: str = "Uncalibrated confidence estimate"


class InferenceResponse(BaseModel):
    success: bool
    id: str
    prediction: str
    token_ids: List[int]
    input_frames: int
    ctc_steps: int
    duration_seconds: float
    timings: TimingInfo
    detections: DetectionRates
    confidence: ConfidenceInfo
    warnings: List[str] = []


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    message: str
    detail: Optional[str] = None


class HistoryItem(BaseModel):
    id: str
    prediction: str
    token_ids: List[int]
    created_at: str
    duration_seconds: float
    total_latency_ms: float
    detections: DetectionRates
    confidence: ConfidenceInfo


class HistoryResponse(BaseModel):
    items: List[HistoryItem]

