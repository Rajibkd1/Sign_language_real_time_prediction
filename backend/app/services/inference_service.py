from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from app.config import Settings
from app.errors import MediaPipeFailureError, NoCTCWordsError, RecordingTooShortError
from app.model.checkpoint import ModelRuntime
from app.schemas import ConfidenceInfo, DetectionRates, InferenceResponse, TimingInfo
from app.services.history_service import HistoryService
from app.services.mediapipe_service import DetectionFlags, LandmarkExtractionResult, MediaPipeService, detection_summary
from app.utils.sequence import resample_landmarks_vectorized


def _empty_detection_rates() -> Dict[str, float]:
    return {"pose_rate": 0.0, "face_rate": 0.0, "left_hand_rate": 0.0, "right_hand_rate": 0.0}


@dataclass
class LiveSessionState:
    features: List[np.ndarray] = field(default_factory=list)
    times_s: List[float] = field(default_factory=list)
    flags: List[DetectionFlags] = field(default_factory=list)
    frame_mediapipe_ms: List[float] = field(default_factory=list)
    watch_features: List[np.ndarray] = field(default_factory=list)
    watch_times_s: List[float] = field(default_factory=list)
    watch_flags: List[DetectionFlags] = field(default_factory=list)
    watch_frame_mediapipe_ms: List[float] = field(default_factory=list)
    preroll: List[Tuple[np.ndarray, float, DetectionFlags, float]] = field(default_factory=list)
    segment_index: int = 0
    state: str = "idle"
    last_feature: Optional[np.ndarray] = None
    last_timestamp_s: Optional[float] = None
    last_motion_score: float = 0.0
    idle_motion_ema: float = 0.0
    high_motion_frames: int = 0
    active_motion_frames: int = 0
    motion_score_sum: float = 0.0
    max_motion_score: float = 0.0
    low_motion_since_s: Optional[float] = None
    last_active_s: Optional[float] = None
    mediapipe_ms: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class LivePrediction:
    session_id: str
    segment_id: int
    buffered_frames: int
    duration_seconds: float
    state: str = "idle"
    motion_score: float = 0.0
    capture_fps: float = 0.0
    detections: Dict[str, float] = field(default_factory=_empty_detection_rates)
    is_final: bool = False
    result: Optional[InferenceResponse] = None


class InferenceService:
    def __init__(self, settings: Settings, history: HistoryService):
        self.settings = settings
        self.history = history
        self.runtime = ModelRuntime.load(settings)
        self.mediapipe = MediaPipeService(settings)
        self._live_sessions: Dict[str, LiveSessionState] = {}
        self._live_lock = threading.Lock()

    def close(self) -> None:
        self.mediapipe.close()

    def model_info(self) -> Dict[str, Any]:
        return self.runtime.info(self.settings.max_sequence_length, self.settings.target_fps)

    def predict_video(self, path: Path, *, save_history: bool = True) -> InferenceResponse:
        total_started = time.perf_counter()
        landmarks = self.mediapipe.extract_from_video(path)
        primary_landmarks = self._trim_landmarks_to_active_sentence(landmarks)
        candidate_landmarks = [primary_landmarks]
        if primary_landmarks is not landmarks:
            candidate_landmarks.append(landmarks)

        responses: List[InferenceResponse] = []
        first_error: Optional[Exception] = None
        for candidate in candidate_landmarks:
            try:
                warnings = (
                    ["Upload was decoded from the active signing region; inactive lead/trail frames were ignored."]
                    if candidate is not landmarks
                    else []
                )
                response = self._predict_landmarks(
                    candidate,
                    total_started,
                    save_history=False,
                    warnings=warnings,
                    artifact_source="video_upload_active" if candidate is not landmarks else "video_upload_full",
                )
                responses.append(response)
            except (MediaPipeFailureError, NoCTCWordsError, RecordingTooShortError) as exc:
                if first_error is None:
                    first_error = exc

        if not responses:
            if first_error is not None:
                raise first_error
            raise NoCTCWordsError()

        response = self._choose_video_response(responses)
        if self._decoded_word_count(response) <= 1 and landmarks.duration_seconds >= 1.5:
            response.warnings.append(
                "Only one CTC word was decoded from this multi-second upload; keep the whole signer visible and avoid extra idle time around the sentence."
            )
        if save_history:
            self._save_history_response(response)
        return response

    def predict_frames(self, frames, *, save_history: bool = False) -> InferenceResponse:
        total_started = time.perf_counter()
        landmarks = self.mediapipe.extract_from_frames(frames)
        return self._predict_landmarks(
            landmarks,
            total_started,
            save_history=save_history,
            artifact_source="frame_batch",
        )

    def predict_live_session(
        self,
        *,
        session_id: str | None,
        frames: Sequence[Tuple[np.ndarray, float]],
        reset: bool = False,
        predict: bool = True,
        save_history: bool = False,
    ) -> LivePrediction:
        self._cleanup_live_sessions()
        sid = session_id or str(uuid4())
        if reset:
            with self._live_lock:
                self._live_sessions.pop(sid, None)

        batch = None
        if frames:
            try:
                batch = self.mediapipe.extract_live_frame_batch(frames)
            except MediaPipeFailureError:
                batch = None

        final_landmarks = None
        final_segment_id: Optional[int] = None
        draft_landmarks = None
        watch_landmarks = None
        prediction_segment_id = 0
        detections = _empty_detection_rates()
        capture_fps = 0.0
        with self._live_lock:
            state = self._live_sessions.setdefault(sid, LiveSessionState())
            if batch is not None:
                final_landmarks, final_segment_id = self._ingest_live_batch(state, batch)
                state.updated_at = time.time()

            if final_landmarks is not None:
                buffered_frames = final_landmarks.sampled_frames
                duration_seconds = final_landmarks.duration_seconds
                state_name = "finalizing"
                prediction_segment_id = int(final_segment_id or state.segment_index)
                detections = dict(final_landmarks.detections)
                capture_fps = self._landmarks_capture_fps(final_landmarks)
            else:
                buffered_frames = len(state.features)
                duration_seconds = self._live_duration(state)
                state_name = state.state
                prediction_segment_id = int(state.segment_index)
                detections = self._live_state_detections(state)
                capture_fps = self._live_state_capture_fps(state)

            if not predict and final_landmarks is None:
                return LivePrediction(
                    sid,
                    prediction_segment_id,
                    buffered_frames,
                    duration_seconds,
                    state=state_name,
                    motion_score=state.last_motion_score,
                    capture_fps=capture_fps,
                    detections=detections,
                )

            if final_landmarks is None and state.state == "signing" and self._segment_can_predict_draft(state):
                try:
                    draft_landmarks = self._landmarks_from_live_state(state)
                except (MediaPipeFailureError, RecordingTooShortError):
                    draft_landmarks = None

            if final_landmarks is None and draft_landmarks is None and self._watch_window_can_predict(state):
                try:
                    watch_landmarks = self._landmarks_from_watch_state(state)
                    buffered_frames = watch_landmarks.sampled_frames
                    duration_seconds = watch_landmarks.duration_seconds
                    detections = dict(watch_landmarks.detections)
                    capture_fps = self._landmarks_capture_fps(watch_landmarks)
                except (MediaPipeFailureError, RecordingTooShortError):
                    watch_landmarks = None

            motion_score = state.last_motion_score

        landmarks = final_landmarks or draft_landmarks or watch_landmarks
        if landmarks is None:
            return LivePrediction(
                sid,
                prediction_segment_id,
                buffered_frames,
                duration_seconds,
                state=state_name,
                motion_score=motion_score,
                capture_fps=capture_fps,
                detections=detections,
            )

        is_final = final_landmarks is not None
        is_watch_fallback = watch_landmarks is not None and final_landmarks is None and draft_landmarks is None
        try:
            if is_final or is_watch_fallback:
                response = self._predict_live_final_landmarks(
                    landmarks,
                    time.perf_counter(),
                    save_history=bool(save_history and is_final),
                )
            else:
                response = self._predict_landmarks(
                    landmarks,
                    time.perf_counter(),
                    save_history=False,
                )
        except (MediaPipeFailureError, NoCTCWordsError, RecordingTooShortError):
            return LivePrediction(
                sid,
                prediction_segment_id,
                buffered_frames,
                duration_seconds,
                state="result" if is_final else state_name,
                motion_score=motion_score,
                capture_fps=self._landmarks_capture_fps(landmarks),
                detections=dict(landmarks.detections),
                is_final=is_final,
            )

        response.timings.mediapipe_ms = float(landmarks.mediapipe_ms)
        response.timings.total_ms = float(response.timings.mediapipe_ms + response.timings.model_ms)
        response_detections = (
            response.detections.model_dump()
            if hasattr(response.detections, "model_dump")
            else response.detections.dict()
        )
        return LivePrediction(
            sid,
            prediction_segment_id,
            buffered_frames,
            duration_seconds,
            state="result" if is_final else "signing" if is_watch_fallback else state_name,
            motion_score=motion_score,
            capture_fps=self._landmarks_capture_fps(landmarks),
            detections=response_detections,
            is_final=is_final,
            result=response,
        )

    def clear_live_session(self, session_id: str) -> None:
        with self._live_lock:
            self._live_sessions.pop(session_id, None)

    @staticmethod
    def _capture_fps_from_times(times_s: Sequence[float]) -> float:
        if len(times_s) < 2:
            return 0.0
        duration = max(0.0, float(max(times_s) - min(times_s)))
        if duration <= 1e-6:
            return 0.0
        return float((len(times_s) - 1) / duration)

    @classmethod
    def _landmarks_capture_fps(cls, landmarks: LandmarkExtractionResult) -> float:
        return cls._capture_fps_from_times(landmarks.times_s)

    def _live_state_detections(self, state: LiveSessionState) -> Dict[str, float]:
        if state.state == "signing" and state.flags:
            return detection_summary(state.flags)
        if state.watch_flags:
            return detection_summary(state.watch_flags)
        return _empty_detection_rates()

    def _live_state_capture_fps(self, state: LiveSessionState) -> float:
        if state.state == "signing" and state.times_s:
            return self._capture_fps_from_times(state.times_s)
        if state.watch_times_s:
            return self._capture_fps_from_times(state.watch_times_s)
        return 0.0

    def _cleanup_live_sessions(self) -> None:
        cutoff = time.time() - 300
        with self._live_lock:
            expired = [sid for sid, state in self._live_sessions.items() if state.updated_at < cutoff]
            for sid in expired:
                self._live_sessions.pop(sid, None)

    def _ingest_live_batch(
        self,
        state: LiveSessionState,
        batch,
    ) -> Tuple[Optional[LandmarkExtractionResult], Optional[int]]:
        final_landmarks = None
        final_segment_id = None
        for feature, timestamp_s, detection, frame_mp_ms in zip(
            batch.features,
            batch.times_s,
            batch.flags,
            batch.mediapipe_ms_by_frame,
        ):
            timestamp = float(timestamp_s)
            motion = self._motion_components(feature, state.last_feature)
            motion_score = motion["score"]
            hand_motion = motion["hand"]
            arm_motion = motion["arms"]
            state.last_motion_score = motion_score
            state.last_feature = np.asarray(feature, dtype=np.float32)
            state.last_timestamp_s = timestamp

            self._push_preroll(state, feature, timestamp, detection, float(frame_mp_ms))
            self._append_watch_frame(state, feature, timestamp, detection, float(frame_mp_ms))
            has_hand_signal = detection.left_hand or detection.right_hand
            has_pose_signal = detection.pose
            start_threshold = self._effective_start_threshold(state)
            active_threshold = self._effective_active_threshold(state)
            hand_start = has_hand_signal and hand_motion >= self.settings.live_min_hand_motion and motion_score >= start_threshold
            pose_start = (
                not has_hand_signal
                and has_pose_signal
                and arm_motion >= self.settings.live_min_hand_motion * 2.2
                and motion_score >= start_threshold * 1.15
            )
            hand_active = (
                has_hand_signal
                and hand_motion >= self.settings.live_min_hand_motion * 0.55
                and motion_score >= active_threshold
            )
            pose_active = (
                not has_hand_signal
                and has_pose_signal
                and arm_motion >= self.settings.live_min_hand_motion * 1.2
                and motion_score >= active_threshold * 1.1
            )
            high_motion = hand_start or pose_start
            active_motion = hand_active or pose_active

            if state.state in {"idle", "result"}:
                if high_motion:
                    state.high_motion_frames += 1
                    if state.high_motion_frames >= self.settings.live_start_consecutive_frames:
                        self._start_live_segment(state)
                        state.last_active_s = timestamp
                        state.active_motion_frames = 1
                        state.motion_score_sum = motion_score
                        state.max_motion_score = motion_score
                else:
                    state.high_motion_frames = max(0, state.high_motion_frames - 1)
                    if self._looks_like_idle_noise(state, motion_score, hand_motion, arm_motion):
                        self._update_idle_motion_baseline(state, motion_score)
                continue

            if state.state != "signing":
                continue

            self._append_live_frame(state, feature, timestamp, detection, float(frame_mp_ms))
            self._trim_live_state(state)

            if active_motion:
                state.last_active_s = timestamp
                state.active_motion_frames += 1
                state.motion_score_sum += motion_score
                state.max_motion_score = max(state.max_motion_score, motion_score)
                state.low_motion_since_s = None
            else:
                state.low_motion_since_s = state.low_motion_since_s if state.low_motion_since_s is not None else timestamp

            if self._segment_can_finalize(state) and state.low_motion_since_s is not None:
                low_duration = timestamp - state.low_motion_since_s
                if low_duration >= self.settings.live_end_silence_seconds:
                    if self._segment_has_meaningful_activity(state):
                        final_landmarks = self._landmarks_from_live_state(state)
                        final_segment_id = int(state.segment_index)
                    self._finish_live_segment(state)
                    if final_landmarks is not None:
                        break

        return final_landmarks, final_segment_id

    def _append_watch_frame(
        self,
        state: LiveSessionState,
        feature: np.ndarray,
        timestamp_s: float,
        detection: DetectionFlags,
        frame_mp_ms: float,
    ) -> None:
        if state.watch_times_s and timestamp_s <= state.watch_times_s[-1] + 1e-9:
            return
        state.watch_features.append(np.asarray(feature, dtype=np.float32))
        state.watch_times_s.append(timestamp_s)
        state.watch_flags.append(detection)
        state.watch_frame_mediapipe_ms.append(frame_mp_ms)
        self._trim_watch_state(state)

    def _trim_watch_state(self, state: LiveSessionState) -> None:
        if not state.watch_times_s:
            return
        latest = max(state.watch_times_s)
        cutoff = latest - self.settings.max_video_seconds
        keep = [index for index, timestamp in enumerate(state.watch_times_s) if timestamp >= cutoff]
        if len(keep) == len(state.watch_times_s):
            return
        state.watch_features = [state.watch_features[index] for index in keep]
        state.watch_times_s = [state.watch_times_s[index] for index in keep]
        state.watch_flags = [state.watch_flags[index] for index in keep]
        state.watch_frame_mediapipe_ms = [state.watch_frame_mediapipe_ms[index] for index in keep]

    def _push_preroll(
        self,
        state: LiveSessionState,
        feature: np.ndarray,
        timestamp_s: float,
        detection: DetectionFlags,
        frame_mp_ms: float,
    ) -> None:
        state.preroll.append((np.asarray(feature, dtype=np.float32), timestamp_s, detection, frame_mp_ms))
        cutoff = timestamp_s - self.settings.live_preroll_seconds
        state.preroll = [item for item in state.preroll if item[1] >= cutoff]

    def _start_live_segment(self, state: LiveSessionState) -> None:
        state.segment_index += 1
        state.features = [item[0] for item in state.preroll]
        state.times_s = [item[1] for item in state.preroll]
        state.flags = [item[2] for item in state.preroll]
        state.frame_mediapipe_ms = [item[3] for item in state.preroll]
        state.mediapipe_ms = float(sum(state.frame_mediapipe_ms))
        state.state = "signing"
        state.high_motion_frames = 0
        state.active_motion_frames = 0
        state.motion_score_sum = 0.0
        state.max_motion_score = 0.0
        state.low_motion_since_s = None

    def _append_live_frame(
        self,
        state: LiveSessionState,
        feature: np.ndarray,
        timestamp_s: float,
        detection: DetectionFlags,
        frame_mp_ms: float,
    ) -> None:
        if state.times_s and timestamp_s <= state.times_s[-1] + 1e-9:
            return
        state.features.append(np.asarray(feature, dtype=np.float32))
        state.times_s.append(timestamp_s)
        state.flags.append(detection)
        state.frame_mediapipe_ms.append(frame_mp_ms)
        state.mediapipe_ms = float(sum(state.frame_mediapipe_ms))

    def _finish_live_segment(self, state: LiveSessionState) -> None:
        state.features = []
        state.times_s = []
        state.flags = []
        state.frame_mediapipe_ms = []
        state.watch_features = []
        state.watch_times_s = []
        state.watch_flags = []
        state.watch_frame_mediapipe_ms = []
        state.preroll = []
        state.mediapipe_ms = 0.0
        state.state = "result"
        state.last_feature = None
        state.last_timestamp_s = None
        state.last_motion_score = 0.0
        state.high_motion_frames = 0
        state.active_motion_frames = 0
        state.motion_score_sum = 0.0
        state.max_motion_score = 0.0
        state.low_motion_since_s = None
        state.last_active_s = None

    def _segment_can_finalize(self, state: LiveSessionState) -> bool:
        return len(state.features) >= 2 and self._live_duration(state) >= self.settings.min_video_seconds

    def _segment_can_predict_draft(self, state: LiveSessionState) -> bool:
        if not self._segment_can_finalize(state) or not self._segment_has_meaningful_activity(state):
            return False
        if state.low_motion_since_s is None or not state.times_s:
            return True
        return max(state.times_s) - state.low_motion_since_s <= self.settings.live_draft_activity_grace_seconds

    def _watch_window_can_predict(self, state: LiveSessionState) -> bool:
        if state.state == "signing":
            return False
        if len(state.watch_features) < max(self.settings.min_model_frames, 10):
            return False
        if self._watch_duration(state) < max(self.settings.min_video_seconds, 0.9):
            return False
        rates = detection_summary(state.watch_flags)
        hand_rate = max(rates["left_hand_rate"], rates["right_hand_rate"])
        return hand_rate >= 0.04 or rates["pose_rate"] >= 0.20

    def _segment_has_meaningful_activity(self, state: LiveSessionState) -> bool:
        if state.active_motion_frames < self.settings.live_min_active_motion_frames:
            return False
        average_motion = state.motion_score_sum / max(1, state.active_motion_frames)
        return (
            average_motion >= self.settings.live_min_average_motion
            and state.max_motion_score >= self.settings.live_start_motion_threshold
        )

    def _effective_start_threshold(self, state: LiveSessionState) -> float:
        noise = min(state.idle_motion_ema, 0.018)
        return max(self.settings.live_start_motion_threshold, noise * 2.2 + 0.018)

    def _effective_active_threshold(self, state: LiveSessionState) -> float:
        noise = min(state.idle_motion_ema, 0.018)
        return max(self.settings.live_active_motion_threshold, noise * 1.45 + 0.008)

    def _looks_like_idle_noise(
        self,
        state: LiveSessionState,
        motion_score: float,
        hand_motion: float,
        arm_motion: float,
    ) -> bool:
        return (
            motion_score <= self._effective_active_threshold(state) * 0.85
            or (
                hand_motion < self.settings.live_min_hand_motion * 0.55
                and arm_motion < self.settings.live_min_hand_motion * 1.25
            )
        )

    @staticmethod
    def _update_idle_motion_baseline(state: LiveSessionState, motion_score: float) -> None:
        motion_score = min(float(motion_score), 0.018)
        if state.idle_motion_ema <= 0:
            state.idle_motion_ema = motion_score
        else:
            state.idle_motion_ema = state.idle_motion_ema * 0.94 + motion_score * 0.06

    def _trim_live_state(self, state: LiveSessionState) -> None:
        if not state.times_s:
            return
        latest = max(state.times_s)
        cutoff = latest - self.settings.max_video_seconds
        keep = [index for index, timestamp in enumerate(state.times_s) if timestamp >= cutoff]
        if len(keep) == len(state.times_s):
            return
        state.features = [state.features[index] for index in keep]
        state.times_s = [state.times_s[index] for index in keep]
        state.flags = [state.flags[index] for index in keep]
        state.frame_mediapipe_ms = [state.frame_mediapipe_ms[index] for index in keep]
        state.mediapipe_ms = float(sum(state.frame_mediapipe_ms))

    @staticmethod
    def _live_duration(state: LiveSessionState) -> float:
        if len(state.times_s) < 2:
            return 0.0
        return max(0.0, float(max(state.times_s) - min(state.times_s)))

    @staticmethod
    def _watch_duration(state: LiveSessionState) -> float:
        if len(state.watch_times_s) < 2:
            return 0.0
        return max(0.0, float(max(state.watch_times_s) - min(state.watch_times_s)))

    def _landmarks_from_live_state(self, state: LiveSessionState) -> LandmarkExtractionResult:
        if len(state.features) < 2:
            raise RecordingTooShortError("At least two live frames are required.")
        times = np.asarray(state.times_s, dtype=np.float64)
        duration = self._live_duration(state)
        return LandmarkExtractionResult(
            features=np.ascontiguousarray(np.stack(state.features, axis=0), dtype=np.float32),
            times_s=times,
            raw_video_frames=len(state.features),
            sampled_frames=len(state.features),
            duration_seconds=duration,
            video_decode_ms=0.0,
            mediapipe_ms=float(state.mediapipe_ms),
            detections=detection_summary(state.flags),
        )

    def _landmarks_from_watch_state(self, state: LiveSessionState) -> LandmarkExtractionResult:
        if len(state.watch_features) < 2:
            raise RecordingTooShortError("At least two live frames are required.")
        times = np.asarray(state.watch_times_s, dtype=np.float64)
        duration = self._watch_duration(state)
        return LandmarkExtractionResult(
            features=np.ascontiguousarray(np.stack(state.watch_features, axis=0), dtype=np.float32),
            times_s=times,
            raw_video_frames=len(state.watch_features),
            sampled_frames=len(state.watch_features),
            duration_seconds=duration,
            video_decode_ms=0.0,
            mediapipe_ms=float(sum(state.watch_frame_mediapipe_ms)),
            detections=detection_summary(state.watch_flags),
        )

    @staticmethod
    def _xy_motion(
        current: np.ndarray,
        previous: np.ndarray,
        *,
        start: int,
        count: int,
        stride: int,
        indexes: Sequence[int] | None = None,
        visibility: bool = False,
    ) -> float:
        current_points = np.asarray(current[start : start + count * stride], dtype=np.float32).reshape(count, stride)
        previous_points = np.asarray(previous[start : start + count * stride], dtype=np.float32).reshape(count, stride)
        if indexes is not None:
            current_points = current_points[list(indexes)]
            previous_points = previous_points[list(indexes)]

        current_xy = current_points[:, :2]
        previous_xy = previous_points[:, :2]
        valid = (np.abs(current_xy).sum(axis=1) > 1e-6) & (np.abs(previous_xy).sum(axis=1) > 1e-6)
        if visibility and current_points.shape[1] >= 4 and previous_points.shape[1] >= 4:
            valid &= (current_points[:, 3] > 0.15) & (previous_points[:, 3] > 0.15)
        if not bool(valid.any()):
            return 0.0
        return float(np.linalg.norm(current_xy[valid] - previous_xy[valid], axis=1).mean())

    @staticmethod
    def _pose_anchor(feature: np.ndarray) -> Optional[np.ndarray]:
        points = np.asarray(feature[:132], dtype=np.float32).reshape(33, 4)
        indexes = (11, 12, 23, 24)
        selected = points[list(indexes)]
        valid = (np.abs(selected[:, :2]).sum(axis=1) > 1e-6) & (selected[:, 3] > 0.15)
        if not bool(valid.any()):
            return None
        return np.asarray(selected[valid, :2].mean(axis=0), dtype=np.float32)

    @classmethod
    def _relative_hand_motion(cls, current: np.ndarray, previous: np.ndarray, *, start: int) -> Optional[float]:
        current_anchor = cls._pose_anchor(current)
        previous_anchor = cls._pose_anchor(previous)
        if current_anchor is None or previous_anchor is None:
            return None

        current_points = np.asarray(current[start : start + 21 * 3], dtype=np.float32).reshape(21, 3)
        previous_points = np.asarray(previous[start : start + 21 * 3], dtype=np.float32).reshape(21, 3)
        current_xy = current_points[:, :2]
        previous_xy = previous_points[:, :2]
        valid = (np.abs(current_xy).sum(axis=1) > 1e-6) & (np.abs(previous_xy).sum(axis=1) > 1e-6)
        if not bool(valid.any()):
            return 0.0

        current_relative = current_xy[valid] - current_anchor
        previous_relative = previous_xy[valid] - previous_anchor
        return float(np.linalg.norm(current_relative - previous_relative, axis=1).mean())

    @classmethod
    def _hand_motion(cls, current: np.ndarray, previous: np.ndarray, *, start: int) -> float:
        relative = cls._relative_hand_motion(current, previous, start=start)
        if relative is not None:
            return float(relative)
        return cls._xy_motion(current, previous, start=start, count=21, stride=3)

    @classmethod
    def _motion_components(cls, current: np.ndarray, previous: Optional[np.ndarray]) -> Dict[str, float]:
        if previous is None:
            return {"score": 0.0, "hand": 0.0, "arms": 0.0, "body": 0.0}

        pose_start = 0
        left_hand_start = 132 + 1404
        right_hand_start = left_hand_start + 63
        arm_indexes = (11, 12, 13, 14, 15, 16)

        left_hand = cls._hand_motion(current, previous, start=left_hand_start)
        right_hand = cls._hand_motion(current, previous, start=right_hand_start)
        hand = max(left_hand, right_hand) + 0.5 * (left_hand + right_hand)
        arms = cls._xy_motion(current, previous, start=pose_start, count=33, stride=4, indexes=arm_indexes, visibility=True)
        body = cls._xy_motion(current, previous, start=pose_start, count=33, stride=4, visibility=True)
        score = float(hand * 3.2 + arms * 1.8 + body * 0.35)
        return {"score": score, "hand": float(hand), "arms": float(arms), "body": float(body)}

    @classmethod
    def _motion_score(cls, current: np.ndarray, previous: Optional[np.ndarray]) -> float:
        return cls._motion_components(current, previous)["score"]

    def _predict_live_final_landmarks(
        self,
        landmarks: LandmarkExtractionResult,
        total_started: float,
        *,
        save_history: bool,
    ) -> InferenceResponse:
        primary_landmarks = self._trim_landmarks_to_active_sentence(landmarks)
        candidate_landmarks = [primary_landmarks]
        if primary_landmarks is not landmarks:
            candidate_landmarks.append(landmarks)

        responses: List[InferenceResponse] = []
        first_error: Optional[Exception] = None
        for candidate in candidate_landmarks:
            try:
                warnings = (
                    ["Live final sentence used the active signing window; inactive lead/trail frames were ignored."]
                    if candidate is not landmarks
                    else []
                )
                responses.append(
                    self._predict_landmarks(
                        candidate,
                        total_started,
                        save_history=False,
                        warnings=warnings,
                        artifact_source="live_final_active" if candidate is not landmarks else "live_final_full",
                    )
                )
            except (MediaPipeFailureError, NoCTCWordsError, RecordingTooShortError) as exc:
                if first_error is None:
                    first_error = exc

        if not responses:
            if first_error is not None:
                raise first_error
            raise NoCTCWordsError()

        response = self._choose_video_response(responses)
        if save_history:
            self._save_history_response(response)
        return response

    def _trim_landmarks_to_active_sentence(self, landmarks: LandmarkExtractionResult) -> LandmarkExtractionResult:
        features = np.asarray(landmarks.features, dtype=np.float32)
        times = np.asarray(landmarks.times_s, dtype=np.float64)
        frame_count = int(features.shape[0])
        if frame_count < max(3, self.settings.min_model_frames) or len(times) != frame_count:
            return landmarks

        motion_scores = np.zeros(frame_count, dtype=np.float32)
        hand_scores = np.zeros(frame_count, dtype=np.float32)
        arm_scores = np.zeros(frame_count, dtype=np.float32)
        for index in range(1, frame_count):
            motion = self._motion_components(features[index], features[index - 1])
            motion_scores[index] = float(motion["score"])
            hand_scores[index] = float(motion["hand"])
            arm_scores[index] = float(motion["arms"])

        active_threshold = max(0.012, float(self.settings.live_active_motion_threshold) * 0.75)
        strong_threshold = max(active_threshold * 1.8, float(self.settings.live_start_motion_threshold) * 0.75)
        hand_floor = float(self.settings.live_min_hand_motion) * 0.45
        arm_floor = float(self.settings.live_min_hand_motion) * 1.0
        active_mask = (
            (motion_scores >= active_threshold)
            & ((hand_scores >= hand_floor) | (arm_scores >= arm_floor))
        ) | (motion_scores >= strong_threshold)
        active_indexes = np.flatnonzero(active_mask)
        if len(active_indexes) < self.settings.live_min_active_motion_frames:
            return landmarks

        active_scores = motion_scores[active_indexes]
        if (
            float(active_scores.max(initial=0.0)) < float(self.settings.live_start_motion_threshold) * 0.75
            or float(active_scores.mean()) < float(self.settings.live_min_average_motion) * 0.65
        ):
            return landmarks

        sample_period = self._sample_period_from_times(times, fallback_fps=self.settings.target_fps)
        preroll_frames = max(2, int(round(max(0.28, self.settings.live_preroll_seconds) / sample_period)))
        postroll_frames = max(2, int(round(max(0.34, self.settings.live_end_silence_seconds * 0.65) / sample_period)))
        start = max(0, int(active_indexes[0]) - preroll_frames)
        end = min(frame_count, int(active_indexes[-1]) + postroll_frames + 1)

        trimmed_frames = start + (frame_count - end)
        min_meaningful_trim = max(3, int(round(0.18 / sample_period)))
        if trimmed_frames < min_meaningful_trim or end - start < max(2, self.settings.min_model_frames):
            return landmarks

        trimmed_times = np.asarray(times[start:end], dtype=np.float64)
        if len(trimmed_times) > 0:
            trimmed_times = trimmed_times - trimmed_times[0]
        trimmed_duration = float(trimmed_times[-1]) if len(trimmed_times) > 1 else 0.0
        trimmed_duration = max(trimmed_duration, float(max(0, end - start - 1) * sample_period))
        if trimmed_duration < self.settings.min_video_seconds:
            return landmarks

        return LandmarkExtractionResult(
            features=np.ascontiguousarray(features[start:end], dtype=np.float32),
            times_s=np.ascontiguousarray(trimmed_times, dtype=np.float64),
            raw_video_frames=landmarks.raw_video_frames,
            sampled_frames=int(end - start),
            duration_seconds=trimmed_duration,
            video_decode_ms=float(landmarks.video_decode_ms),
            mediapipe_ms=float(landmarks.mediapipe_ms),
            detections=dict(landmarks.detections),
        )

    @staticmethod
    def _sample_period_from_times(times: np.ndarray, *, fallback_fps: float) -> float:
        values = np.asarray(times, dtype=np.float64)
        diffs = np.diff(np.sort(values))
        positive = diffs[diffs > 1e-6]
        if len(positive) > 0:
            return max(1e-3, float(np.median(positive)))
        fps = max(1.0, float(fallback_fps))
        return 1.0 / fps

    @staticmethod
    def _decoded_word_count(response: InferenceResponse) -> int:
        return max(len(response.prediction.split()), len(response.token_ids))

    @classmethod
    def _choose_video_response(cls, responses: Sequence[InferenceResponse]) -> InferenceResponse:
        return max(
            responses,
            key=lambda response: (
                cls._decoded_word_count(response),
                float(response.confidence.value),
                int(response.input_frames),
            ),
        )

    def _predict_landmarks(
        self,
        landmarks: LandmarkExtractionResult,
        total_started: float,
        *,
        save_history: bool,
        warnings: Optional[List[str]] = None,
        artifact_source: Optional[str] = None,
    ) -> InferenceResponse:
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
            warnings=list(warnings or []),
        )

        if artifact_source:
            self._save_inference_artifacts(artifact_source, landmarks, model_features, response)

        if save_history:
            self._save_history_response(response)
        return response

    def _save_inference_artifacts(
        self,
        source: str,
        landmarks: LandmarkExtractionResult,
        model_features: np.ndarray,
        response: InferenceResponse,
    ) -> None:
        if not self.settings.save_inference_artifacts:
            return

        try:
            target_dir = self.settings.inference_artifact_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_source = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in source)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            stem = f"{stamp}_{safe_source}_{response.id[:8]}"

            raw_path = target_dir / f"{stem}_raw.npy"
            model_path = target_dir / f"{stem}_25fps.npy"
            meta_path = target_dir / f"{stem}_meta.json"

            np.save(raw_path, np.ascontiguousarray(landmarks.features, dtype=np.float32))
            np.save(model_path, np.ascontiguousarray(model_features, dtype=np.float32))
            metadata = {
                "id": response.id,
                "source": source,
                "prediction": response.prediction,
                "token_ids": response.token_ids,
                "raw_shape": list(landmarks.features.shape),
                "model_shape": list(model_features.shape),
                "raw_duration_seconds": float(landmarks.duration_seconds),
                "raw_capture_fps": self._landmarks_capture_fps(landmarks),
                "target_fps": float(self.settings.target_fps),
                "detections": response.detections.model_dump()
                if hasattr(response.detections, "model_dump")
                else response.detections.dict(),
                "timings": response.timings.model_dump()
                if hasattr(response.timings, "model_dump")
                else response.timings.dict(),
                "raw_path": str(raw_path),
                "model_path": str(model_path),
            }
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _save_history_response(self, response: InferenceResponse) -> None:
        self.history.add(
            {
                "id": response.id,
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
