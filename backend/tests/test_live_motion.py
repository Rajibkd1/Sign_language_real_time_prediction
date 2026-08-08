from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from app.config import Settings
from app.services.inference_service import InferenceService, LiveSessionState
from app.services.mediapipe_service import DetectionFlags, LandmarkExtractionResult, detection_summary
from app.utils.sequence import FEATURE_DIM


def _service() -> InferenceService:
    service = InferenceService.__new__(InferenceService)
    service.settings = Settings()
    return service


def _live_service_with_batch(batch) -> tuple[InferenceService, list[dict]]:
    saved_history: list[dict] = []
    service = _service()
    service._live_sessions = {}
    service._live_lock = threading.Lock()
    service.mediapipe = SimpleNamespace(extract_live_frame_batch=lambda _frames: batch)
    service.history = SimpleNamespace(add=saved_history.append)
    service.runtime = SimpleNamespace(
        predict=lambda features, _min_frames: SimpleNamespace(
            sentence="test sentence",
            token_ids=[1, 2],
            input_frames=int(len(features)),
            output_steps=max(1, int(len(features) // 2)),
            model_ms=1.0,
            confidence_value=0.9,
        )
    )
    return service, saved_history


def _left_hand_feature(x: float, y: float = 0.45) -> np.ndarray:
    feature = np.zeros(FEATURE_DIM, dtype=np.float32)
    left_hand_start = 132 + 1404
    for point in range(21):
        base = left_hand_start + point * 3
        feature[base : base + 2] = [x, y]
    return feature


def _pose_and_left_hand_feature(pose_x: float, hand_x: float, y: float = 0.45) -> np.ndarray:
    feature = _left_hand_feature(hand_x, y)
    pose = feature[:132].reshape(33, 4)
    for point in (11, 12, 13, 14, 15, 16, 23, 24):
        pose[point, :2] = [pose_x, y]
        pose[point, 3] = 1.0
    return feature


def _batch(features: list[np.ndarray], step_s: float = 0.08):
    return SimpleNamespace(
        features=np.stack(features, axis=0),
        times_s=np.asarray([index * step_s for index in range(len(features))], dtype=np.float64),
        flags=[DetectionFlags(pose=True, face=False, left_hand=True, right_hand=False) for _ in features],
        mediapipe_ms_by_frame=[1.0 for _ in features],
    )


def _landmarks(features: list[np.ndarray], step_s: float = 0.08) -> LandmarkExtractionResult:
    flags = [DetectionFlags(pose=True, face=False, left_hand=True, right_hand=False) for _ in features]
    return LandmarkExtractionResult(
        features=np.ascontiguousarray(np.stack(features, axis=0), dtype=np.float32),
        times_s=np.asarray([index * step_s for index in range(len(features))], dtype=np.float64),
        raw_video_frames=len(features),
        sampled_frames=len(features),
        duration_seconds=(len(features) - 1) * step_s,
        video_decode_ms=4.0,
        mediapipe_ms=float(len(features)),
        detections=detection_summary(flags),
    )


def test_live_motion_score_weights_hand_movement():
    previous = np.zeros(FEATURE_DIM, dtype=np.float32)
    current = np.zeros(FEATURE_DIM, dtype=np.float32)
    left_hand_start = 132 + 1404

    for point in range(21):
        base = left_hand_start + point * 3
        previous[base : base + 2] = [0.45, 0.45]
        current[base : base + 2] = [0.49, 0.45]

    score = InferenceService._motion_score(current, previous)

    assert score > 0.018


def test_live_motion_score_ignores_missing_landmarks():
    previous = np.zeros(FEATURE_DIM, dtype=np.float32)
    current = np.zeros(FEATURE_DIM, dtype=np.float32)

    assert InferenceService._motion_score(current, previous) == 0.0


def test_idle_jitter_does_not_start_live_segment():
    service = _service()
    state = LiveSessionState()
    features = [_left_hand_feature(0.45 + (0.0008 if index % 2 else 0.0)) for index in range(24)]

    final, segment_id = service._ingest_live_batch(state, _batch(features))

    assert final is None
    assert segment_id is None
    assert state.state == "idle"
    assert len(state.features) == 0
    assert len(state.watch_features) == len(features)


def test_idle_watch_window_can_feed_screen_replay_fallback():
    service = _service()
    state = LiveSessionState()
    features = [_pose_and_left_hand_feature(0.35, 0.52 + index * 0.002) for index in range(16)]

    final, segment_id = service._ingest_live_batch(state, _batch(features))
    landmarks = service._landmarks_from_watch_state(state)

    assert final is None
    assert segment_id is None
    assert state.state == "idle"
    assert service._watch_window_can_predict(state)
    assert landmarks.sampled_frames == len(features)
    assert landmarks.duration_seconds >= 0.9


def test_meaningful_hand_motion_finalizes_after_silence():
    service = _service()
    state = LiveSessionState()
    moving = [_left_hand_feature(0.35 + index * 0.028) for index in range(16)]
    still = [_left_hand_feature(0.35 + 15 * 0.028) for _ in range(24)]

    final, segment_id = service._ingest_live_batch(state, _batch([*moving, *still]))

    assert final is not None
    assert segment_id == 1
    assert final.sampled_frames >= 8
    assert state.state == "result"
    assert state.segment_index == 1
    assert state.preroll == []
    assert state.last_feature is None


def test_live_final_decode_is_returned_even_on_non_predict_tick():
    moving = [_left_hand_feature(0.35 + index * 0.028) for index in range(16)]
    still = [_left_hand_feature(0.35 + 15 * 0.028) for _ in range(24)]
    service, saved_history = _live_service_with_batch(_batch([*moving, *still]))

    live = service.predict_live_session(
        session_id=None,
        frames=[(np.zeros((2, 2, 3), dtype=np.uint8), 0.0)],
        reset=True,
        predict=False,
        save_history=True,
    )

    assert live.is_final
    assert live.state == "result"
    assert live.result is not None
    assert live.result.prediction == "test sentence"
    assert len(saved_history) == 1


def test_live_sentence_does_not_finalize_on_short_internal_pause():
    service = _service()
    state = LiveSessionState()
    first_word = [_left_hand_feature(0.35 + index * 0.028) for index in range(12)]
    pause = [_left_hand_feature(0.35 + 11 * 0.028) for _ in range(9)]
    second_word = [_left_hand_feature(0.35 + 11 * 0.028 - index * 0.026) for index in range(10)]

    final, segment_id = service._ingest_live_batch(state, _batch([*first_word, *pause, *second_word]))

    assert final is None
    assert segment_id is None
    assert state.state == "signing"
    assert state.segment_index == 1
    assert state.low_motion_since_s is None


def test_slow_meaningful_hand_motion_can_start_segment():
    service = _service()
    state = LiveSessionState()
    moving = [_left_hand_feature(0.35 + index * 0.010) for index in range(12)]

    final, segment_id = service._ingest_live_batch(state, _batch(moving))

    assert final is None
    assert segment_id is None
    assert state.state == "signing"
    assert state.segment_index == 1
    assert state.active_motion_frames >= 3


def test_scaled_screen_replay_motion_can_start_segment():
    service = _service()
    state = LiveSessionState()
    moving = [_left_hand_feature(0.35 + index * 0.006) for index in range(14)]

    final, segment_id = service._ingest_live_batch(state, _batch(moving))

    assert final is None
    assert segment_id is None
    assert state.state == "signing"
    assert state.segment_index == 1
    assert state.active_motion_frames >= 3


def test_relative_hand_motion_ignores_global_camera_jitter():
    service = _service()
    state = LiveSessionState()
    jitter = [_pose_and_left_hand_feature(0.35 + index * 0.006, 0.52 + index * 0.006) for index in range(14)]

    final, segment_id = service._ingest_live_batch(state, _batch(jitter))

    assert final is None
    assert segment_id is None
    assert state.state == "idle"


def test_relative_hand_motion_keeps_true_screen_replay_signing():
    service = _service()
    state = LiveSessionState()
    moving = [_pose_and_left_hand_feature(0.35, 0.52 + index * 0.006) for index in range(14)]

    final, segment_id = service._ingest_live_batch(state, _batch(moving))

    assert final is None
    assert segment_id is None
    assert state.state == "signing"
    assert state.active_motion_frames >= 3


def test_draft_prediction_stops_when_hands_are_idle():
    service = _service()
    state = LiveSessionState()
    moving = [_left_hand_feature(0.35 + index * 0.028) for index in range(16)]
    almost_still = [_left_hand_feature(0.35 + 15 * 0.028) for _ in range(4)]

    final, segment_id = service._ingest_live_batch(state, _batch([*moving, *almost_still]))

    assert final is None
    assert segment_id is None
    assert state.state == "signing"
    assert state.low_motion_since_s is not None
    assert service._segment_has_meaningful_activity(state)
    assert not service._segment_can_predict_draft(state)


def test_upload_sentence_trim_removes_inactive_lead_and_trail():
    service = _service()
    leading_idle = [_left_hand_feature(0.35) for _ in range(8)]
    moving_sentence = [_left_hand_feature(0.35 + index * 0.028) for index in range(12)]
    trailing_idle = [_left_hand_feature(0.35 + 11 * 0.028) for _ in range(10)]
    original = _landmarks([*leading_idle, *moving_sentence, *trailing_idle])

    trimmed = service._trim_landmarks_to_active_sentence(original)

    assert trimmed is not original
    assert trimmed.sampled_frames < original.sampled_frames
    assert trimmed.duration_seconds < original.duration_seconds
    assert trimmed.times_s[0] == 0.0
    assert trimmed.raw_video_frames == original.raw_video_frames


def test_upload_sentence_trim_keeps_idle_jitter_unchanged():
    service = _service()
    jitter = [_left_hand_feature(0.45 + (0.0008 if index % 2 else 0.0)) for index in range(30)]
    original = _landmarks(jitter)

    trimmed = service._trim_landmarks_to_active_sentence(original)

    assert trimmed is original
