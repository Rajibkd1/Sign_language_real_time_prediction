from __future__ import annotations

from app.services.mediapipe_service import MediaPipe1662Extractor
from app.services.mediapipe_service import MediaPipeService


def test_timestamp_sanitizer_preserves_relative_timing_after_reset():
    extractor = MediaPipe1662Extractor.__new__(MediaPipe1662Extractor)
    extractor._last_timestamp_ms = -1
    extractor._timestamp_offset_ms = 0

    assert extractor._sanitize_timestamp_ms(0) == 0
    assert extractor._sanitize_timestamp_ms(40) == 40

    assert extractor._sanitize_timestamp_ms(0) == 41
    assert extractor._sanitize_timestamp_ms(40) == 81


def test_video_fps_filter_rejects_unreliable_metadata():
    assert MediaPipeService._trusted_video_fps(0) == 0.0
    assert MediaPipeService._trusted_video_fps(1000) == 0.0
    assert MediaPipeService._trusted_video_fps(30) == 30.0
    assert MediaPipeService._trusted_video_fps(60) == 60.0
