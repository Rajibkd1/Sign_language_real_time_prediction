from __future__ import annotations

import numpy as np

from app.utils.sequence import FEATURE_DIM, MAX_SEQ_LEN, resample_landmarks_vectorized


def test_resample_caps_first_384_frames():
    features = np.zeros((500, FEATURE_DIM), dtype=np.float32)
    times = np.linspace(0, 20, 500)
    out = resample_landmarks_vectorized(features, times, target_fps=25.0, max_seq_len=MAX_SEQ_LEN)
    assert out.shape == (384, FEATURE_DIM)


def test_resample_121_input_duration_keeps_expected_shape():
    features = np.zeros((121, FEATURE_DIM), dtype=np.float32)
    times = np.linspace(0, 4.8, 121)
    out = resample_landmarks_vectorized(features, times, target_fps=25.0, max_seq_len=MAX_SEQ_LEN)
    assert out.shape == (121, FEATURE_DIM)

