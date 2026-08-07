from __future__ import annotations

import numpy as np


FEATURE_DIM = 1662
TARGET_FPS = 25.0
MAX_SEQ_LEN = 384


def resample_landmarks_vectorized(
    features: np.ndarray,
    times_s: np.ndarray,
    target_fps: float = TARGET_FPS,
    max_seq_len: int = MAX_SEQ_LEN,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    t = np.asarray(times_s, dtype=np.float64)

    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected [N,{FEATURE_DIM}], got {x.shape}")
    if len(x) != len(t):
        raise ValueError("Feature/time length mismatch.")
    if len(x) < 2:
        return np.ascontiguousarray(x[:max_seq_len])

    order = np.argsort(t, kind="stable")
    x = x[order]
    t = t[order]
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > 1e-6
    x = x[keep]
    t = t[keep]
    if len(x) < 2:
        return np.ascontiguousarray(x[:max_seq_len])

    t = t - t[0]
    duration = float(t[-1])
    if duration <= 0:
        return np.ascontiguousarray(x[:max_seq_len])

    n_target = int(round(duration * target_fps)) + 1
    n_target = max(2, min(max_seq_len, n_target))
    target_t = np.linspace(0.0, duration, n_target, dtype=np.float64)

    right = np.searchsorted(t, target_t, side="right")
    right = np.clip(right, 1, len(t) - 1)
    left = right - 1
    denom = np.maximum(t[right] - t[left], 1e-9)
    alpha = ((target_t - t[left]) / denom).astype(np.float32)
    out = x[left] * (1.0 - alpha[:, None]) + x[right] * alpha[:, None]
    return np.ascontiguousarray(out, dtype=np.float32)

