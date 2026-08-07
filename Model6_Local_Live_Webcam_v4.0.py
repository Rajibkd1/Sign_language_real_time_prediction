from __future__ import annotations

"""
Model 6 - Efficient Local Live Webcam Inference v4.0
=====================================================

TRUE live webcam deployment for the selected BdSL Model 6.
Run on a LOCAL Windows/Linux/macOS machine. Do not run the live camera app
inside Kaggle/Colab because those kernels cannot directly access your laptop
camera through cv2.VideoCapture(0).

Required files beside this script:
    model6_best_wer_checkpoint.pt
    bangla_vocab_tsv_fixed.json

Controls:
    SPACE : start sentence / stop and predict
    C     : clear prediction
    Q/ESC : quit

Design:
    Webcam -> latest-frame queue -> MediaPipe Tasks -> [T,1662]
    -> temporal resampling to 25 FPS -> Model 6 -> greedy word-level CTC

Important:
    Extraction is NEVER mirrored. Only the preview may be mirrored.
"""

# -----------------------------------------------------------------------------
# 0. Lightweight environment guard BEFORE heavy imports
# -----------------------------------------------------------------------------
import argparse
import contextlib
import gc
import importlib.util
import json
import math
import os
import platform
import queue
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple


def running_in_remote_notebook() -> bool:
    return (
        Path('/kaggle').exists()
        or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
        or 'COLAB_GPU' in os.environ
        or 'COLAB_RELEASE_TAG' in os.environ
    )


def dependency_error() -> RuntimeError:
    py = sys.executable
    cmd = f'"{py}" -m pip install -r requirements.txt'
    return RuntimeError(
        'One or more required Python packages are missing.\n\n'
        'Recommended setup:\n'
        '  Windows: double-click setup_and_run.bat\n'
        '  Linux/macOS: ./setup_and_run.sh\n\n'
        f'Or run manually:\n  {cmd}\n'
    )


missing_modules = [
    name for name in ('numpy', 'cv2', 'torch', 'PIL', 'mediapipe')
    if importlib.util.find_spec(name) is None
]
if missing_modules:
    raise dependency_error()

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp

# Reduce CPU oversubscription. MediaPipe already uses native worker threads.
try:
    cv2.setNumThreads(1)
except Exception:
    pass

if torch.get_num_threads() > max(1, (os.cpu_count() or 2) // 2):
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
FEATURE_DIM = 1662
MODALITY_DIMS = (132, 1404, 63, 63)
REGION_EMBED_DIMS = (64, 64, 96, 96)
D_MODEL = sum(REGION_EMBED_DIMS)  # 320
STREAM_GATE_HIDDEN = 160
TEMPORAL_KERNEL_SIZE = 7
TEMPORAL_SUBSAMPLE_STRIDE = 2
TRANSFORMER_LAYERS = 4
NHEAD = 8
FFN_DIM = 960
DROPOUT = 0.15
LAYER_SCALE_INIT = 0.10
POSITIONAL_MAX_LEN = 512
TARGET_FPS = 25.0
MAX_SEQ_LEN = 384
MAX_RECORD_SECONDS = MAX_SEQ_LEN / TARGET_FPS
EXPECTED_PARAM_COUNT_FOR_VOCAB39 = 4_924_163


@dataclass(frozen=True)
class PerformanceProfile:
    name: str
    camera_width: int
    camera_height: int
    mediapipe_width: int
    landmark_sample_fps: float


PROFILES = {
    'fast': PerformanceProfile('fast', 640, 480, 480, 10.0),
    'balanced': PerformanceProfile('balanced', 960, 540, 640, 15.0),
    'accurate': PerformanceProfile('accurate', 1280, 720, 640, 25.0),
}


@dataclass
class AppConfig:
    root: Path
    checkpoint_path: Path
    vocab_path: Path
    mediapipe_model_dir: Path
    output_dir: Path
    camera_index: int = 0
    profile: PerformanceProfile = PROFILES['balanced']
    prefer_ema: bool = True
    use_amp: bool = True
    mirror_preview: bool = True
    save_features: bool = True
    min_raw_samples: int = 8
    bengali_font_path: str = ''


# -----------------------------------------------------------------------------
# 2. Vocabulary
# -----------------------------------------------------------------------------
def load_vocab(path: Path):
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if 'word2id' not in data or 'id2word' not in data:
        raise KeyError("Vocabulary JSON must contain 'word2id' and 'id2word'.")

    word2id = {str(word): int(idx) for word, idx in data['word2id'].items()}
    id2word = {int(idx): str(word) for idx, word in data['id2word'].items()}
    max_idx = max(word2id.values(), default=-1)
    vocab_size = int(data.get('vocab_size', max_idx + 1))

    if vocab_size != max_idx + 1:
        raise ValueError(
            f'Vocabulary IDs must be contiguous 0..{max_idx}; vocab_size={vocab_size}.'
        )
    expected = set(range(vocab_size))
    actual = set(word2id.values())
    if actual != expected:
        missing = sorted(expected - actual)
        raise ValueError(f'Vocabulary token IDs are not contiguous. Missing IDs: {missing[:10]}')
    for idx in expected:
        if idx not in id2word:
            raise ValueError(f'id2word is missing token id {idx}.')

    return word2id, id2word, vocab_size


def resolve_special_token(word2id: Mapping[str, int], aliases: List[str], name: str) -> int:
    normalized = {str(k).strip().casefold(): int(v) for k, v in word2id.items()}
    for alias in aliases:
        key = alias.strip().casefold()
        if key in normalized:
            return normalized[key]
    raise KeyError(f'Could not resolve {name}. Tried aliases: {aliases}')


# -----------------------------------------------------------------------------
# 3. Model 6 exact deployment architecture
# -----------------------------------------------------------------------------
class ModalityAwareNormalizer(nn.Module):
    def __init__(self, dims=MODALITY_DIMS):
        super().__init__()
        self.dims = tuple(int(x) for x in dims)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in self.dims])

    def forward(self, x):
        chunks = torch.split(x, self.dims, dim=-1)
        return torch.cat([norm(chunk) for norm, chunk in zip(self.norms, chunks)], dim=-1)


class CompactRegionProjector(nn.Module):
    def __init__(self, in_dims=MODALITY_DIMS, out_dims=REGION_EMBED_DIMS, dropout=DROPOUT):
        super().__init__()
        self.in_dims = tuple(int(x) for x in in_dims)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout),
            )
            for in_dim, out_dim in zip(in_dims, out_dims)
        ])

    def forward(self, x):
        chunks = torch.split(x, self.in_dims, dim=-1)
        return torch.cat([block(chunk) for block, chunk in zip(self.blocks, chunks)], dim=-1)


class RawVelocityGate(nn.Module):
    def __init__(self, d_model=D_MODEL, hidden=STREAM_GATE_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        nn.init.constant_(self.gate[-1].bias, 1.0)

    def forward(self, raw, velocity):
        raw_weight = torch.sigmoid(self.gate(torch.cat([raw, velocity], dim=-1)))
        fused = raw_weight * raw + (1.0 - raw_weight) * velocity
        return fused, raw_weight


class StridedDepthwiseTemporalBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, kernel_size=TEMPORAL_KERNEL_SIZE,
                 stride=TEMPORAL_SUBSAMPLE_STRIDE, dropout=DROPOUT):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            stride=stride, padding=padding, groups=d_model, bias=False,
        )
        self.pointwise_glu = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = F.glu(self.pointwise_glu(y), dim=1).transpose(1, 2)
        return self.dropout(self.output_norm(y))


class ResidualDepthwiseTemporalBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, kernel_size=TEMPORAL_KERNEL_SIZE,
                 dropout=DROPOUT, layer_scale=LAYER_SCALE_INIT):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=padding, groups=d_model, bias=False,
        )
        self.pointwise_glu = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((d_model,), float(layer_scale)))

    def forward(self, x, padding_mask=None):
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = F.glu(self.pointwise_glu(y), dim=1).transpose(1, 2)
        x = x + self.dropout(y) * self.layer_scale
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, max_len=POSITIONAL_MAX_LEN, dropout=DROPOUT):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(1, max_len, d_model, dtype=torch.float32)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encoding', encoding, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.size(1) > self.encoding.size(1):
            raise ValueError(
                f'Sequence length {x.size(1)} exceeds positional limit {self.encoding.size(1)}'
            )
        return self.dropout(
            x + self.encoding[:, :x.size(1)].to(device=x.device, dtype=x.dtype)
        )


class Model6(nn.Module):
    def __init__(self, vocab_size: int, blank_idx: int):
        super().__init__()
        self.normalizer = ModalityAwareNormalizer()
        self.raw_projector = CompactRegionProjector()
        self.velocity_projector = CompactRegionProjector()
        self.stream_gate = RawVelocityGate()
        self.temporal_subsampler = StridedDepthwiseTemporalBlock()
        self.temporal_refiner = ResidualDepthwiseTemporalBlock()
        self.positional_encoding = SinusoidalPositionalEncoding()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TRANSFORMER_LAYERS,
            norm=nn.LayerNorm(D_MODEL),
        )
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.classifier = nn.Linear(D_MODEL, int(vocab_size))

        nn.init.constant_(self.classifier.bias, 0.0)
        with torch.no_grad():
            self.classifier.bias[int(blank_idx)] = -0.25

    @staticmethod
    def output_lengths(input_lengths):
        return torch.div(
            input_lengths - 1,
            TEMPORAL_SUBSAMPLE_STRIDE,
            rounding_mode='floor',
        ) + 1

    def forward(self, x, input_lengths, padding_mask=None, return_gate=False):
        normalized = self.normalizer(x)

        velocity = torch.zeros_like(normalized)
        velocity[:, 1:] = normalized[:, 1:] - normalized[:, :-1]

        if padding_mask is not None:
            normalized = normalized.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            velocity = velocity.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        raw_embedding = self.raw_projector(normalized)
        velocity_embedding = self.velocity_projector(velocity)
        x, raw_weight = self.stream_gate(raw_embedding, velocity_embedding)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = self.temporal_subsampler(x)
        output_lengths = self.output_lengths(input_lengths)
        output_mask = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0)
            >= output_lengths.unsqueeze(1)
        )
        x = x.masked_fill(output_mask.unsqueeze(-1), 0.0)

        x = self.temporal_refiner(x, output_mask)
        x = self.positional_encoding(x)
        x = self.transformer(x, src_key_padding_mask=output_mask)
        x = self.output_norm(x).masked_fill(output_mask.unsqueeze(-1), 0.0)

        log_probs = F.log_softmax(self.classifier(x).float(), dim=-1)
        if return_gate:
            return log_probs, output_lengths, output_mask, raw_weight
        return log_probs, output_lengths, output_mask


# -----------------------------------------------------------------------------
# 4. Checkpoint loading
# -----------------------------------------------------------------------------
def torch_load_full(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def strip_common_prefixes(state_dict: Mapping[str, torch.Tensor]):
    result = dict(state_dict)
    for prefix in ('module.', '_orig_mod.'):
        if result and all(k.startswith(prefix) for k in result):
            result = {k[len(prefix):]: v for k, v in result.items()}
    return result


def load_model6(config: AppConfig, vocab_size: int, blank_idx: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch_load_full(config.checkpoint_path)

    state_key = (
        'ema_state'
        if config.prefer_ema and checkpoint.get('ema_state') is not None
        else 'model_state'
    )
    state_dict = checkpoint.get(state_key)
    if state_dict is None:
        raise KeyError(f'Checkpoint has no usable {state_key!r} state.')
    state_dict = strip_common_prefixes(state_dict)

    classifier_key = next(
        (k for k in state_dict if k.endswith('classifier.weight')),
        None,
    )
    if classifier_key is None:
        raise KeyError('Could not find classifier.weight in checkpoint state dict.')

    checkpoint_vocab_size = int(state_dict[classifier_key].shape[0])
    if checkpoint_vocab_size != vocab_size:
        raise ValueError(
            f'Vocabulary mismatch: checkpoint={checkpoint_vocab_size}, JSON={vocab_size}. '
            'Use the exact bangla_vocab_tsv_fixed.json from this training run.'
        )

    model = Model6(vocab_size, blank_idx).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    metadata = {
        'state_key': state_key,
        'epoch': checkpoint.get('epoch'),
        'checkpoint_kind': checkpoint.get('checkpoint_kind'),
        'model_variant': checkpoint.get('model_variant'),
        'device': str(device),
    }

    # Release duplicate CPU checkpoint payload immediately.
    del state_dict
    del checkpoint
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if vocab_size == 39 and param_count != EXPECTED_PARAM_COUNT_FOR_VOCAB39:
        raise RuntimeError(
            f'Unexpected Model 6 parameter count: {param_count:,}; '
            f'expected {EXPECTED_PARAM_COUNT_FOR_VOCAB39:,} for vocab size 39.'
        )
    metadata['param_count'] = param_count
    return model, device, metadata


# -----------------------------------------------------------------------------
# 5. MediaPipe assets and exact 1662-D extractor
# -----------------------------------------------------------------------------
MP_ASSETS = {
    'pose_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_full/float16/latest/pose_landmarker_full.task'
    ),
    'hand_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
        'hand_landmarker/float16/latest/hand_landmarker.task'
    ),
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
        'face_landmarker/float16/latest/face_landmarker.task'
    ),
}


def ensure_mediapipe_assets(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MP_ASSETS.items():
        target = model_dir / filename
        if target.exists() and target.stat().st_size > 1024:
            continue
        print(f'Downloading MediaPipe asset: {filename}')
        try:
            tmp = target.with_suffix(target.suffix + '.part')
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)
        except Exception as exc:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise RuntimeError(
                f'Could not download {filename}.\n'
                f'Download it manually and place it at:\n  {target}\n'
                f'Original error: {exc}'
            ) from exc


@dataclass
class DetectionFlags:
    pose: bool
    face: bool
    left_hand: bool
    right_hand: bool


class MediaPipe1662Extractor:
    """Exact feature order: pose132 + face1404 + left63 + right63."""

    def __init__(self, config: AppConfig):
        self.config = config
        BaseOptions = mp.tasks.BaseOptions
        RunningMode = mp.tasks.vision.RunningMode

        self.pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(config.mediapipe_model_dir / 'pose_landmarker.task')
                ),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(config.mediapipe_model_dir / 'hand_landmarker.task')
                ),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.face_detector = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(config.mediapipe_model_dir / 'face_landmarker.task')
                ),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._last_timestamp_ms = -1

    def close(self) -> None:
        for detector in (self.pose_detector, self.hand_detector, self.face_detector):
            try:
                detector.close()
            except Exception:
                pass

    def _sanitize_timestamp_ms(self, timestamp_ms: int) -> int:
        ts = int(timestamp_ms)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    def _resize(self, rgb: np.ndarray) -> np.ndarray:
        target_width = self.config.profile.mediapipe_width
        h, w = rgb.shape[:2]
        if w <= target_width:
            return np.ascontiguousarray(rgb)
        scale = target_width / float(w)
        nh = max(1, int(round(h * scale)))
        return cv2.resize(rgb, (target_width, nh), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _pose(result) -> Tuple[np.ndarray, bool]:
        if result.pose_landmarks:
            values = []
            for lm in result.pose_landmarks[0]:
                visibility = float(lm.visibility) if lm.visibility is not None else 0.0
                values.extend([float(lm.x), float(lm.y), float(lm.z), visibility])
            arr = np.asarray(values, dtype=np.float32)
            if arr.size == 132:
                return arr, True
        return np.zeros(132, dtype=np.float32), False

    @staticmethod
    def _face(result) -> Tuple[np.ndarray, bool]:
        if result.face_landmarks:
            pts = result.face_landmarks[0][:468]
            arr = np.asarray(
                [[lm.x, lm.y, lm.z] for lm in pts], dtype=np.float32
            ).reshape(-1)
            if arr.size == 1404:
                return arr, True
        return np.zeros(1404, dtype=np.float32), False

    @staticmethod
    def _hands(result) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
        left = np.zeros(63, dtype=np.float32)
        right = np.zeros(63, dtype=np.float32)
        left_found = False
        right_found = False

        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                pts = np.asarray(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32
                ).reshape(-1)
                if pts.size != 63:
                    continue

                label = ''
                if i < len(result.handedness) and result.handedness[i]:
                    label = str(result.handedness[i][0].category_name)

                if label.casefold() == 'left':
                    left = pts
                    left_found = True
                elif label.casefold() == 'right':
                    right = pts
                    right_found = True

        return left, right, left_found, right_found

    def extract(self, bgr_frame: np.ndarray, capture_elapsed_s: float):
        # IMPORTANT: do not mirror before extraction.
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = self._resize(rgb)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb, dtype=np.uint8),
        )
        timestamp_ms = self._sanitize_timestamp_ms(round(capture_elapsed_s * 1000.0))

        started = time.perf_counter()
        pose_res = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
        hand_res = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
        face_res = self.face_detector.detect_for_video(mp_image, timestamp_ms)
        mp_ms = (time.perf_counter() - started) * 1000.0

        pose, pose_found = self._pose(pose_res)
        face, face_found = self._face(face_res)
        left, right, left_found, right_found = self._hands(hand_res)
        feature = np.concatenate([pose, face, left, right]).astype(np.float32, copy=False)

        if feature.shape != (FEATURE_DIM,):
            raise RuntimeError(f'Unexpected feature shape: {feature.shape}')
        if not np.isfinite(feature).all():
            feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

        flags = DetectionFlags(pose_found, face_found, left_found, right_found)
        return feature, mp_ms, flags


# -----------------------------------------------------------------------------
# 6. Latest-frame landmark worker
# -----------------------------------------------------------------------------
class LandmarkWorker:
    def __init__(self, extractor: MediaPipe1662Extractor):
        self.extractor = extractor
        self.jobs: queue.Queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.features: List[np.ndarray] = []
        self.times_s: List[float] = []
        self.mp_ms: List[float] = []
        self.flags: List[DetectionFlags] = []
        self.errors: List[str] = []
        self._stop = threading.Event()
        self._busy = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='mediapipe-worker')
        self.thread.start()

    def reset(self) -> None:
        self.flush(timeout=2.0)
        with self.lock:
            self.features.clear()
            self.times_s.clear()
            self.mp_ms.clear()
            self.flags.clear()
            self.errors.clear()

    def submit_latest(self, frame: np.ndarray, elapsed_s: float) -> None:
        # The OpenCV frame object is new on each cap.read(); make one compact copy for worker ownership.
        job = (np.ascontiguousarray(frame), float(elapsed_s))
        try:
            self.jobs.put_nowait(job)
            return
        except queue.Full:
            pass

        # Replace stale pending work with newest frame.
        try:
            self.jobs.get_nowait()
            self.jobs.task_done()
        except queue.Empty:
            pass
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            pass

    def flush(self, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.jobs.empty() and not self._busy.is_set():
                return True
            time.sleep(0.005)
        return self.jobs.empty() and not self._busy.is_set()

    def snapshot(self):
        with self.lock:
            return (
                np.asarray(self.features, dtype=np.float32),
                np.asarray(self.times_s, dtype=np.float64),
                np.asarray(self.mp_ms, dtype=np.float64),
                list(self.flags),
                list(self.errors),
            )

    def count(self) -> int:
        with self.lock:
            return len(self.features)

    def close(self) -> None:
        self._stop.set()
        # Remove pending frame and insert sentinel.
        while True:
            try:
                self.jobs.get_nowait()
                self.jobs.task_done()
            except queue.Empty:
                break
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue

            if job is None:
                self.jobs.task_done()
                break

            frame, elapsed_s = job
            self._busy.set()
            try:
                feature, mp_ms, flags = self.extractor.extract(frame, elapsed_s)
                with self.lock:
                    self.features.append(feature)
                    self.times_s.append(float(elapsed_s))
                    self.mp_ms.append(float(mp_ms))
                    self.flags.append(flags)
            except Exception as exc:
                with self.lock:
                    self.errors.append(f'{type(exc).__name__}: {exc}')
            finally:
                self._busy.clear()
                self.jobs.task_done()


# -----------------------------------------------------------------------------
# 7. Resampling + CTC + inference
# -----------------------------------------------------------------------------
def resample_landmarks_vectorized(features: np.ndarray, times_s: np.ndarray,
                                  target_fps: float = TARGET_FPS) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    t = np.asarray(times_s, dtype=np.float64)

    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f'Expected [N,{FEATURE_DIM}], got {x.shape}')
    if len(x) != len(t):
        raise ValueError('Feature/time length mismatch.')
    if len(x) < 2:
        return np.ascontiguousarray(x[:MAX_SEQ_LEN])

    order = np.argsort(t, kind='stable')
    x = x[order]
    t = t[order]
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > 1e-6
    x = x[keep]
    t = t[keep]
    if len(x) < 2:
        return np.ascontiguousarray(x[:MAX_SEQ_LEN])

    t = t - t[0]
    duration = float(t[-1])
    if duration <= 0:
        return np.ascontiguousarray(x[:MAX_SEQ_LEN])

    n_target = int(round(duration * target_fps)) + 1
    n_target = max(2, min(MAX_SEQ_LEN, n_target))
    target_t = np.linspace(0.0, duration, n_target, dtype=np.float64)

    right = np.searchsorted(t, target_t, side='right')
    right = np.clip(right, 1, len(t) - 1)
    left = right - 1
    denom = np.maximum(t[right] - t[left], 1e-9)
    alpha = ((target_t - t[left]) / denom).astype(np.float32)
    out = x[left] * (1.0 - alpha[:, None]) + x[right] * alpha[:, None]
    return np.ascontiguousarray(out, dtype=np.float32)


def greedy_ctc_decode(log_probs: torch.Tensor, valid_length: int, blank_idx: int) -> List[int]:
    ids = log_probs[:int(valid_length)].argmax(-1).detach().cpu().tolist()
    output: List[int] = []
    previous = None
    for token in ids:
        token = int(token)
        if token != previous:
            if token != blank_idx:
                output.append(token)
            previous = token
    return output


def token_ids_to_sentence(token_ids: List[int], id2word: Mapping[int, str]) -> str:
    words: List[str] = []
    for idx in token_ids:
        word = id2word.get(int(idx), f'<ID:{idx}>')
        if word.strip().casefold() in {'<pad>', '[pad]', 'pad'}:
            continue
        words.append(word)
    return ' '.join(words).strip()


def run_model6_inference(model, device, features: np.ndarray, blank_idx: int,
                         id2word: Mapping[int, str], use_amp: bool,
                         min_frames: int):
    features = np.ascontiguousarray(features[:MAX_SEQ_LEN], dtype=np.float32)
    if len(features) < min_frames:
        raise ValueError(f'Too few model frames ({len(features)}). Sign a little longer.')

    x = torch.from_numpy(features).unsqueeze(0).to(device, non_blocking=True)
    lengths = torch.tensor([x.shape[1]], device=device, dtype=torch.long)
    padding_mask = torch.zeros((1, x.shape[1]), device=device, dtype=torch.bool)

    amp_context = (
        torch.autocast(device_type='cuda', dtype=torch.float16)
        if device.type == 'cuda' and use_amp
        else contextlib.nullcontext()
    )

    if device.type == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), amp_context:
        log_probs, output_lengths, _ = model(x, lengths, padding_mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    model_ms = (time.perf_counter() - started) * 1000.0

    valid_len = int(output_lengths[0].item())
    token_ids = greedy_ctc_decode(log_probs[0], valid_len, blank_idx)
    sentence = token_ids_to_sentence(token_ids, id2word)

    return {
        'sentence': sentence,
        'token_ids': token_ids,
        'input_frames': int(len(features)),
        'output_steps': valid_len,
        'model_ms': float(model_ms),
    }


def detection_summary(flags: List[DetectionFlags]) -> Dict[str, float]:
    if not flags:
        return {'pose': 0.0, 'face': 0.0, 'left_hand': 0.0, 'right_hand': 0.0}
    n = len(flags)
    return {
        'pose': sum(f.pose for f in flags) / n,
        'face': sum(f.face for f in flags) / n,
        'left_hand': sum(f.left_hand for f in flags) / n,
        'right_hand': sum(f.right_hand for f in flags) / n,
    }


# -----------------------------------------------------------------------------
# 8. Async prediction worker - preview stays responsive after Stop
# -----------------------------------------------------------------------------
@dataclass
class PredictionMessage:
    status: str
    sentence: str
    info: str
    payload: Optional[dict] = None


class AsyncPredictor:
    def __init__(self, config: AppConfig, worker: LandmarkWorker, model, device,
                 blank_idx: int, id2word: Mapping[int, str]):
        self.config = config
        self.worker = worker
        self.model = model
        self.device = device
        self.blank_idx = blank_idx
        self.id2word = id2word
        self.jobs: queue.Queue = queue.Queue(maxsize=1)
        self.results: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._busy = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='prediction-worker')
        self.thread.start()

    def busy(self) -> bool:
        return self._busy.is_set()

    def submit(self) -> bool:
        if self.busy() or not self.jobs.empty():
            return False
        try:
            self.jobs.put_nowait(True)
            return True
        except queue.Full:
            return False

    def poll(self) -> Optional[PredictionMessage]:
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)

    def _publish(self, message: PredictionMessage) -> None:
        while True:
            try:
                self.results.get_nowait()
            except queue.Empty:
                break
        try:
            self.results.put_nowait(message)
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                self.jobs.task_done()
                break

            self._busy.set()
            try:
                self.worker.flush(timeout=2.0)
                raw, times_s, mp_times, flags, errors = self.worker.snapshot()
                message = self._predict(raw, times_s, mp_times, flags, errors)
                self._publish(message)
            except Exception as exc:
                self._publish(PredictionMessage(
                    status='error',
                    sentence='',
                    info=f'Prediction failed: {type(exc).__name__}: {exc}',
                ))
            finally:
                self._busy.clear()
                self.jobs.task_done()

    def _predict(self, raw, times_s, mp_times, flags, errors) -> PredictionMessage:
        if raw.ndim != 2 or len(raw) < self.config.min_raw_samples:
            count = len(raw) if raw.ndim == 2 else 0
            extra = f' Last MediaPipe error: {errors[-1]}' if errors else ''
            return PredictionMessage(
                status='error',
                sentence='',
                info=f'Too few landmark samples ({count}). Record a little longer.{extra}',
            )

        started = time.perf_counter()
        model_features = resample_landmarks_vectorized(raw, times_s, TARGET_FPS)
        result = run_model6_inference(
            self.model, self.device, model_features,
            self.blank_idx, self.id2word,
            self.config.use_amp, self.config.min_raw_samples,
        )
        total_ms = (time.perf_counter() - started) * 1000.0

        sentence = result['sentence'] or '<blank / no words decoded>'
        avg_mp = float(mp_times.mean()) if len(mp_times) else float('nan')
        duration = float(times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0
        rates = detection_summary(flags)

        info = (
            f'raw={len(raw)} / {duration:.2f}s | 25fps={len(model_features)} | '
            f'MP={avg_mp:.0f}ms | Model6={result["model_ms"]:.1f}ms | '
            f'pose={rates["pose"]*100:.0f}% face={rates["face"]*100:.0f}% '
            f'L={rates["left_hand"]*100:.0f}% R={rates["right_hand"]*100:.0f}%'
        )

        payload = {
            **result,
            'sentence': sentence,
            'raw_samples': int(len(raw)),
            'duration_s': duration,
            'avg_mediapipe_ms': avg_mp,
            'total_post_stop_ms': total_ms,
            'detection_rates': rates,
            'recoverable_mediapipe_errors': errors,
        }

        if self.config.save_features:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            np.save(self.config.output_dir / f'live_raw_{stamp}.npy', raw)
            np.save(self.config.output_dir / f'live_25fps_{stamp}.npy', model_features)
            with (self.config.output_dir / f'live_result_{stamp}.json').open('w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        print('\n' + '=' * 72)
        print('MODEL 6 PREDICTION')
        print('=' * 72)
        print('Sentence :', sentence)
        print('Token IDs:', result['token_ids'])
        print('Info     :', info)
        print('=' * 72)

        return PredictionMessage('ok', sentence, info, payload)


# -----------------------------------------------------------------------------
# 9. Unicode/Bangla rendering
# -----------------------------------------------------------------------------
def find_bengali_font(manual_path: str = '') -> Optional[str]:
    if manual_path and Path(manual_path).is_file():
        return manual_path

    candidates = [
        r'C:\Windows\Fonts\Nirmala.ttf',
        r'C:\Windows\Fonts\vrinda.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansBengali-Regular.ttf',
        '/usr/share/fonts/truetype/lohit-bengali/lohit-bengali.ttf',
        '/Library/Fonts/NotoSansBengali-Regular.ttf',
        '/System/Library/Fonts/Supplemental/Noto Sans Bengali.ttf',
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


class UnicodeRenderer:
    def __init__(self, font_path: Optional[str]):
        self.font_path = font_path
        self._cache: Dict[int, ImageFont.FreeTypeFont] = {}

    def available(self) -> bool:
        return bool(self.font_path)

    def draw(self, bgr: np.ndarray, text: str, xy: Tuple[int, int], size: int = 32) -> np.ndarray:
        if not text or not self.font_path:
            return bgr
        font = self._cache.get(size)
        if font is None:
            font = ImageFont.truetype(self.font_path, size)
            self._cache[size] = font
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.text(xy, text, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


# -----------------------------------------------------------------------------
# 10. Camera and display
# -----------------------------------------------------------------------------
def open_camera(index: int, profile: PerformanceProfile) -> cv2.VideoCapture:
    system = platform.system().lower()
    backends = []
    if system.startswith('win'):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    cap = None
    for backend in backends:
        candidate = cv2.VideoCapture(index, backend)
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f'Could not open webcam index {index}. Try --camera 1 or --camera 2. '
            'Also close Zoom/Teams/browser apps that may be holding the camera.'
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, profile.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, profile.camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def make_display_frame(frame: np.ndarray, config: AppConfig, renderer: UnicodeRenderer,
                       recording: bool, predicting: bool, elapsed: float,
                       sample_count: int, prediction: str, info_line: str) -> np.ndarray:
    preview = cv2.flip(frame, 1) if config.mirror_preview else frame.copy()
    h, w = preview.shape[:2]
    panel_h = 150
    canvas = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    canvas[:h] = preview

    if recording:
        status = f'RECORDING {elapsed:5.1f}s | samples={sample_count} | SPACE=stop+predict'
        status_color = (0, 80, 255)
    elif predicting:
        status = 'PROCESSING... camera remains live | please wait'
        status_color = (0, 200, 255)
    else:
        status = 'READY | SPACE=start | C=clear | Q=quit'
        status_color = (0, 220, 0)

    cv2.rectangle(canvas, (0, 0), (w, 54), (20, 20, 20), -1)
    cv2.putText(canvas, status, (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.70, status_color, 2, cv2.LINE_AA)

    cv2.putText(canvas, 'Prediction:', (16, h + 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (220, 220, 220), 2, cv2.LINE_AA)

    if prediction and renderer.available():
        canvas = renderer.draw(canvas, prediction, (165, h + 12), size=32)
    elif prediction:
        cv2.putText(canvas, 'Bangla prediction printed in terminal', (165, h + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)

    if info_line:
        # Keep one compact line in GUI; full diagnostics stay in terminal/JSON.
        compact = info_line[:max(40, int(w / 7.5))]
        cv2.putText(canvas, compact, (16, h + 95), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (180, 180, 180), 1, cv2.LINE_AA)

    return canvas


# -----------------------------------------------------------------------------
# 11. App
# -----------------------------------------------------------------------------
def build_config(args) -> AppConfig:
    default_root = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    root = Path(args.root).resolve() if args.root else default_root
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else root / 'model6_best_wer_checkpoint.pt'
    vocab = Path(args.vocab).resolve() if args.vocab else root / 'bangla_vocab_tsv_fixed.json'
    mp_dir = root / 'mediapipe_models'
    output_dir = root / 'model6_live_output'

    if not checkpoint.is_file():
        raise FileNotFoundError(f'Model 6 checkpoint not found:\n  {checkpoint}')
    if not vocab.is_file():
        raise FileNotFoundError(f'Vocabulary JSON not found:\n  {vocab}')

    return AppConfig(
        root=root,
        checkpoint_path=checkpoint,
        vocab_path=vocab,
        mediapipe_model_dir=mp_dir,
        output_dir=output_dir,
        camera_index=args.camera,
        profile=PROFILES[args.profile],
        prefer_ema=not args.no_ema,
        use_amp=not args.no_amp,
        mirror_preview=not args.no_mirror,
        save_features=not args.no_save,
        bengali_font_path=args.font or '',
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Model 6 efficient local live webcam inference')
    parser.add_argument('--root', default='', help='Project folder containing checkpoint/vocab')
    parser.add_argument('--checkpoint', default='', help='Checkpoint path override')
    parser.add_argument('--vocab', default='', help='Vocabulary JSON path override')
    parser.add_argument('--camera', type=int, default=0, help='Webcam index (default: 0)')
    parser.add_argument('--profile', choices=sorted(PROFILES), default='balanced',
                        help='fast / balanced / accurate')
    parser.add_argument('--font', default='', help='Optional Bengali .ttf/.otf font path')
    parser.add_argument('--no-ema', action='store_true', help='Do not prefer EMA checkpoint weights')
    parser.add_argument('--no-amp', action='store_true', help='Disable CUDA autocast')
    parser.add_argument('--no-mirror', action='store_true', help='Do not mirror preview')
    parser.add_argument('--no-save', action='store_true', help='Do not save captured features/results')
    args = parser.parse_args(argv)

    if running_in_remote_notebook():
        raise RuntimeError(
            'This LIVE v4.0 app must run on your LOCAL PC. '
            'Kaggle/Colab cannot directly expose your laptop webcam to cv2.VideoCapture(0).'
        )

    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('MODEL 6 - EFFICIENT LOCAL LIVE WEBCAM v4.0')
    print('=' * 72)
    print('Root            :', config.root)
    print('Checkpoint      :', config.checkpoint_path.name)
    print('Vocabulary      :', config.vocab_path.name)
    print('Profile         :', config.profile)
    print('Camera index    :', config.camera_index)
    print('=' * 72)

    word2id, id2word, vocab_size = load_vocab(config.vocab_path)
    blank_idx = resolve_special_token(
        word2id, ['<BLANK>', '<blank>', '[BLANK]', 'blank'], 'BLANK token'
    )
    if blank_idx != 0:
        raise ValueError(f'Expected BLANK index 0 for this Model 6 training run; got {blank_idx}.')

    model, device, meta = load_model6(config, vocab_size, blank_idx)
    print('Device          :', device)
    if device.type == 'cuda':
        print('GPU             :', torch.cuda.get_device_name(0))
    print('Vocab size      :', vocab_size)
    print('Blank index     :', blank_idx)
    print('Checkpoint state:', meta['state_key'])
    print('Epoch           :', meta['epoch'])
    print('Checkpoint kind :', meta['checkpoint_kind'])
    print('Parameters      :', f"{meta['param_count']:,}")

    # Smoke test exact temporal shape.
    with torch.inference_mode():
        smoke_x = torch.zeros(1, 121, FEATURE_DIM, device=device)
        smoke_len = torch.tensor([121], device=device, dtype=torch.long)
        smoke_mask = torch.zeros(1, 121, device=device, dtype=torch.bool)
        smoke_y, smoke_out_len, _ = model(smoke_x, smoke_len, smoke_mask)
    if tuple(smoke_y.shape) != (1, 61, vocab_size) or int(smoke_out_len[0]) != 61:
        raise RuntimeError(f'Model smoke test failed: output={tuple(smoke_y.shape)}, lengths={smoke_out_len.tolist()}')
    del smoke_x, smoke_len, smoke_mask, smoke_y, smoke_out_len
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    ensure_mediapipe_assets(config.mediapipe_model_dir)
    extractor = MediaPipe1662Extractor(config)
    landmark_worker = LandmarkWorker(extractor)
    predictor = AsyncPredictor(config, landmark_worker, model, device, blank_idx, id2word)
    renderer = UnicodeRenderer(find_bengali_font(config.bengali_font_path))
    cap = open_camera(config.camera_index, config.profile)

    print('Landmark FPS    :', config.profile.landmark_sample_fps)
    print('MP input width  :', config.profile.mediapipe_width)
    print('Model timeline  :', TARGET_FPS, 'FPS')
    print('Bengali font    :', renderer.font_path or 'not auto-detected')
    print('Controls        : SPACE start/stop+predict | C clear | Q/ESC quit')
    print('=' * 72)

    recording = False
    record_started_at = 0.0
    last_submit_at = 0.0
    prediction = ''
    info_line = 'Ready'
    window_name = 'Model 6 - Live Bangla Sign Recognition v4.0'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                info_line = 'Camera frame read failed.'
                print(info_line)
                break

            now = time.monotonic()
            elapsed = now - record_started_at if recording else 0.0

            # Poll completed prediction without blocking preview.
            message = predictor.poll()
            if message is not None:
                prediction = message.sentence
                info_line = message.info

            if recording:
                min_dt = 1.0 / config.profile.landmark_sample_fps
                if now - last_submit_at >= min_dt:
                    landmark_worker.submit_latest(frame.copy(), elapsed)
                    last_submit_at = now

                if elapsed >= MAX_RECORD_SECONDS:
                    recording = False
                    info_line = 'Max duration reached. Processing...'
                    predictor.submit()

            display_frame = make_display_frame(
                frame, config, renderer,
                recording=recording,
                predicting=predictor.busy(),
                elapsed=elapsed,
                sample_count=landmark_worker.count(),
                prediction=prediction,
                info_line=info_line,
            )
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break

            if key in (ord('c'), ord('C')):
                if predictor.busy():
                    info_line = 'Prediction is running; wait a moment before clearing.'
                else:
                    recording = False
                    landmark_worker.reset()
                    prediction = ''
                    info_line = 'Cleared'
                continue

            if key == 32:  # SPACE
                if predictor.busy():
                    info_line = 'Prediction is running; wait for result.'
                    continue

                if not recording:
                    landmark_worker.reset()
                    recording = True
                    record_started_at = time.monotonic()
                    last_submit_at = 0.0
                    prediction = ''
                    info_line = 'Recording...'
                    print('\nRecording started. Perform ONE complete sentence, then press SPACE.')
                else:
                    recording = False
                    info_line = 'Processing...'
                    print('Recording stopped. Processing in background...')
                    if not predictor.submit():
                        info_line = 'Could not start prediction worker.'

    finally:
        cap.release()
        predictor.close()
        landmark_worker.close()
        extractor.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
