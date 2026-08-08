from __future__ import annotations

import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from app.config import Settings
from app.errors import EmptyVideoError, MediaPipeFailureError, RecordingTooShortError, SequenceTooLongError, UnsupportedVideoError
from app.utils.sequence import FEATURE_DIM


try:
    cv2.setNumThreads(1)
except Exception:
    pass


MP_ASSETS = {
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
}


def ensure_mediapipe_assets(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MP_ASSETS.items():
        target = model_dir / filename
        if target.exists() and target.stat().st_size > 1024:
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)
        except Exception as exc:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(
                f"Could not download MediaPipe asset {filename}. "
                f"Download it manually and place it at {target}. Original error: {exc}"
            ) from exc


@dataclass(frozen=True)
class DetectionFlags:
    pose: bool
    face: bool
    left_hand: bool
    right_hand: bool


@dataclass
class LandmarkExtractionResult:
    features: np.ndarray
    times_s: np.ndarray
    raw_video_frames: int
    sampled_frames: int
    duration_seconds: float
    video_decode_ms: float
    mediapipe_ms: float
    detections: Dict[str, float]


@dataclass
class LiveFrameExtractionResult:
    features: np.ndarray
    times_s: np.ndarray
    flags: List[DetectionFlags]
    mediapipe_ms_by_frame: List[float]
    mediapipe_ms: float


def detection_summary(flags: List[DetectionFlags]) -> Dict[str, float]:
    if not flags:
        return {"pose_rate": 0.0, "face_rate": 0.0, "left_hand_rate": 0.0, "right_hand_rate": 0.0}
    n = len(flags)
    return {
        "pose_rate": sum(f.pose for f in flags) / n,
        "face_rate": sum(f.face for f in flags) / n,
        "left_hand_rate": sum(f.left_hand for f in flags) / n,
        "right_hand_rate": sum(f.right_hand for f in flags) / n,
    }


class MediaPipe1662Extractor:
    """Exact feature order: pose132 + face1404 + left63 + right63."""

    def __init__(self, settings: Settings):
        self.settings = settings
        BaseOptions = mp.tasks.BaseOptions
        RunningMode = mp.tasks.vision.RunningMode

        self.pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(settings.mediapipe_asset_dir / "pose_landmarker.task")),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.hand_detector = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(settings.mediapipe_asset_dir / "hand_landmarker.task")),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.face_detector = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(settings.mediapipe_asset_dir / "face_landmarker.task")),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._last_timestamp_ms = -1
        self._timestamp_offset_ms = 0

    def close(self) -> None:
        for detector in (self.pose_detector, self.hand_detector, self.face_detector):
            try:
                detector.close()
            except Exception:
                pass

    def _sanitize_timestamp_ms(self, timestamp_ms: int) -> int:
        raw_ts = int(timestamp_ms)
        ts = raw_ts + self._timestamp_offset_ms
        if ts <= self._last_timestamp_ms:
            self._timestamp_offset_ms = self._last_timestamp_ms + 1 - raw_ts
            ts = raw_ts + self._timestamp_offset_ms
        self._last_timestamp_ms = ts
        return ts

    def _resize(self, rgb: np.ndarray) -> np.ndarray:
        target_width = self.settings.mediapipe_width
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
            arr = np.asarray([[lm.x, lm.y, lm.z] for lm in pts], dtype=np.float32).reshape(-1)
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
                pts = np.asarray([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32).reshape(-1)
                if pts.size != 63:
                    continue

                label = ""
                if i < len(result.handedness) and result.handedness[i]:
                    label = str(result.handedness[i][0].category_name)

                if label.casefold() == "left":
                    left = pts
                    left_found = True
                elif label.casefold() == "right":
                    right = pts
                    right_found = True

        return left, right, left_found, right_found

    def extract(self, bgr_frame: np.ndarray, capture_elapsed_s: float):
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
            raise RuntimeError(f"Unexpected feature shape: {feature.shape}")
        if not np.isfinite(feature).all():
            feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

        return feature, mp_ms, DetectionFlags(pose_found, face_found, left_found, right_found)


class MediaPipeService:
    def __init__(self, settings: Settings):
        self.settings = settings
        ensure_mediapipe_assets(settings.mediapipe_asset_dir)
        self.extractor = MediaPipe1662Extractor(settings)
        self._lock = threading.Lock()

    def close(self) -> None:
        self.extractor.close()

    def extract_from_video(self, path: Path) -> LandmarkExtractionResult:
        with self._lock:
            return self._extract_from_video_locked(path)

    def extract_from_frames(self, frames: Sequence[Tuple[np.ndarray, float]]) -> LandmarkExtractionResult:
        with self._lock:
            return self._extract_from_frames_locked(frames)

    def extract_live_frame_batch(self, frames: Sequence[Tuple[np.ndarray, float]]) -> LiveFrameExtractionResult:
        with self._lock:
            return self._extract_live_frame_batch_locked(frames)

    def _extract_live_frame_batch_locked(self, frames: Sequence[Tuple[np.ndarray, float]]) -> LiveFrameExtractionResult:
        ordered = sorted(frames, key=lambda item: item[1])
        features: List[np.ndarray] = []
        times: List[float] = []
        flags: List[DetectionFlags] = []
        mediapipe_ms_by_frame: List[float] = []
        mediapipe_ms = 0.0

        for frame, timestamp_s in ordered:
            if frame is None or frame.size <= 0:
                continue
            feature, mp_ms, detection = self.extractor.extract(frame, float(timestamp_s))
            features.append(feature)
            times.append(float(timestamp_s))
            flags.append(detection)
            mediapipe_ms_by_frame.append(float(mp_ms))
            mediapipe_ms += mp_ms

        if not features:
            raise MediaPipeFailureError("No live frames were usable for MediaPipe.")

        return LiveFrameExtractionResult(
            features=np.ascontiguousarray(np.stack(features, axis=0), dtype=np.float32),
            times_s=np.asarray(times, dtype=np.float64),
            flags=flags,
            mediapipe_ms_by_frame=mediapipe_ms_by_frame,
            mediapipe_ms=float(mediapipe_ms),
        )

    def _extract_from_frames_locked(self, frames: Sequence[Tuple[np.ndarray, float]]) -> LandmarkExtractionResult:
        if len(frames) < 2:
            raise RecordingTooShortError("At least two live frames are required.")

        ordered = sorted(frames, key=lambda item: item[1])
        features: List[np.ndarray] = []
        times: List[float] = []
        flags: List[DetectionFlags] = []
        mediapipe_ms = 0.0

        for frame, timestamp_s in ordered:
            if frame is None or frame.size <= 0:
                continue
            feature, mp_ms, detection = self.extractor.extract(frame, float(timestamp_s))
            features.append(feature)
            times.append(float(timestamp_s))
            flags.append(detection)
            mediapipe_ms += mp_ms

        if not features:
            raise MediaPipeFailureError("No live frames were usable for MediaPipe.")

        duration_seconds = max(float(times[-1] - times[0]) if len(times) > 1 else 0.0, len(features) / float(self.settings.target_fps))
        if duration_seconds < self.settings.min_video_seconds:
            raise RecordingTooShortError(detail=f"Duration {duration_seconds:.2f}s is below the minimum.")

        rates = detection_summary(flags)
        if rates["pose_rate"] < 0.05 and (rates["left_hand_rate"] + rates["right_hand_rate"]) < 0.05:
            raise MediaPipeFailureError("Pose and hand landmarks were nearly absent.")

        return LandmarkExtractionResult(
            features=np.ascontiguousarray(np.stack(features, axis=0), dtype=np.float32),
            times_s=np.asarray(times, dtype=np.float64),
            raw_video_frames=len(ordered),
            sampled_frames=len(features),
            duration_seconds=duration_seconds,
            video_decode_ms=0.0,
            mediapipe_ms=float(mediapipe_ms),
            detections=rates,
        )

    def _extract_from_video_locked(self, path: Path) -> LandmarkExtractionResult:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise UnsupportedVideoError("OpenCV could not open the uploaded video.")

        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = self._trusted_video_fps(raw_fps)
        frame_count_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        metadata_duration = frame_count_meta / fps if fps > 0 and frame_count_meta > 0 else 0.0
        if metadata_duration > self.settings.max_video_seconds + 0.75:
            cap.release()
            raise SequenceTooLongError(detail=f"Video duration {metadata_duration:.2f}s exceeds the limit.")

        features: List[np.ndarray] = []
        times: List[float] = []
        flags: List[DetectionFlags] = []
        decode_ms = 0.0
        mediapipe_ms = 0.0
        frame_index = 0
        next_sample_s = 0.0
        sample_fps = min(float(self.settings.target_fps), max(1.0, float(self.settings.video_sample_fps)))
        sample_period = 1.0 / sample_fps
        last_timestamp_s = 0.0

        try:
            while True:
                started_decode = time.perf_counter()
                ok, frame = cap.read()
                decode_ms += (time.perf_counter() - started_decode) * 1000.0
                if not ok or frame is None:
                    break

                pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if fps > 0:
                    timestamp_s = frame_index / fps
                elif pos_ms > 0:
                    timestamp_s = pos_ms / 1000.0
                else:
                    timestamp_s = frame_index * sample_period

                last_timestamp_s = max(last_timestamp_s, timestamp_s)
                if timestamp_s + 1e-9 >= next_sample_s:
                    feature, mp_ms, detection = self.extractor.extract(frame, timestamp_s)
                    features.append(feature)
                    times.append(timestamp_s)
                    flags.append(detection)
                    mediapipe_ms += mp_ms

                    while next_sample_s <= timestamp_s + 1e-9:
                        next_sample_s += sample_period

                    if len(features) >= self.settings.max_sequence_length:
                        break

                frame_index += 1
                if timestamp_s > self.settings.max_video_seconds + 0.75:
                    raise SequenceTooLongError(detail=f"Video duration exceeds {self.settings.max_video_seconds:.2f}s.")
        finally:
            cap.release()

        if frame_index <= 0:
            raise EmptyVideoError()
        if not features:
            raise MediaPipeFailureError("No frames were sampled for MediaPipe.")

        sampled_duration = float((len(features) - 1) / sample_fps) if len(features) > 1 and sample_fps > 0 else 0.0
        duration_seconds = max(
            float(last_timestamp_s),
            float(metadata_duration),
            float(times[-1] - times[0] if len(times) > 1 else 0.0),
            sampled_duration,
        )
        if duration_seconds < self.settings.min_video_seconds:
            raise RecordingTooShortError(detail=f"Duration {duration_seconds:.2f}s is below the minimum.")

        rates = detection_summary(flags)
        if rates["pose_rate"] < 0.05 and (rates["left_hand_rate"] + rates["right_hand_rate"]) < 0.05:
            raise MediaPipeFailureError("Pose and hand landmarks were nearly absent.")

        return LandmarkExtractionResult(
            features=np.ascontiguousarray(np.stack(features, axis=0), dtype=np.float32),
            times_s=np.asarray(times, dtype=np.float64),
            raw_video_frames=int(frame_index),
            sampled_frames=len(features),
            duration_seconds=duration_seconds,
            video_decode_ms=float(decode_ms),
            mediapipe_ms=float(mediapipe_ms),
            detections=rates,
        )

    @staticmethod
    def _trusted_video_fps(fps: float) -> float:
        fps = float(fps or 0.0)
        if 5.0 <= fps <= 120.0:
            return fps
        return 0.0
