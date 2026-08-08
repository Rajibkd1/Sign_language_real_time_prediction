from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

try:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env", override=False)
    load_dotenv(BACKEND_DIR / ".env.local", override=False)
except Exception:
    pass


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return Path(raw).expanduser().resolve()


def _csv_from_env(name: str, default: str) -> Tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "BdSL Vision"
    api_prefix: str = "/api"
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))

    checkpoint_path: Path = _path_from_env(
        "MODEL_CHECKPOINT_PATH",
        BACKEND_DIR / "models" / "model6_best_wer_checkpoint.pt",
    )
    vocab_path: Path = _path_from_env(
        "VOCAB_PATH",
        BACKEND_DIR / "models" / "bangla_vocab_tsv_fixed.json",
    )
    mediapipe_asset_dir: Path = _path_from_env(
        "MEDIAPIPE_ASSET_DIR",
        BACKEND_DIR / "assets" / "mediapipe",
    )
    history_path: Path = _path_from_env("HISTORY_PATH", BACKEND_DIR / "data" / "history.json")
    inference_artifact_dir: Path = _path_from_env("INFERENCE_ARTIFACT_DIR", BACKEND_DIR / "data" / "artifacts")

    cors_origins: Tuple[str, ...] = _csv_from_env(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )

    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "80"))
    max_video_seconds: float = float(os.getenv("MAX_VIDEO_SECONDS", str(384 / 25.0)))
    min_video_seconds: float = float(os.getenv("MIN_VIDEO_SECONDS", "0.65"))
    max_sequence_length: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "384"))
    target_fps: float = float(os.getenv("TARGET_FPS", "25.0"))
    video_sample_fps: float = float(os.getenv("VIDEO_SAMPLE_FPS", str(os.getenv("TARGET_FPS", "25.0"))))
    mediapipe_width: int = int(os.getenv("MEDIAPIPE_WIDTH", "640"))
    min_model_frames: int = int(os.getenv("MIN_MODEL_FRAMES", "8"))
    live_start_motion_threshold: float = float(os.getenv("LIVE_START_MOTION_THRESHOLD", "0.026"))
    live_active_motion_threshold: float = float(os.getenv("LIVE_ACTIVE_MOTION_THRESHOLD", "0.012"))
    live_min_hand_motion: float = float(os.getenv("LIVE_MIN_HAND_MOTION", "0.0030"))
    live_min_active_motion_frames: int = int(os.getenv("LIVE_MIN_ACTIVE_MOTION_FRAMES", "3"))
    live_min_average_motion: float = float(os.getenv("LIVE_MIN_AVERAGE_MOTION", "0.014"))
    live_start_consecutive_frames: int = int(os.getenv("LIVE_START_CONSECUTIVE_FRAMES", "2"))
    live_end_silence_seconds: float = float(os.getenv("LIVE_END_SILENCE_SECONDS", "0.90"))
    live_draft_activity_grace_seconds: float = float(os.getenv("LIVE_DRAFT_ACTIVITY_GRACE_SECONDS", "0.20"))
    live_preroll_seconds: float = float(os.getenv("LIVE_PREROLL_SECONDS", "0.40"))
    prefer_ema: bool = _bool_from_env("PREFER_EMA", True)
    use_amp: bool = _bool_from_env("USE_CUDA_AMP", True)
    enable_history: bool = _bool_from_env("ENABLE_HISTORY", True)
    save_inference_artifacts: bool = _bool_from_env("SAVE_INFERENCE_ARTIFACTS", False)
    max_history_items: int = int(os.getenv("MAX_HISTORY_ITEMS", "200"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def get_settings() -> Settings:
    return Settings()
