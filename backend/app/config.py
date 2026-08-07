from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent


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

    cors_origins: Tuple[str, ...] = _csv_from_env(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )

    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "80"))
    max_video_seconds: float = float(os.getenv("MAX_VIDEO_SECONDS", str(384 / 25.0)))
    min_video_seconds: float = float(os.getenv("MIN_VIDEO_SECONDS", "0.65"))
    max_sequence_length: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "384"))
    target_fps: float = float(os.getenv("TARGET_FPS", "25.0"))
    mediapipe_width: int = int(os.getenv("MEDIAPIPE_WIDTH", "640"))
    min_model_frames: int = int(os.getenv("MIN_MODEL_FRAMES", "8"))
    prefer_ema: bool = _bool_from_env("PREFER_EMA", True)
    use_amp: bool = _bool_from_env("USE_CUDA_AMP", True)
    enable_history: bool = _bool_from_env("ENABLE_HISTORY", True)
    max_history_items: int = int(os.getenv("MAX_HISTORY_ITEMS", "200"))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def get_settings() -> Settings:
    return Settings()

