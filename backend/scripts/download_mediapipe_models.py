from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


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


def download_assets(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MP_ASSETS.items():
        target = target_dir / filename
        if target.exists() and target.stat().st_size > 1024:
            print(f"Already present: {target}")
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {filename}")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)
            print(f"Saved {target}")
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MediaPipe task models for BdSL Vision.")
    parser.add_argument(
        "--target",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "mediapipe"),
        help="Directory where .task files will be stored.",
    )
    args = parser.parse_args()
    download_assets(Path(args.target).expanduser().resolve())


if __name__ == "__main__":
    main()

