from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


ASSET_MANIFEST = Path(__file__).resolve().parents[1] / "assets" / "mediapipe" / "manifest.json"


def matches_manifest(path: Path, record: dict) -> bool:
    if not path.is_file() or path.stat().st_size != record["size_bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == record["sha256"]


def download_assets(target_dir: Path, manifest_path: Path = ASSET_MANIFEST) -> None:
    assets = json.loads(manifest_path.read_text(encoding="utf-8"))["assets"]
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, record in assets.items():
        if Path(filename).name != filename or not filename.endswith(".task"):
            raise ValueError("Manifest asset names must be .task basenames.")
        target = target_dir / filename
        if matches_manifest(target, record):
            print(f"Verified: {target}")
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {filename}")
        try:
            urllib.request.urlretrieve(record["url"], tmp)
            if not matches_manifest(tmp, record):
                raise ValueError(f"Downloaded asset failed size/SHA-256 verification: {filename}")
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

