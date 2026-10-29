"""Build a reproducible code-only prerelease from a committed Git tree."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path


MODEL_SUFFIXES = {".pt", ".pth", ".task", ".onnx", ".safetensors", ".npy", ".npz", ".mp4", ".webm"}


def build(repo: Path, output: Path, version: str, commit: str, ci_run_url: str) -> dict:
    if not re.fullmatch(r"\d+\.\d+\.\d+-(?:alpha|beta|rc)\.\d+", version):
        raise ValueError("A prerelease version such as 0.1.0-rc.1 is required.")
    if output.exists():
        raise ValueError("Choose a new output directory; previous evidence will not be overwritten.")
    revision = subprocess.check_output(["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=repo, text=True).strip()
    raw = subprocess.check_output(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
                                   "archive", "--format=tar", revision], cwd=repo)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("Only ordinary files within the archive are supported.")
            if path.suffix.lower() in MODEL_SUFFIXES or (path.name.startswith("bangla_vocab") and path.suffix == ".json"):
                raise ValueError(f"Code-only archive leaked a research/model asset: {member.name}")
            files[member.name] = (archive.extractfile(member).read(), member.mode)
    manifest = {
        "schema_version": 1, "version": version, "git_revision": revision,
        "scope": "code-only-prerelease", "ci_run_url": ci_run_url,
        "project_checkpoint_included": False, "vocabulary_included": False,
        "mediapipe_binaries_included": False, "dataset_included": False,
        "new_video_evaluation_completed": False, "open_issues": [2, 3],
        "files": {name: hashlib.sha256(data).hexdigest() for name, (data, _) in sorted(files.items())},
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    output.mkdir(parents=True)
    manifest_path = output / "RELEASE_MANIFEST.json"
    manifest_path.write_bytes(manifest_bytes)
    zip_path = output / f"BdSL-Vision-{version}-source.zip"
    prefix = f"BdSL-Vision-{version}/"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, (data, mode) in sorted({**files, "RELEASE_MANIFEST.json": (manifest_bytes, 0o644)}.items()):
            info = zipfile.ZipInfo(prefix + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100000 | (0o755 if mode & 0o111 else 0o644)) << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, data)
    checksums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in (zip_path, manifest_path))
    (output / "SHA256SUMS").write_text(checksums, encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ci-run-url", required=True)
    args = parser.parse_args()
    manifest = build(Path(__file__).resolve().parents[2], args.output, args.version, args.commit, args.ci_run_url)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
