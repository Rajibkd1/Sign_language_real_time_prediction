"""Evaluate the deployed upload pipeline against a private, labeled JSONL manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.utils.evaluation import latency_summary, score_records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_fingerprints() -> dict[str, str]:
    """Identify saved Python source even when a benchmark is run before commit."""
    backend = Path(__file__).resolve().parents[1]
    return {path.relative_to(backend).as_posix(): sha256(path)
            for folder in ("app", "scripts") for path in sorted((backend / folder).rglob("*.py"))}


def processor_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def read_manifest(path: Path, *, require_videos: bool = True) -> list[dict]:
    rows, ids = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in (("id", "video", "reference", "group_id") if require_videos else ("id", "reference", "group_id")):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Line {number}: {key} must be a nonempty string.")
        if row["id"] in ids:
            raise ValueError(f"Duplicate sample ID: {row['id']}")
        ids.add(row["id"])
        if require_videos:
            video = (path.parent / row["video"]).resolve()
            if not video.is_file():
                raise ValueError(f"Line {number}: video file does not exist.")
            row = {**row, "video": video}
        rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty.")
    return rows


def evaluate_samples(rows: list[dict], service) -> tuple[list[dict], list[float], list[float]]:
    samples, wall_times, model_times = [], [], []
    for row in rows:
        sample = {"id": row["id"], "group_id": row["group_id"],
                  "video_sha256": sha256(row["video"]), "reference": row["reference"]}
        started = time.perf_counter()
        try:
            response = service.predict_video(row["video"], save_history=False)
            sample.update(status="ok", prediction=response.prediction,
                          response=response.model_dump(mode="json"))
            model_times.append(response.timings.model_ms)
        except Exception as exc:
            # A decode failure still counts as deletions of every reference word.
            sample.update(status="error", prediction="", error_type=type(exc).__name__)
        sample["pipeline_wall_ms"] = (time.perf_counter() - started) * 1000.0
        wall_times.append(sample["pipeline_wall_ms"])
        samples.append(sample)
        print(f"{sample['id']}: {sample['status']}")
    return samples, wall_times, model_times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSONL: id, video, reference, group_id")
    parser.add_argument("--split", required=True, choices=("validation", "test", "demo"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--normalization", choices=("nfc-whitespace", "bdsl-v1"), default="nfc-whitespace")
    parser.add_argument("--exclude-manifest", action="append", default=[], type=Path,
                        help="Train/validation manifests whose sentence groups must not overlap; repeatable")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve earlier evidence.")
    evaluated_groups = {row["group_id"] for row in rows}
    excluded = []
    for other in args.exclude_manifest:
        other_rows = read_manifest(other, require_videos=False)
        if evaluated_groups & {row["group_id"] for row in other_rows}:
            parser.error("Sentence groups overlap an excluded manifest.")
        excluded.append({"sha256": sha256(other), "groups": len({r['group_id'] for r in other_rows})})

    # Import heavy dependencies only after validating the inputs.
    import cv2
    import mediapipe
    import torch
    from app.config import get_settings
    from app.services.history_service import HistoryService
    from app.services.inference_service import InferenceService

    settings = get_settings()
    service = InferenceService(settings, HistoryService(settings))
    try:
        samples, wall_times, model_times = evaluate_samples(rows, service)
        payload = {
            "schema_version": 1, "kind": "labeled_video_evaluation", "split": args.split,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "git_revision": revision(),
            "source_files_sha256": source_fingerprints(),
            "manifest_sha256": sha256(args.manifest), "excluded_manifests": excluded,
            "group_disjointness": "checked against supplied exclusions only; signer independence not established",
            "checkpoint_sha256": sha256(settings.checkpoint_path), "vocab_sha256": sha256(settings.vocab_path),
            "mediapipe_assets": {p.name: sha256(p) for p in sorted(settings.mediapipe_asset_dir.glob('*.task'))},
            "model": service.model_info(), "settings": asdict(settings),
            "environment": {"platform": platform.platform(), "processor": processor_name(),
                            "python": platform.python_version(), "torch": torch.__version__,
                            "opencv": cv2.__version__, "mediapipe": mediapipe.__version__,
                            "device": str(service.runtime.device), "torch_threads": torch.get_num_threads()},
            "metrics": score_records(samples, normalization=args.normalization,
                                     vocabulary=set(service.runtime.id2word.values())),
            "failed_samples": sum(s['status'] == 'error' for s in samples),
            "pipeline_wall_latency": latency_summary(wall_times),
            "selected_forward_latency_successes_only": latency_summary(model_times) if model_times else None,
            "timing_scope": "Per-video predict_video wall time, including decode, landmarks, candidate forwards and CTC; excludes startup, file hashing, HTTP transfer and browser. Selected model_ms is one chosen candidate's forward only.",
            "samples": samples,
        }
    finally:
        service.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    print(json.dumps(payload["metrics"], indent=2))
    return 1 if payload["failed_samples"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
