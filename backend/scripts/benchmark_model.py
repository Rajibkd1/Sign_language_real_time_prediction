"""Synthetic-input latency benchmark. This does not measure recognition accuracy."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_videos import revision, sha256, source_fingerprints
from app.utils.evaluation import latency_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", default=[128, 384])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 1 or args.threads < 1:
        parser.error("runs, warmup and threads must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new path.")
    import numpy as np
    import torch
    from app.config import get_settings
    from app.model.checkpoint import ModelRuntime
    from app.model.model6 import FEATURE_DIM
    settings = get_settings()
    if any(n < settings.min_model_frames or n > settings.max_sequence_length for n in args.frames):
        parser.error("Frame lengths must be within the configured min/max model length.")
    torch.set_num_threads(args.threads)
    runtime = ModelRuntime.load(settings)
    records = []
    for frames in args.frames:
        features = np.zeros((frames, FEATURE_DIM), dtype=np.float32)
        for _ in range(args.warmup):
            runtime.predict(features, settings.min_model_frames)
        wall, forward = [], []
        for _ in range(args.runs):
            if runtime.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = runtime.predict(features, settings.min_model_frames)
            if runtime.device.type == "cuda":
                torch.cuda.synchronize()
            wall.append((time.perf_counter() - start) * 1000.0)
            forward.append(prediction.model_ms)
        records.append({"frames": frames, "batch_size": 1,
                        "forward": latency_summary(forward), "predict_wall": latency_summary(wall)})
    payload = {
        "schema_version": 1, "kind": "synthetic_latency_only", "accuracy_measured": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "git_revision": revision(),
        "source_files_sha256": source_fingerprints(),
        "checkpoint_sha256": sha256(settings.checkpoint_path), "vocab_sha256": sha256(settings.vocab_path),
        "model": runtime.info(settings.max_sequence_length, settings.target_fps),
        "environment": {"platform": platform.platform(), "processor": platform.processor(),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "torch_threads": torch.get_num_threads(),
                        "cuda_device": torch.cuda.get_device_name() if runtime.device.type == "cuda" else None},
        "input": "all-zero float32 [T,1662]; synthetic, not signing video",
        "warmup_per_length": args.warmup, "runs_per_length": args.runs, "use_amp": runtime.use_amp,
        "scope": "forward is timed model forward only; predict_wall includes tensor preparation, forward, greedy CTC, confidence and cleanup. Both exclude video decode, MediaPipe, transport, browser and sentence-end waiting.",
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
