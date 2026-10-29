from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.utils.evaluation import score_records
from scripts.evaluate_videos import evaluate_samples, read_manifest


def test_runner_keeps_failed_video_and_continues_without_history(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video for mocked service")
    calls = []

    class Response:
        prediction = "আমি খাই"
        timings = SimpleNamespace(model_ms=1.0)

        def model_dump(self, **kwargs):
            return {"prediction": self.prediction}

    class Service:
        def predict_video(self, path, *, save_history):
            calls.append(save_history)
            if len(calls) == 1:
                raise RuntimeError("example decode failure")
            return Response()

    rows = [{"id": str(i), "video": clip, "reference": "আমি খাই", "group_id": "g"} for i in range(2)]
    samples, walls, forwards = evaluate_samples(rows, Service())
    assert [row["status"] for row in samples] == ["error", "ok"]
    assert samples[0]["prediction"] == ""
    assert samples[0]["error_type"] == "RuntimeError"
    assert len(walls) == 2
    assert forwards == [1.0]
    assert calls == [False, False]
    assert score_records(samples)["wer_percent"] == 50.0


def test_manifest_resolves_relative_video_and_rejects_duplicate_ids(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"mock")
    row = {"id": "a", "video": "clip.mp4", "reference": "আমি", "group_id": "g"}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert read_manifest(manifest)[0]["video"] == clip.resolve()
    manifest.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest)


def test_manifest_rejects_missing_media_and_empty_references(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"mock")
    manifest = tmp_path / "manifest.jsonl"
    for video, reference in [("missing.mp4", "আমি"), ("clip.mp4", " ")]:
        manifest.write_text(json.dumps({"id": "a", "video": video, "reference": reference,
                                        "group_id": "g"}), encoding="utf-8")
        with pytest.raises(ValueError):
            read_manifest(manifest)


def test_exclusion_manifest_checks_groups_without_training_video_access(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"id": "a", "group_id": "g", "reference": "আমি"}), encoding="utf-8")
    assert read_manifest(manifest, require_videos=False)[0]["group_id"] == "g"
    with pytest.raises(ValueError, match="video"):
        read_manifest(manifest)
