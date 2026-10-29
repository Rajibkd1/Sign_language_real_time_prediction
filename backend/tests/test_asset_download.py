from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.download_mediapipe_models import ASSET_MANIFEST, download_assets, matches_manifest


def fixture_manifest(tmp_path, payload=b"verified model"):
    record = {"url": "https://example.invalid/model.task", "size_bytes": len(payload),
              "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"assets": {"model.task": record}}))
    return manifest, record


def test_verified_existing_asset_does_not_download(tmp_path, monkeypatch):
    manifest, _ = fixture_manifest(tmp_path)
    (tmp_path / "model.task").write_bytes(b"verified model")

    def unexpected(*args):
        raise AssertionError("A verified asset must not be downloaded again")

    monkeypatch.setattr("urllib.request.urlretrieve", unexpected)
    download_assets(tmp_path, manifest)


def test_corrupt_existing_asset_is_repaired_after_verification(tmp_path, monkeypatch):
    manifest, record = fixture_manifest(tmp_path)
    target = tmp_path / "model.task"
    target.write_bytes(b"corrupt model!")
    monkeypatch.setattr("urllib.request.urlretrieve", lambda url, path: Path(path).write_bytes(b"verified model"))
    download_assets(tmp_path, manifest)
    assert matches_manifest(target, record)


def test_bad_download_preserves_existing_asset_and_removes_partial(tmp_path, monkeypatch):
    manifest, _ = fixture_manifest(tmp_path)
    target = tmp_path / "model.task"
    target.write_bytes(b"old model")
    monkeypatch.setattr("urllib.request.urlretrieve", lambda url, path: Path(path).write_bytes(b"wrong download"))
    with pytest.raises(ValueError, match="SHA-256"):
        download_assets(tmp_path, manifest)
    assert target.read_bytes() == b"old model"
    assert not (tmp_path / "model.task.part").exists()


def test_bundled_assets_match_verified_upstream_bytes():
    records = json.loads(ASSET_MANIFEST.read_text())["assets"]
    assert all(matches_manifest(ASSET_MANIFEST.parent / name, record) for name, record in records.items())


def test_asset_manifest_cannot_escape_target_directory(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"assets": {"../outside.task": {}}}))
    with pytest.raises(ValueError, match="basenames"):
        download_assets(tmp_path, manifest)
