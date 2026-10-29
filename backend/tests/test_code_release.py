import hashlib
import json
import subprocess
import zipfile

import pytest

from scripts.build_code_release import build


def repository(tmp_path, *, exclude=True):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / "app.py").write_text("print('example')\n")
    (root / "weights.pt").write_bytes(b"model weights")
    (root / "bangla_vocab.json").write_text("{}")
    if exclude:
        (root / ".gitattributes").write_text("*.pt export-ignore\nbangla_vocab*.json export-ignore\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "--quiet", "-m", "fixture"], cwd=root, check=True)
    return root


def test_release_is_reproducible_and_excludes_research_assets(tmp_path):
    root = repository(tmp_path)
    first, second = tmp_path / "a", tmp_path / "b"
    for output in (first, second):
        build(root, output, "0.1.0-rc.1", "HEAD", "https://example.invalid/ci/1")
    name = "BdSL-Vision-0.1.0-rc.1-source.zip"
    assert (first / name).read_bytes() == (second / name).read_bytes()
    manifest = json.loads((first / "RELEASE_MANIFEST.json").read_text())
    assert manifest["new_video_evaluation_completed"] is False
    assert manifest["project_checkpoint_included"] is False
    with zipfile.ZipFile(first / name) as archive:
        assert not any(n.endswith((".pt", "bangla_vocab.json")) for n in archive.namelist())
        for path, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read("BdSL-Vision-0.1.0-rc.1/" + path)).hexdigest() == expected


def test_release_fails_if_export_rules_allow_model_assets(tmp_path):
    root = repository(tmp_path, exclude=False)
    with pytest.raises(ValueError, match="leaked"):
        build(root, tmp_path / "output", "0.1.0-rc.1", "HEAD", "https://example.invalid/ci/1")
    assert not (tmp_path / "output").exists()
