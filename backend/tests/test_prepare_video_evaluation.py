import json
from pathlib import Path

import pytest

from scripts.prepare_video_evaluation import prepare


def dataset_fixture(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "splits").mkdir(parents=True)
    for split, group in (("train", "1"), ("val", "2"), ("test", "3")):
        folder = dataset / "Sign_Videos" / group
        folder.mkdir(parents=True)
        (folder / "z.mp4").write_bytes(b"fake z")
        (folder / "a.mp4").write_bytes(b"fake a")
        (dataset / "splits" / f"{split}.tsv").write_text(
            f"id\ttranslation\n{group}_00\tআমি খাই।\n", encoding="utf-8")
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"word2id": {"আমি": 1, "খাই": 2}}), encoding="utf-8")
    return dataset, vocab


def test_prepare_uses_original_sorted_recording_order_and_preserves_reference(tmp_path):
    dataset, vocab = dataset_fixture(tmp_path)
    out = tmp_path / "private"
    audit = prepare(dataset, vocab, out)
    row = json.loads((out / "test.jsonl").read_text(encoding="utf-8"))
    assert Path(row["video"]).name == "a.mp4"
    assert row["reference"] == "আমি খাই।"
    assert audit["sentence_groups_disjoint"] is True
    assert audit["accuracy_measured"] is False


def test_prepare_rejects_group_leakage_before_creating_output(tmp_path):
    dataset, vocab = dataset_fixture(tmp_path)
    (dataset / "splits" / "test.tsv").write_text("id\ttranslation\n1_01\tআমি খাই।\n", encoding="utf-8")
    out = tmp_path / "private"
    with pytest.raises(ValueError, match="overlap"):
        prepare(dataset, vocab, out)
    assert not out.exists()
