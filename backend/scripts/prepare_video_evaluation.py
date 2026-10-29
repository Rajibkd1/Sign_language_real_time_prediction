"""Audit existing sentence-group TSV splits and map IDs to the original sorted MP4 files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.utils.evaluation import normalize_bdsl_v1, normalize_text
from scripts.evaluate_videos import sha256


def prepare(dataset: Path, vocabulary_path: Path, output: Path,
            map_splits: tuple[str, ...] = ("test",)) -> dict:
    if output.exists():
        raise ValueError("Choose a new output directory to preserve earlier evidence.")
    vocabulary = {normalize_text(word) for word in
                  json.loads(vocabulary_path.read_text(encoding="utf-8"))["word2id"]}
    groups, manifests, sources, clip_cache = {}, {}, {}, {}
    for split in ("train", "val", "test"):
        path = dataset / "splits" / f"{split}.tsv"
        sources[split] = sha256(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        ids, records, split_groups = set(), [], set()
        for row in rows:
            sample_id = row["id"]
            if sample_id in ids:
                raise ValueError(f"Duplicate {split} sample: {sample_id}")
            ids.add(sample_id)
            group, index = sample_id.rsplit("_", 1)
            reference = row["translation"]
            words = normalize_bdsl_v1(reference, vocabulary).split()
            if not words or any(word not in vocabulary for word in words):
                raise ValueError(f"Empty or out-of-vocabulary reference: {sample_id}")
            record = {"id": sample_id, "group_id": group, "reference": reference}
            if split in map_splits:
                # Matches the supplied extraction script's sorted '*.mp4' enumeration.
                if group not in clip_cache:
                    clip_cache[group] = sorted((dataset / "Sign_Videos" / group).glob("*.mp4"))
                clips = clip_cache[group]
                number = int(index)
                if number < 0 or number >= len(clips):
                    raise ValueError(f"Cannot map sample to an existing video: {sample_id}")
                record["video"] = str(clips[number].resolve())
            records.append(record)
            split_groups.add(group)
        if not records:
            raise ValueError(f"Empty split: {split}")
        manifests[split], groups[split] = records, split_groups
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b]:
            raise ValueError(f"Sentence groups overlap: {a}/{b}")
    audit = {"schema_version": 1, "kind": "split_preparation_only", "accuracy_measured": False,
             "vocab_sha256": sha256(vocabulary_path), "source_tsv_sha256": sources,
             "normalization": "bdsl-v1", "sentence_groups_disjoint": True,
             "video_mapped_splits": list(map_splits),
             "training_membership_of_checkpoint_verified": False,
             "video_id_mapping": "numeric suffix indexes lexicographically sorted MP4 names in the sentence folder",
             "splits": {s: {"samples": len(manifests[s]), "groups": len(groups[s])} for s in manifests}}
    output.mkdir(parents=True)
    for split, records in manifests.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    audit["manifest_sha256"] = {s: sha256(output / f"{s}.jsonl") for s in manifests}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--map-splits", nargs="+", choices=("train", "val", "test"), default=["test"])
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.vocabulary, args.output, tuple(args.map_splits)), indent=2))


if __name__ == "__main__":
    main()
