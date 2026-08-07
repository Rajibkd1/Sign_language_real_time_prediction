from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple


def load_vocab(path: Path) -> Tuple[Dict[str, int], Dict[int, str], int]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "word2id" not in data or "id2word" not in data:
        raise KeyError("Vocabulary JSON must contain 'word2id' and 'id2word'.")

    word2id = {str(word): int(idx) for word, idx in data["word2id"].items()}
    id2word = {int(idx): str(word) for idx, word in data["id2word"].items()}
    max_idx = max(word2id.values(), default=-1)
    vocab_size = int(data.get("vocab_size", max_idx + 1))

    if vocab_size != max_idx + 1:
        raise ValueError(
            f"Vocabulary IDs must be contiguous 0..{max_idx}; vocab_size={vocab_size}."
        )

    expected = set(range(vocab_size))
    actual = set(word2id.values())
    if actual != expected:
        missing = sorted(expected - actual)
        raise ValueError(f"Vocabulary token IDs are not contiguous. Missing IDs: {missing[:10]}")

    for idx in expected:
        if idx not in id2word:
            raise ValueError(f"id2word is missing token id {idx}.")

    return word2id, id2word, vocab_size


def resolve_special_token(word2id: Mapping[str, int], aliases: Iterable[str], name: str) -> int:
    normalized = {str(k).strip().casefold(): int(v) for k, v in word2id.items()}
    for alias in aliases:
        key = alias.strip().casefold()
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Could not resolve {name}. Tried aliases: {list(aliases)}")


def validate_special_tokens(word2id: Mapping[str, int]) -> dict[str, int]:
    return {
        "blank": resolve_special_token(word2id, ["<BLANK>", "<blank>", "[BLANK]", "blank"], "BLANK token"),
        "pad": resolve_special_token(word2id, ["<PAD>", "<pad>", "[PAD]", "pad"], "PAD token"),
        "unk": resolve_special_token(word2id, ["<UNK>", "<unk>", "[UNK]", "unk"], "UNK token"),
    }

