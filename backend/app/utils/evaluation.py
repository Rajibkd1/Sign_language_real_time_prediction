"""Dependency-free corpus scoring for NFC-normalized, whitespace-tokenized text."""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Reference and prediction must be strings.")
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_bdsl_v1(text: str, vocabulary: set[str]) -> str:
    """Explicit training-notebook convention; preserve lexical hyphens in known words."""
    value = normalize_text(text)
    vocabulary = {normalize_text(word) for word in vocabulary}
    zero_width = {"\u200b", "\u200c", "\u200d", "\ufeff"}
    dashes = {"-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"}
    cleaned = "".join("-" if ch in dashes else " " if unicodedata.category(ch).startswith("P")
                      else ch for ch in value if ch not in zero_width)
    words = []
    for token in cleaned.split():
        parts = [part for part in token.split("-") if part]
        if token not in vocabulary and "-" in token and parts and all(p in vocabulary for p in parts):
            words.extend(parts)
        else:
            words.append(token)
    return " ".join(words)


@dataclass(frozen=True)
class WordEdits:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def word_edits(reference: str, prediction: str) -> WordEdits:
    """Levenshtein alignment, preferring substitution, deletion, then insertion on ties."""
    ref, hyp = normalize_text(reference).split(), normalize_text(prediction).split()
    previous = [WordEdits(insertions=i) for i in range(len(hyp) + 1)]
    for i, ref_word in enumerate(ref, 1):
        current = [WordEdits(deletions=i)]
        for j, hyp_word in enumerate(hyp, 1):
            diagonal, above, left = previous[j - 1], previous[j], current[j - 1]
            if ref_word == hyp_word:
                current.append(diagonal)
            else:
                candidates = [
                    WordEdits(diagonal.substitutions + 1, diagonal.deletions, diagonal.insertions),
                    WordEdits(above.substitutions, above.deletions + 1, above.insertions),
                    WordEdits(left.substitutions, left.deletions, left.insertions + 1),
                ]
                current.append(min(candidates, key=lambda x: x.total))
        previous = current
    return previous[-1]


def score_records(records: list[dict], *, normalization: str = "nfc-whitespace",
                  vocabulary: set[str] | None = None) -> dict:
    """Corpus WER (not mean sample WER); failed decodes must have an empty prediction."""
    if not records:
        raise ValueError("Evaluation requires at least one sample.")
    if normalization == "bdsl-v1":
        if not vocabulary:
            raise ValueError("bdsl-v1 normalization requires the evaluated model vocabulary.")
        normalizer = lambda value: normalize_bdsl_v1(value, vocabulary)
    elif normalization == "nfc-whitespace":
        normalizer = normalize_text
    else:
        raise ValueError("Unknown normalization protocol.")
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0}
    words = exact = 0
    for record in records:
        reference = normalizer(record["reference"])
        prediction = normalizer(record["prediction"])
        if not reference:
            raise ValueError("Each evaluation reference must contain at least one word.")
        edits = word_edits(reference, prediction)
        for key in totals:
            totals[key] += getattr(edits, key)
        words += len(reference.split())
        exact += reference == prediction
    return {
        "samples": len(records), "reference_words": words, **totals,
        "word_errors": sum(totals.values()),
        "wer_percent": 100.0 * sum(totals.values()) / words,
        "exact_sentences": exact,
        "exact_sentence_accuracy_percent": 100.0 * exact / len(records),
        "normalization": normalization,
    }


def latency_summary(values: list[float]) -> dict:
    """Nearest-rank percentiles; no network/browser time is implied by this helper."""
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Latency samples must be finite, nonnegative and nonempty.")
    ordered = sorted(values)
    return {
        "count": len(values), "mean_ms": sum(values) / len(values),
        "p50_ms": ordered[math.ceil(len(values) * 0.50) - 1],
        "p95_ms": ordered[math.ceil(len(values) * 0.95) - 1],
        "min_ms": ordered[0], "max_ms": ordered[-1],
    }
