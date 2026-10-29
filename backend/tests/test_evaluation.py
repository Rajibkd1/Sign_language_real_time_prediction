from __future__ import annotations

import math

import pytest

from app.utils.evaluation import latency_summary, normalize_bdsl_v1, normalize_text, score_records, word_edits


def test_substitution_deletion_and_insertion():
    assert word_edits("আমি ভাত খাই", "আমি চা খাই").substitutions == 1
    assert word_edits("আমি ভাত খাই", "আমি খাই").deletions == 1
    assert word_edits("আমি খাই", "আমি ভাত খাই").insertions == 1


def test_corpus_wer_uses_total_words_and_counts_failed_decodes():
    result = score_records([
        {"reference": "আমি ভাত খাই", "prediction": "আমি ভাত খাই"},
        {"reference": "খাও", "prediction": "", "status": "error"},
    ])
    assert result["reference_words"] == 4
    assert result["deletions"] == 1
    assert result["wer_percent"] == 25.0
    assert result["exact_sentence_accuracy_percent"] == 50.0


def test_unicode_and_whitespace_normalization_retains_punctuation():
    assert normalize_text("  আমি\u00a0ভাত\nখাই  ") == "আমি ভাত খাই"
    assert word_edits("\u09dc", "\u09a1\u09bc").total == 0
    assert word_edits("খাই", "খাই।").total == 1


def test_insertions_can_make_wer_exceed_100_percent():
    result = score_records([{"reference": "আমি", "prediction": "সে ভাত খাবে"}])
    assert result["wer_percent"] == 300.0


def test_training_normalization_is_explicit_and_does_not_change_default_scoring():
    rows = [{"reference": "আমি খাই।", "prediction": "আমি খাই"}]
    assert score_records(rows)["wer_percent"] == 50.0
    assert score_records(rows, normalization="bdsl-v1", vocabulary={"আমি", "খাই"})["wer_percent"] == 0.0
    assert rows[0]["reference"] == "আমি খাই।"
    with pytest.raises(ValueError, match="vocabulary"):
        score_records(rows, normalization="bdsl-v1")


def test_training_normalization_preserves_known_compounds_and_removes_zero_width():
    vocab = {"চা-কফি", "চা", "কফি", "খাই"}
    assert normalize_bdsl_v1("চা—কফি খা\u200dই।", vocab) == "চা-কফি খাই"
    assert normalize_bdsl_v1("চা-কফি", {"চা", "কফি"}) == "চা কফি"


def test_training_vocabulary_comparison_normalizes_bangla_ye():
    assert normalize_bdsl_v1("খেয়ো-খায়নি", {"খেয়ো", "খায়নি"}) == "খেয়ো খায়নি"


@pytest.mark.parametrize("rows", [[], [{"reference": " \n", "prediction": "খাই"}]])
def test_empty_corpora_and_references_are_rejected(rows):
    with pytest.raises(ValueError):
        score_records(rows)


def test_latency_nearest_rank_and_invalid_values():
    result = latency_summary(list(range(1, 21)))
    assert result["p50_ms"] == 10
    assert result["p95_ms"] == 19
    assert result["mean_ms"] == 10.5
    for values in ([], [-1.0], [math.nan], [math.inf]):
        with pytest.raises(ValueError):
            latency_summary(values)
