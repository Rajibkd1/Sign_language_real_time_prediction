from __future__ import annotations

from app.utils.ctc import collapse_ctc_token_ids, token_ids_to_sentence


def test_ctc_collapse_removes_blanks_and_duplicates():
    ids = [30, 30, 0, 36, 36, 0, 5, 0, 14, 14]
    assert collapse_ctc_token_ids(ids, blank_idx=0) == [30, 36, 5, 14]


def test_token_ids_to_sentence_uses_exact_mapping():
    id2word = {0: "<BLANK>", 1: "<PAD>", 5: "কফি", 14: "খায়নি", 30: "বাবা", 36: "সকালে"}
    assert token_ids_to_sentence([30, 36, 5, 14], id2word) == "বাবা সকালে কফি খায়নি"

