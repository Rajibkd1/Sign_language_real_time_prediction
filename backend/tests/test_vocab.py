from __future__ import annotations

from pathlib import Path

from app.utils.vocab import load_vocab, validate_special_tokens


def test_vocab_parses_and_resolves_special_tokens():
    vocab_path = Path(__file__).resolve().parents[1] / "models" / "bangla_vocab_tsv_fixed.json"
    word2id, id2word, vocab_size = load_vocab(vocab_path)
    special = validate_special_tokens(word2id)

    assert vocab_size == 39
    assert special["blank"] == 0
    assert special["pad"] == 1
    assert special["unk"] == 2
    assert id2word[0] == "<BLANK>"

