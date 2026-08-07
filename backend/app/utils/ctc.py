from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence


PAD_ALIASES = {"<pad>", "[pad]", "pad"}


def collapse_ctc_token_ids(token_ids: Iterable[int], blank_idx: int) -> List[int]:
    output: List[int] = []
    previous = None
    for token in token_ids:
        token = int(token)
        if token != previous:
            if token != int(blank_idx):
                output.append(token)
            previous = token
    return output


def greedy_ctc_decode_log_probs(log_probs, valid_length: int, blank_idx: int) -> List[int]:
    ids = log_probs[: int(valid_length)].argmax(-1).detach().cpu().tolist()
    return collapse_ctc_token_ids(ids, blank_idx)


def token_ids_to_sentence(token_ids: Sequence[int], id2word: Mapping[int, str]) -> str:
    words: List[str] = []
    for idx in token_ids:
        word = id2word.get(int(idx), f"<ID:{idx}>")
        if word.strip().casefold() in PAD_ALIASES:
            continue
        words.append(word)
    return " ".join(words).strip()

