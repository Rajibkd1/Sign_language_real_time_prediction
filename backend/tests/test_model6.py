from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.model.model6 import EXPECTED_PARAM_COUNT_FOR_VOCAB39, FEATURE_DIM, Model6


def test_model6_parameter_count_for_vocab39():
    model = Model6(vocab_size=39, blank_idx=0)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == EXPECTED_PARAM_COUNT_FOR_VOCAB39


def test_model6_output_temporal_length_for_121_frames():
    model = Model6(vocab_size=39, blank_idx=0)
    model.eval()
    x = torch.zeros(1, 121, FEATURE_DIM)
    lengths = torch.tensor([121], dtype=torch.long)
    mask = torch.zeros(1, 121, dtype=torch.bool)
    with torch.inference_mode():
        y, out_lengths, _ = model(x, lengths, mask)
    assert tuple(y.shape) == (1, 61, 39)
    assert int(out_lengths[0]) == 61

