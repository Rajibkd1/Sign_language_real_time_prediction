from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from app.config import Settings
from app.errors import CheckpointFailureError, VocabMismatchError
from app.model.model6 import EXPECTED_PARAM_COUNT_FOR_VOCAB39, FEATURE_DIM, Model6
from app.utils.ctc import greedy_ctc_decode_log_probs, token_ids_to_sentence
from app.utils.vocab import load_vocab, validate_special_tokens


def torch_load_full(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def strip_common_prefixes(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict(state_dict)
    for prefix in ("module.", "_orig_mod."):
        if result and all(k.startswith(prefix) for k in result):
            result = {k[len(prefix) :]: v for k, v in result.items()}
    return result


@dataclass
class ModelPrediction:
    sentence: str
    token_ids: list[int]
    input_frames: int
    output_steps: int
    model_ms: float
    confidence_value: float


@dataclass
class ModelRuntime:
    model: Model6
    device: torch.device
    blank_idx: int
    id2word: Dict[int, str]
    vocab_size: int
    weight_source: str
    parameters: int
    metadata: Dict[str, Any]
    use_amp: bool

    @classmethod
    def load(cls, settings: Settings) -> "ModelRuntime":
        try:
            word2id, id2word, vocab_size = load_vocab(settings.vocab_path)
            special = validate_special_tokens(word2id)
            blank_idx = int(special["blank"])
            if blank_idx != 0:
                raise VocabMismatchError(
                    f"Expected BLANK index 0 for this Model 6 training run; got {blank_idx}."
                )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch_load_full(settings.checkpoint_path)
            if not isinstance(checkpoint, dict):
                raise CheckpointFailureError("Checkpoint payload is not a dictionary.")

            state_key = (
                "ema_state"
                if settings.prefer_ema and checkpoint.get("ema_state") is not None
                else "model_state"
            )
            state_dict = checkpoint.get(state_key)
            if state_dict is None:
                raise CheckpointFailureError(f"Checkpoint has no usable {state_key!r} state.")
            state_dict = strip_common_prefixes(state_dict)

            classifier_key = next((k for k in state_dict if k.endswith("classifier.weight")), None)
            if classifier_key is None:
                raise CheckpointFailureError("Could not find classifier.weight in checkpoint state dict.")

            checkpoint_vocab_size = int(state_dict[classifier_key].shape[0])
            if checkpoint_vocab_size != vocab_size:
                raise VocabMismatchError(
                    f"Vocabulary mismatch: checkpoint={checkpoint_vocab_size}, JSON={vocab_size}. "
                    "Use the exact bangla_vocab_tsv_fixed.json from this training run."
                )

            model = Model6(vocab_size, blank_idx).to(device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()

            metadata = {
                "epoch": checkpoint.get("epoch"),
                "checkpoint_kind": checkpoint.get("checkpoint_kind"),
                "model_variant": checkpoint.get("model_variant"),
            }

            del state_dict
            del checkpoint
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if vocab_size == 39 and parameters != EXPECTED_PARAM_COUNT_FOR_VOCAB39:
                raise CheckpointFailureError(
                    f"Unexpected Model 6 parameter count: {parameters:,}; "
                    f"expected {EXPECTED_PARAM_COUNT_FOR_VOCAB39:,} for vocab size 39."
                )

            runtime = cls(
                model=model,
                device=device,
                blank_idx=blank_idx,
                id2word=id2word,
                vocab_size=vocab_size,
                weight_source=state_key,
                parameters=parameters,
                metadata=metadata,
                use_amp=settings.use_amp,
            )
            runtime.warm_up()
            return runtime
        except (CheckpointFailureError, VocabMismatchError):
            raise
        except Exception as exc:
            raise CheckpointFailureError(str(exc)) from exc

    def warm_up(self) -> None:
        with torch.inference_mode():
            smoke_x = torch.zeros(1, 8, FEATURE_DIM, device=self.device)
            smoke_len = torch.tensor([8], device=self.device, dtype=torch.long)
            smoke_mask = torch.zeros(1, 8, device=self.device, dtype=torch.bool)
            self.model(smoke_x, smoke_len, smoke_mask)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def predict(self, features: np.ndarray, min_frames: int) -> ModelPrediction:
        features = np.ascontiguousarray(features, dtype=np.float32)
        if len(features) < min_frames:
            raise ValueError(f"Too few model frames ({len(features)}).")

        x = torch.from_numpy(features).unsqueeze(0).to(self.device, non_blocking=True)
        lengths = torch.tensor([x.shape[1]], device=self.device, dtype=torch.long)
        padding_mask = torch.zeros((1, x.shape[1]), device=self.device, dtype=torch.bool)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda" and self.use_amp
            else contextlib.nullcontext()
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), amp_context:
            log_probs, output_lengths, _ = self.model(x, lengths, padding_mask)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        model_ms = (time.perf_counter() - started) * 1000.0

        valid_len = int(output_lengths[0].item())
        token_ids = greedy_ctc_decode_log_probs(log_probs[0], valid_len, self.blank_idx)
        sentence = token_ids_to_sentence(token_ids, self.id2word)

        if valid_len > 0:
            max_probs = log_probs[0, :valid_len].detach().float().exp().max(dim=-1).values
            confidence_value = float(max_probs.mean().clamp(0.0, 1.0).item())
        else:
            confidence_value = 0.0

        del x, lengths, padding_mask, log_probs, output_lengths
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return ModelPrediction(
            sentence=sentence,
            token_ids=token_ids,
            input_frames=int(len(features)),
            output_steps=valid_len,
            model_ms=float(model_ms),
            confidence_value=confidence_value,
        )

    def info(self, max_sequence_length: int, target_fps: float) -> dict[str, Any]:
        return {
            "model": "Model 6",
            "architecture": "Lightweight Region-Motion Depthwise Conv-Transformer + Word-Level CTC",
            "parameters": self.parameters,
            "vocab_size": self.vocab_size,
            "blank_index": self.blank_idx,
            "weight_source": self.weight_source,
            "device": str(self.device),
            "max_sequence_length": max_sequence_length,
            "target_fps": target_fps,
            "ready": True,
        }

