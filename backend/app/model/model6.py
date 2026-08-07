from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_DIM = 1662
MODALITY_DIMS = (132, 1404, 63, 63)
REGION_EMBED_DIMS = (64, 64, 96, 96)
D_MODEL = sum(REGION_EMBED_DIMS)
STREAM_GATE_HIDDEN = 160
TEMPORAL_KERNEL_SIZE = 7
TEMPORAL_SUBSAMPLE_STRIDE = 2
TRANSFORMER_LAYERS = 4
NHEAD = 8
FFN_DIM = 960
DROPOUT = 0.15
LAYER_SCALE_INIT = 0.10
POSITIONAL_MAX_LEN = 512
EXPECTED_PARAM_COUNT_FOR_VOCAB39 = 4_924_163


class ModalityAwareNormalizer(nn.Module):
    def __init__(self, dims=MODALITY_DIMS):
        super().__init__()
        self.dims = tuple(int(x) for x in dims)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in self.dims])

    def forward(self, x):
        chunks = torch.split(x, self.dims, dim=-1)
        return torch.cat([norm(chunk) for norm, chunk in zip(self.norms, chunks)], dim=-1)


class CompactRegionProjector(nn.Module):
    def __init__(self, in_dims=MODALITY_DIMS, out_dims=REGION_EMBED_DIMS, dropout=DROPOUT):
        super().__init__()
        self.in_dims = tuple(int(x) for x in in_dims)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                    nn.LayerNorm(out_dim),
                    nn.Dropout(dropout),
                )
                for in_dim, out_dim in zip(in_dims, out_dims)
            ]
        )

    def forward(self, x):
        chunks = torch.split(x, self.in_dims, dim=-1)
        return torch.cat([block(chunk) for block, chunk in zip(self.blocks, chunks)], dim=-1)


class RawVelocityGate(nn.Module):
    def __init__(self, d_model=D_MODEL, hidden=STREAM_GATE_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        nn.init.constant_(self.gate[-1].bias, 1.0)

    def forward(self, raw, velocity):
        raw_weight = torch.sigmoid(self.gate(torch.cat([raw, velocity], dim=-1)))
        fused = raw_weight * raw + (1.0 - raw_weight) * velocity
        return fused, raw_weight


class StridedDepthwiseTemporalBlock(nn.Module):
    def __init__(
        self,
        d_model=D_MODEL,
        kernel_size=TEMPORAL_KERNEL_SIZE,
        stride=TEMPORAL_SUBSAMPLE_STRIDE,
        dropout=DROPOUT,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=d_model,
            bias=False,
        )
        self.pointwise_glu = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = F.glu(self.pointwise_glu(y), dim=1).transpose(1, 2)
        return self.dropout(self.output_norm(y))


class ResidualDepthwiseTemporalBlock(nn.Module):
    def __init__(
        self,
        d_model=D_MODEL,
        kernel_size=TEMPORAL_KERNEL_SIZE,
        dropout=DROPOUT,
        layer_scale=LAYER_SCALE_INIT,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=padding,
            groups=d_model,
            bias=False,
        )
        self.pointwise_glu = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((d_model,), float(layer_scale)))

    def forward(self, x, padding_mask=None):
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = F.glu(self.pointwise_glu(y), dim=1).transpose(1, 2)
        x = x + self.dropout(y) * self.layer_scale
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, max_len=POSITIONAL_MAX_LEN, dropout=DROPOUT):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(1, max_len, d_model, dtype=torch.float32)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.size(1) > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional limit {self.encoding.size(1)}"
            )
        return self.dropout(x + self.encoding[:, : x.size(1)].to(device=x.device, dtype=x.dtype))


class Model6(nn.Module):
    def __init__(self, vocab_size: int, blank_idx: int):
        super().__init__()
        self.normalizer = ModalityAwareNormalizer()
        self.raw_projector = CompactRegionProjector()
        self.velocity_projector = CompactRegionProjector()
        self.stream_gate = RawVelocityGate()
        self.temporal_subsampler = StridedDepthwiseTemporalBlock()
        self.temporal_refiner = ResidualDepthwiseTemporalBlock()
        self.positional_encoding = SinusoidalPositionalEncoding()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TRANSFORMER_LAYERS,
            norm=nn.LayerNorm(D_MODEL),
        )
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.classifier = nn.Linear(D_MODEL, int(vocab_size))

        nn.init.constant_(self.classifier.bias, 0.0)
        with torch.no_grad():
            self.classifier.bias[int(blank_idx)] = -0.25

    @staticmethod
    def output_lengths(input_lengths):
        return torch.div(input_lengths - 1, TEMPORAL_SUBSAMPLE_STRIDE, rounding_mode="floor") + 1

    def forward(self, x, input_lengths, padding_mask=None, return_gate=False):
        normalized = self.normalizer(x)

        velocity = torch.zeros_like(normalized)
        velocity[:, 1:] = normalized[:, 1:] - normalized[:, :-1]

        if padding_mask is not None:
            normalized = normalized.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            velocity = velocity.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        raw_embedding = self.raw_projector(normalized)
        velocity_embedding = self.velocity_projector(velocity)
        x, raw_weight = self.stream_gate(raw_embedding, velocity_embedding)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = self.temporal_subsampler(x)
        output_lengths = self.output_lengths(input_lengths)
        output_mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= output_lengths.unsqueeze(1)
        x = x.masked_fill(output_mask.unsqueeze(-1), 0.0)

        x = self.temporal_refiner(x, output_mask)
        x = self.positional_encoding(x)
        x = self.transformer(x, src_key_padding_mask=output_mask)
        x = self.output_norm(x).masked_fill(output_mask.unsqueeze(-1), 0.0)

        log_probs = F.log_softmax(self.classifier(x).float(), dim=-1)
        if return_gate:
            return log_probs, output_lengths, output_mask, raw_weight
        return log_probs, output_lengths, output_mask

