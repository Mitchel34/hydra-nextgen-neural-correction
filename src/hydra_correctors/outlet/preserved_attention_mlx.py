"""Narrow unchanged backbone extraction for corrected-error-experiments-v1.

Source and symbol hashes: provenance/CORRECTED_ERROR_MODEL_REUSE.json.
Mamba-2/SSD adaptation: Copyright (c) 2024 Tri Dao, Albert Gu (Apache-2.0);
Copyright (c) 2025 Apple Inc. (MIT). Licenses are preserved in licenses/.
Only the named backbone definitions were copied; legacy datasets, trained
weights, scalers, queues, old heads and evaluation code are not imported.
"""
from __future__ import annotations

import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn


def historical_age_positions(history_hours, dimensions=128):
    """Fixed sinusoidal positions at ages -L,...,-1; no learned length table."""
    ages = np.arange(-history_hours, 0, dtype=np.float64)[:, None]
    frequency = np.exp(-math.log(10000.0) * np.arange(0, dimensions, 2, dtype=np.float64) / dimensions)
    phase = ages * frequency[None, :]
    position = np.empty((history_hours, dimensions), dtype=np.float32)
    position[:, 0::2] = np.sin(phase)
    position[:, 1::2] = np.cos(phase)
    return position


class FullHistoricalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3*config.d_model, bias=True)
        self.out = nn.Linear(config.d_model, config.d_model, bias=True)

    def __call__(self, x):
        c = self.config
        batch, length, _ = x.shape
        head_dim = c.d_model // c.num_heads
        q, k, v = [part.reshape(batch, length, c.num_heads, head_dim).transpose(0, 2, 1, 3)
                   for part in mx.split(self.qkv(x), 3, axis=-1)]
        if c.attention_backend == 'fast':
            # MLX 0.32.2 full-attention fused kernels require head width >=64.
            # Zero extension preserves QK^T and V exactly. Keep the ORIGINAL
            # 32-dimensional scaling, then discard the zero output coordinates.
            # This is a kernel adapter, not a change to the four-head model.
            if c.force_fused_attention and head_dim < 64:
                pads = ((0, 0), (0, 0), (0, 0), (0, 64-head_dim))
                q, k, v = [mx.pad(part, pads) for part in (q, k, v)]
            attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5,
                                                           mask=None, force_fused=c.force_fused_attention)
            attended = attended[:, :, :, :head_dim]
        else:
            scores = (q @ k.transpose(0, 1, 3, 2)) * (head_dim**-0.5)
            attended = mx.softmax(scores, axis=-1) @ v
        combined = attended.transpose(0, 2, 1, 3).reshape(batch, length, c.d_model)
        return self.out(combined)


class PreNormBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, f, p = config.d_model, config.dim_feedforward, config.dropout
        self.norm1 = nn.LayerNorm(d, eps=config.layer_norm_eps, affine=True, bias=True)
        self.attention = FullHistoricalAttention(config)
        self.norm2 = nn.LayerNorm(d, eps=config.layer_norm_eps, affine=True, bias=True)
        self.linear1 = nn.Linear(d, f, bias=True)
        self.linear2 = nn.Linear(f, d, bias=True)
        self.attention_residual_dropout = nn.Dropout(p)
        self.ff_inner_dropout = nn.Dropout(p)
        self.ff_residual_dropout = nn.Dropout(p)

    def __call__(self, x):
        x = x + self.attention_residual_dropout(self.attention(self.norm1(x)))
        hidden = self.ff_inner_dropout(nn.gelu(self.linear1(self.norm2(x))))  # exact erf GELU
        return x + self.ff_residual_dropout(self.linear2(hidden))
