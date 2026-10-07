"""Narrow unchanged backbone extraction for corrected-error-experiments-v1.

Source and symbol hashes: provenance/CORRECTED_ERROR_MODEL_REUSE.json.
Mamba-2/SSD adaptation: Copyright (c) 2024 Tri Dao, Albert Gu (Apache-2.0);
Copyright (c) 2025 Apple Inc. (MIT). Licenses are preserved in licenses/.
Only the named backbone definitions were copied; legacy datasets, trained
weights, scalers, queues, old heads and evaluation code are not imported.
"""
from __future__ import annotations

import math
import mlx.core as mx
import mlx.nn as nn


def stable_segment_sum(x):
    """sum(x[j+1:i+1]), without subtracting large cumulative sums.

    The upper triangle has no mathematical meaning and is masked after exp.
    All entries there remain finite, avoiding invalid exp/gradient products.
    """
    length = x.shape[-1]
    return mx.cumsum(mx.tril(mx.broadcast_to(x[..., None], x.shape + (length,)), -1), axis=-2)


def ssd_full_sequence(x, A_log, B, C, D, dt, dt_bias, chunk_size=64):
    """Mamba-2 SSD with zero initial state and differentiable chunk carry.

    x[B,L,H,P], B/C[B,L,G,N], dt[B,L,H]; G=1 in the frozen models.
    State[P,N] obeys S_t=exp(dt_t*A)*S_(t-1)+dt_t*x_t*B_t.
    Each final partial chunk is handled directly. No state escapes the call.
    """
    batch, length, heads, head_dim = x.shape
    groups, state_dim = B.shape[-2:]
    if heads % groups:
        raise ValueError('SSM heads must be divisible by groups')
    repeats = heads // groups
    # Dynamics stay float32, including the stable decay calculation and carry.
    delta = nn.softplus(dt.astype(mx.float32) + dt_bias)
    delta_A = delta * (-mx.exp(A_log.astype(mx.float32)))[None, None, :]
    delta_x = delta[..., None] * x
    state = None
    outputs = []
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        dx = delta_x[:, start:stop]
        da = delta_A[:, start:stop]
        b = B[:, start:stop].transpose(0, 2, 3, 1)
        c = C[:, start:stop]
        cb = mx.repeat(c.swapaxes(1, 2) @ b, repeats, axis=1)
        decay = mx.exp(stable_segment_sum(da.swapaxes(1, 2)))
        intra = mx.tril(cb * decay) @ dx.swapaxes(1, 2)
        y = intra.swapaxes(1, 2)
        final_decay = decay[:, :, -1:, :].transpose(0, 3, 1, 2)
        b_heads = mx.repeat(b, repeats, axis=1).swapaxes(2, 3)
        next_state = (dx * final_decay).transpose(0, 2, 3, 1) @ b_heads
        if state is not None:
            carry_decay = mx.exp(mx.cumsum(da, axis=1))
            c_heads = mx.repeat(c, repeats, axis=2)
            # [B,H,L,N]@[B,H,N,P], avoiding materialized[B,L,H,P,N].
            prior = (c_heads.swapaxes(1, 2) @ state.swapaxes(-1, -2)).swapaxes(1, 2)
            y = y + carry_decay[..., None] * prior
            next_state = next_state + carry_decay[:, -1, :, None, None] * state
        outputs.append(y)
        state = next_state
    return mx.concatenate(outputs, axis=1) + x * D[None, None, :, None]


class Mamba2MLX(nn.Module):
    """Canonical compact Mamba-2: selective scalar-head SSD plus gated RMSNorm."""
    def __init__(self, d_state, chunk_size=64):
        super().__init__()
        self.d_state = int(d_state)
        self.chunk_size = int(chunk_size)
        self.d_model = self.d_inner = 128
        self.headdim, self.nheads, self.ngroups, self.d_conv = 64, 2, 1, 4
        conv_dim = 128 + 2 * self.d_state
        self.in_proj = nn.Linear(128, 256 + 2*self.d_state + 2, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, 4, groups=conv_dim, bias=True)
        self.out_proj = nn.Linear(128, 128, bias=False)
        # Match reference nn.Linear/Conv1d from-scratch uniform initialization.
        self.in_proj.weight = mx.random.uniform(-128**-0.5, 128**-0.5, self.in_proj.weight.shape)
        self.out_proj.weight = mx.random.uniform(-128**-0.5, 128**-0.5, self.out_proj.weight.shape)
        self.conv1d.weight = mx.random.uniform(-0.5, 0.5, self.conv1d.weight.shape)
        self.conv1d.bias = mx.random.uniform(-0.5, 0.5, self.conv1d.bias.shape)
        delta = mx.exp(mx.random.uniform(math.log(0.001), math.log(0.1), (2,)))
        delta = mx.maximum(delta, 1e-4)
        self.dt_bias = delta + mx.log(-mx.expm1(-delta))
        self.A_log = mx.log(mx.random.uniform(1.0, 16.0, (2,)))
        self.D = mx.ones((2,), dtype=mx.float32)
        self.norm = nn.RMSNorm(128, eps=1e-5)

    def __call__(self, hidden):
        batch, length, _ = hidden.shape
        projected = self.in_proj(hidden)
        z, conv, dt = mx.split(projected, [128, 256 + 2*self.d_state], axis=-1)
        # Left padding only: output at t consumes projected inputs[t-3:t+1].
        conv = nn.silu(self.conv1d(mx.pad(conv, ((0, 0), (3, 0), (0, 0)))))
        x, B, C = mx.split(conv, [128, 128+self.d_state], axis=-1)
        y = ssd_full_sequence(x.reshape(batch, length, 2, 64), self.A_log,
                              B.reshape(batch, length, 1, self.d_state),
                              C.reshape(batch, length, 1, self.d_state),
                              self.D, dt, self.dt_bias, self.chunk_size)
        # Reference norm_before_gate=False: SiLU gate precedes RMS normalization.
        return self.out_proj(self.norm(y.reshape(batch, length, 128) * nn.silu(z)))


class ResidualMemoryStage(nn.Module):
    def __init__(self, kind, d_state, chunk_size):
        super().__init__()
        self.norm = nn.RMSNorm(128, eps=1e-5)
        self.mixer = nn.GRU(128, 128, bias=True) if kind == 'gru' else Mamba2MLX(d_state, chunk_size)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x):
        # Native GRU hidden=None and SSD internal state=None both mean zero reset.
        return x + self.dropout(self.mixer(self.norm(x)))
