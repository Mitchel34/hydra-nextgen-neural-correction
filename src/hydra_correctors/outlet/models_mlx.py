"""Controlled scratch-trained GRU/MLX Mamba-2 residual adapters.

Information arms, widths and FiLM placement implement PLAN.md sections 2/6.
The data boundary is float32 NumPy or MLX arrays. This module never selects an
accelerator, launches fitting, reads data, or imports historical review code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
from typing import Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .preserved_attention_mlx import PreNormBlock, historical_age_positions
from .preserved_memory_mlx import ResidualMemoryStage


ARCHITECTURES = ("gru_transformer", "mamba2_transformer_mlx")
VARIANTS = {
    "V0": {"errors": False, "forecast": "own_lead", "early": False},
    "V1": {"errors": True, "forecast": "own_lead", "early": False},
    "V2": {"errors": False, "forecast": "full_18", "early": False},
    "V3": {"errors": True, "forecast": "full_18", "early": False},
    "V4": {"errors": True, "forecast": "full_18", "early": True},
}


@dataclass(frozen=True)
class CorrectedModelConfig:
    history_hours: int = 168
    input_features: int = 10
    num_sites: int = 3
    forecast_leads: int = 18
    error_features: int = 7
    d_model: int = 128
    memory_layers: int = 1
    historical_transformer_layers: int = 1
    num_heads: int = 4
    dim_feedforward: int = 128
    forecast_projection_width: int = 32
    error_projection_width: int = 32
    site_embedding_size: int = 8
    lead_embedding_size: int = 8
    head_hidden_size: int = 64
    mamba_state_dim: int = 192
    mamba_expansion: int = 1
    mamba_heads: int = 2
    mamba_head_dim: int = 64
    mamba_conv_width: int = 4
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    # Equivalent math/backend choices, not model-search hyperparameters.
    attention_backend: str = "fast"
    force_fused_attention: bool = False
    ssd_chunk_size: int = 64

    def __post_init__(self):
        runtime_choices = {"attention_backend", "force_fused_attention", "ssd_chunk_size"}
        for field in fields(self):
            if field.name not in runtime_choices and getattr(self, field.name) != field.default:
                raise ValueError(f"Frozen corrected architecture requires {field.name}={field.default}")
        if self.attention_backend not in ("fast", "dense"):
            raise ValueError("attention_backend must be fast or dense")
        if self.force_fused_attention is not False:
            raise ValueError("Forced fused attention is not enabled in this adapter")
        if self.ssd_chunk_size not in (64, 128):
            raise ValueError("Only equivalent SSD chunk sizes 64/128 are supported")

    def to_dict(self):
        return asdict(self)


def _stream_seed(seed: int, component: str) -> int:
    """Stable seeds do not depend on Python hash randomization or branch order."""
    payload = f"corrected-error-experiments-v1:init:{seed}:{component}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


class IdentityFiLM(nn.Module):
    """Exactly identity at initialization; both scale and shift are trainable."""
    def __init__(self, projection_width=32, d_model=128):
        super().__init__()
        self.affine = nn.Linear(projection_width, 2 * d_model, bias=True)
        self.affine.weight = mx.zeros_like(self.affine.weight)
        self.affine.bias = mx.zeros_like(self.affine.bias)

    def __call__(self, historical, forecast_representation):
        delta_scale, shift = mx.split(self.affine(forecast_representation), 2, axis=-1)
        return historical * (1.0 + delta_scale[:, None, :]) + shift[:, None, :]


class CorrectedResidualMLX(nn.Module):
    """Return normalized residual[N,18]; all recurrent state is call-local.

    V0/V1 evaluate the same 36->32 forecast projection separately for each
    queried lead with only that lead visible. V2-V4 broadcast one full-trajectory
    representation. All variants own the same error branch; V0/V2 pass only
    constant zeros to it, including all error masks. Thus the optional FiLM is
    the only parameter-count difference within an architecture.

    Component initialization resets the global MLX RNG to independent named
    seeds, then to a common dropout seed. The trainer should explicitly seed
    or restore its RNG after construction for a resumable training run.
    """
    def __init__(self, architecture, variant, seed, config=None, *, _initialization_order=None):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {tuple(VARIANTS)}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if config is None:
            config = CorrectedModelConfig()
        elif isinstance(config, Mapping):
            config = CorrectedModelConfig(**dict(config))
        if not isinstance(config, CorrectedModelConfig):
            raise TypeError("config must be CorrectedModelConfig, a mapping, or None")
        self.config, self.architecture, self.variant, self.seed = config, architecture, variant, seed
        self._arm = dict(VARIANTS[variant])
        c = config
        kind = "gru" if architecture == "gru_transformer" else "mamba2"
        head_width = (c.d_model + c.site_embedding_size + c.lead_embedding_size
                      + c.forecast_projection_width + c.error_projection_width)
        factories = {
            "input_projection": lambda: nn.Linear(c.input_features, c.d_model),
            "memory": lambda: ResidualMemoryStage(kind, c.mamba_state_dim, c.ssd_chunk_size),
            "historical_attention": lambda: PreNormBlock(c),
            "final_norm": lambda: nn.LayerNorm(c.d_model, eps=c.layer_norm_eps, affine=True, bias=True),
            "site_embedding": lambda: nn.Embedding(c.num_sites, c.site_embedding_size),
            "lead_embedding": lambda: nn.Embedding(c.forecast_leads, c.lead_embedding_size),
            "forecast_projection": lambda: nn.Linear(2*c.forecast_leads, c.forecast_projection_width),
            "error_projection": lambda: nn.Linear(c.forecast_leads*c.error_features, c.error_projection_width),
            "head_hidden": lambda: nn.Linear(head_width, c.head_hidden_size),
            "head_out": lambda: nn.Linear(c.head_hidden_size, 1),
        }
        if self._arm["early"]:
            factories["film"] = lambda: IdentityFiLM(c.forecast_projection_width, c.d_model)
        order = tuple(factories) if _initialization_order is None else tuple(_initialization_order)
        if len(order) != len(factories) or set(order) != set(factories):
            raise ValueError("Initialization order must contain each component exactly once")
        self._initialization_seeds = {}
        for name in order:
            # Architecture-specific memory differs in shape; all other common
            # modules share initial parameters across architectures and arms.
            stream = f"{name}:{architecture}" if name == "memory" else name
            component_seed = _stream_seed(seed, stream)
            self._initialization_seeds[name] = component_seed
            mx.random.seed(component_seed)
            setattr(self, name, factories[name]())
        self.head_dropout = nn.Dropout(c.dropout)
        self._positions = mx.array(historical_age_positions(c.history_hours, c.d_model))
        self._lead_indices = mx.arange(c.forecast_leads, dtype=mx.int32)
        self._own_visibility = mx.eye(c.forecast_leads, dtype=mx.float32)
        mx.random.seed(_stream_seed(seed, "training_dropout"))

    @property
    def initialization_seeds(self):
        return dict(self._initialization_seeds)

    @property
    def parameter_count(self):
        return sum(value.size for _, value in tree_flatten(self.trainable_parameters()))

    @property
    def no_weight_decay_names(self):
        return tuple(name for name, _ in tree_flatten(self.trainable_parameters())
                     if name.split(".")[-1] in ("dt_bias", "A_log", "D"))

    @property
    def attention_backend(self):
        return ("explicit_dense_float32_full_attention" if self.config.attention_backend == "dense"
                else "mlx_fast_sdpa_backend_heuristics")

    def _forecast_representation(self, forecast):
        batch, leads = forecast.shape
        if self._arm["forecast"] == "own_lead":
            visibility = mx.broadcast_to(self._own_visibility[None], (batch, leads, leads))
            # where prevents invisible NaN/Inf values from contaminating the
            # head's visible lead; the data builder normally guarantees finite Q.
            visible_values = mx.where(visibility > 0, forecast[:, None, :], 0.0)
            inputs = mx.concatenate([visible_values, visibility], axis=-1)
        else:
            inputs = mx.concatenate([forecast, mx.ones_like(forecast)], axis=-1)
        return nn.relu(self.forecast_projection(inputs))

    def _error_representation(self, errors):
        batch = errors.shape[0]
        # Never multiply by zero: even adversarial nonfinite error values must
        # not enter an error-off arm's graph or affect its predictions.
        inputs = (errors.reshape(batch, -1) if self._arm["errors"] else
                  mx.zeros((batch, self.config.forecast_leads*self.config.error_features), mx.float32))
        return nn.relu(self.error_projection(inputs))

    def __call__(self, history, forecast, error_features, gauge_index):
        # Check NumPy's original dtype before MLX can downcast float64 during
        # conversion; a silent dtype repair would hide a broken data contract.
        for value in (history, forecast, error_features):
            if (not isinstance(value, mx.array) and hasattr(value, "dtype")
                    and str(value.dtype) != "float32"):
                raise ValueError("history, forecast and error_features must use float32")
        history, forecast, error_features, gauge_index = [
            value if isinstance(value, mx.array) else mx.array(value)
            for value in (history, forecast, error_features, gauge_index)
        ]
        c = self.config
        if history.ndim != 3 or history.shape[1:] != (c.history_hours, c.input_features):
            raise ValueError("history must be [N,168,10]")
        batch = history.shape[0]
        if batch < 1 or forecast.shape != (batch, c.forecast_leads):
            raise ValueError("forecast must be [N,18] with nonempty batch")
        if error_features.shape != (batch, c.forecast_leads, c.error_features):
            raise ValueError("error_features must be [N,18,7]")
        if gauge_index.shape != (batch,):
            raise ValueError("gauge_index must be [N]")
        if any(value.dtype != mx.float32 for value in (history, forecast, error_features)):
            raise ValueError("history, forecast and error_features must use float32")
        if gauge_index.dtype not in (mx.int32, mx.int64):
            raise ValueError("gauge_index must use int32 or int64")
        forecast_rep = self._forecast_representation(forecast)
        error_rep = self._error_representation(error_features)
        hidden = self.input_projection(history) + self._positions[None]
        if self._arm["early"]:
            hidden = self.film(hidden, forecast_rep)
        hidden = self.historical_attention(self.memory(hidden))
        context = self.final_norm(hidden)[:, -1, :]
        context = mx.broadcast_to(context[:, None], (batch, c.forecast_leads, c.d_model))
        sites = mx.broadcast_to(self.site_embedding(gauge_index)[:, None],
                                (batch, c.forecast_leads, c.site_embedding_size))
        leads = mx.broadcast_to(self.lead_embedding(self._lead_indices)[None],
                                (batch, c.forecast_leads, c.lead_embedding_size))
        if forecast_rep.ndim == 2:
            forecast_rep = mx.broadcast_to(forecast_rep[:, None],
                                          (batch, c.forecast_leads, c.forecast_projection_width))
        error_rep = mx.broadcast_to(error_rep[:, None],
                                   (batch, c.forecast_leads, c.error_projection_width))
        head = mx.concatenate([context, sites, leads, forecast_rep, error_rep], axis=-1)
        return self.head_out(self.head_dropout(nn.relu(self.head_hidden(head))))[:, :, 0]


def build_model(architecture, variant, seed, config=None):
    """Construct a fresh frozen-size model; never load weights or retain state."""
    return CorrectedResidualMLX(architecture, variant, seed, config)
