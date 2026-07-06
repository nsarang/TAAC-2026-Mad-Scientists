"""Low-level building blocks: norms, activations, feed-forward networks, gating."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square layer normalization."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize `hidden_states` by root mean square."""
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


def build_norm(hidden_dim: int, norm_type: Literal["layernorm", "rmsnorm"]) -> nn.Module:
    """Return a normalization layer of the given type."""
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if norm_type == "rmsnorm":
        return RMSNorm(hidden_dim)
    raise ValueError(f"Unsupported norm_type '{norm_type}'")


class ScaledTanh(nn.Module):
    """Scaled tanh activation: ``scale * tanh(x / scale)``."""

    def __init__(self, scale: float = 2.0) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError(f"ScaledTanh scale must be > 0, got {scale}")
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled tanh elementwise."""
        return self.scale * torch.tanh(x / self.scale)


def build_activation(name: str, *, scaled_tanh_scale: float = None) -> nn.Module:
    """Return an activation module by name."""
    normalized = str(name).strip().lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "prelu":
        return nn.PReLU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized in {"scaled_tanh", "scaled-tanh"}:
        if scaled_tanh_scale is None:
            raise ValueError("Activation `scaled_tanh` requires `scaled_tanh_scale` to be set.")
        return ScaledTanh(scaled_tanh_scale)
    raise ValueError(f"Unsupported activation '{name}'")


def ffn_activation(projected: torch.Tensor, ffn_type: str) -> torch.Tensor:
    """Apply the FFN activation function (gelu, silu, or swiglu gating)."""
    if ffn_type == "swiglu":
        gate, value = projected.chunk(2, dim=-1)
        return F.silu(gate) * value
    if ffn_type == "silu":
        return F.silu(projected)
    if ffn_type == "gelu":
        return F.gelu(projected)
    raise ValueError(f"Unsupported ffn_type '{ffn_type}'")


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU gating."""
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network with configurable activation."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        *,
        ffn_type: Literal["gelu", "silu", "swiglu"] = "swiglu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ffn_type = ffn_type
        self.dropout = nn.Dropout(dropout)
        up_dim = ffn_dim * 2 if ffn_type == "swiglu" else ffn_dim
        self.up_projection = nn.Linear(hidden_dim, up_dim)
        self.down_projection = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply up-projection, activation, and down-projection."""
        projected = self.up_projection(hidden_states)
        projected = ffn_activation(projected, self.ffn_type)
        projected = self.dropout(projected)
        return self.down_projection(projected)


class GatedFusion(nn.Module):
    """Softmax-gated fusion of multiple branch outputs."""

    def __init__(self, dim: int, num_branches: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, num_branches, bias=False)
        self.value_projs = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(num_branches)]
        )

    def forward(self, branch_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Fuse `branch_outputs` via learned gating weights."""
        stacked = torch.stack(branch_outputs, dim=1)
        gate_input = stacked.mean(dim=1)
        gates = F.softmax(self.gate_proj(gate_input), dim=-1).unsqueeze(-1)
        values = torch.stack(
            [proj(out) for proj, out in zip(self.value_projs, branch_outputs, strict=True)],
            dim=1,
        )
        return (gates * values).sum(dim=1)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (Lim et al., 2021).

    Parameters
    ----------
    input_dim
        Dimensionality of the primary input and output.
    hidden_dim
        Inner projection dimensionality.
    context_dim
        Dimensionality of an optional conditioning context vector.
    dropout
        Dropout rate applied after the inner projection.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, context_dim: int = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, input_dim * 2)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Apply gated residual transformation.

        Parameters
        ----------
        x
            ``(B, input_dim)`` primary input.
        context
            ``(B, context_dim)`` optional conditioning signal.
        """
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.dropout(self.fc2(h))
        value, gate = h.chunk(2, dim=-1)
        return self.norm(x + torch.sigmoid(gate) * value)


class VariableSelectionNetwork(nn.Module):
    """Per-sample feature selection via softmax gating (Lim et al., 2021).

    Scores each feature's relevance through a GRN-based weight network
    and optionally transforms features through per-feature GRNs with
    a compressed global context.

    Parameters
    ----------
    num_features
        Number of input features.
    feature_dim
        Dimensionality of each feature embedding.
    hidden_dim
        Inner dimensionality for all GRNs.
    dropout
        Dropout rate.
    transform_features
        When True, each feature passes through its own GRN before
        weighting. When False, raw embeddings are weighted directly.
    """

    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        transform_features: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        flat_dim = num_features * feature_dim

        self.weight_grn = GatedResidualNetwork(flat_dim, hidden_dim, dropout=dropout)
        self.weight_proj = nn.Linear(flat_dim, num_features)

        self.transform_features = transform_features
        if transform_features:
            self.context_compress = nn.Linear(flat_dim, hidden_dim)
            self.feature_grns = nn.ModuleList(
                [
                    GatedResidualNetwork(
                        feature_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout
                    )
                    for _ in range(num_features)
                ]
            )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select and optionally transform features.

        Parameters
        ----------
        features
            ``(B, num_features, feature_dim)`` stacked feature embeddings.

        Returns
        -------
        output
            ``(B, num_features * feature_dim)`` weighted (and optionally
            transformed) features, flattened.
        weights
            ``(B, num_features)`` per-sample selection weights.
        """
        B, F, E = features.shape
        flat = features.reshape(B, -1)

        weight_input = self.weight_grn(flat)
        weights = torch.softmax(self.weight_proj(weight_input), dim=-1)

        if self.transform_features:
            context = self.context_compress(flat)
            feature_list = features.unbind(dim=1)
            transformed = torch.stack(
                [grn(feature_list[i], context=context) for i, grn in enumerate(self.feature_grns)],
                dim=1,
            )
        else:
            transformed = features

        output = transformed * weights.unsqueeze(-1)
        return output.reshape(B, -1), weights


class InstanceGuidedMask(nn.Module):
    """Instance-conditioned per-dimension gating (Wang et al., 2021).

    Computes a mask from the full input context via a two-layer MLP
    (aggregation → projection), then element-wise multiplies it onto
    LayerNorm'd tokens. The mask is conditioned on all tokens jointly,
    so user dimensions are gated by a function that sees item features
    and vice versa.

    Parameters
    ----------
    num_tokens
        Number of tokens in the input (T).
    d_model
        Dimension per token (D).
    reduction
        Ratio of aggregation layer width to input width. Aggregation has
        ``num_tokens * d_model * reduction`` neurons.
    dropout
        Dropout after aggregation layer.
    """

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        reduction: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        flat_dim = num_tokens * d_model
        agg_dim = int(flat_dim * reduction)
        self.agg = nn.Linear(flat_dim, agg_dim)
        self.proj = nn.Linear(agg_dim, flat_dim)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply instance-guided mask.

        Parameters
        ----------
        tokens
            ``(B, T, D)`` input tokens.

        Returns
        -------
        torch.Tensor
            ``(B, T, D)`` masked tokens.
        """
        B, T, D = tokens.shape
        flat = tokens.reshape(B, -1)
        mask = self.proj(self.dropout(F.relu(self.agg(flat))))
        mask = mask.view(B, T, D)
        normed = self.norm(tokens)
        return mask * normed
