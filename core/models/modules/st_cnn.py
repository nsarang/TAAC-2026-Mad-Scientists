"""Short-term causal CNN for recent-event encoding.

Context-conditioned Conv1d over a fixed window of recent sequence tokens,
with gated residual output. Adds a recency-aware signal to the fused
sequence representation without modifying the DIN attention path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

_SHORT_TERM_CNN_PARAMS = ("window", "kernel_size", "dropout", "cap", "scale_init")


class ShortTermCausalCNN(nn.Module):
    """Context-conditioned, order-aware short-term view over recent events."""

    def __init__(
        self,
        d_model: int,
        window: int,
        kernel_size: int,
        dropout: float,
        cap: float,
        scale_init: float,
    ) -> None:
        super().__init__()
        self.window = int(window)
        self.kernel_size = int(kernel_size)
        self._cap = float(cap)
        self.cond_proj = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(d_model, d_model, self.kernel_size)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a gated short-term residual [B, D]."""
        recent = tokens[:, : self.window]
        keep = (~padding_mask[:, : self.window]).unsqueeze(-1).to(tokens.dtype)
        cond = self.cond_proj(context).unsqueeze(1)
        recent = recent + cond
        x = (recent * keep).transpose(1, 2)
        x = F.pad(x, (0, self.kernel_size - 1))
        x = self.conv(x).transpose(1, 2)
        x = self.dropout(self.act(self.norm(x)))
        x = x * keep
        pooled = x.sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        gate = self._cap * torch.tanh(self.scale)
        return gate * self.proj(pooled)


def resolve_short_term_cnn_params(
    defaults: dict[str, Any], per_domain: dict[str, Any], domain: str
) -> dict[str, Any]:
    """Merge shared defaults with optional per-domain overrides."""
    params = dict(defaults)
    if domain in per_domain:
        params.update(dict(per_domain[domain]))
    missing = [k for k in _SHORT_TERM_CNN_PARAMS if k not in params]
    if missing:
        raise ValueError(f"short_term_cnn missing params for domain '{domain}': {missing}")
    return {
        "window": int(params["window"]),
        "kernel_size": int(params["kernel_size"]),
        "dropout": float(params["dropout"]),
        "cap": float(params["cap"]),
        "scale_init": float(params["scale_init"]),
    }
