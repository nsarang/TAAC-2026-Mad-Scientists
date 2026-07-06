"""Top-k Mixture-of-Experts building block."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ExpertMLP(nn.Module):
    """Top-k MoE layer: routes each sample to k-of-N expert MLPs.

    Each expert is a two-layer MLP: in_dim → hidden_dim → out_dim.

    Parameters
    ----------
    in_dim
        Input dimension.
    hidden_dim
        Hidden dimension within each expert.
    out_dim
        Output dimension per expert.
    n_experts
        Total number of expert MLPs.
    top_k
        Number of experts activated per sample.
    dropout_rate
        Dropout within each expert.
    balance_weight
        Multiplier for the load-balancing auxiliary loss.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_experts: int,
        top_k: int,
        dropout_rate: float = 0.01,
        balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.balance_weight = balance_weight

        self.router = nn.Linear(in_dim, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, out_dim),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route input to top-k experts and return weighted output + balance loss.

        Parameters
        ----------
        x
            Input tensor [B, in_dim].

        Returns
        -------
        tuple
            ``(output [B, out_dim], balance_loss scalar)``.
        """
        router_logits = self.router(x)  # [B, n_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        top_weights, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, N, D]
        idx = top_indices.unsqueeze(-1).expand(-1, -1, expert_outs.shape[-1])
        selected = expert_outs.gather(1, idx)  # [B, top_k, D]
        output = (selected * top_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # Switch Transformer balance loss: N * sum(freq_i * avg_prob_i)
        top1 = router_probs.argmax(dim=-1)
        freq = torch.zeros(self.n_experts, device=router_probs.device)
        freq.scatter_add_(0, top1, torch.ones_like(top1, dtype=freq.dtype))
        freq = freq / freq.sum()
        avg_prob = router_probs.mean(dim=0)
        balance_loss = self.balance_weight * self.n_experts * (freq * avg_prob).sum()

        return output, balance_loss
