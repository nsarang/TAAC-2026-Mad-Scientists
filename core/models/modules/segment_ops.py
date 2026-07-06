"""Segment operations for jagged tensors.

All ops accept flat values + cu_seqlens (cumulative sequence lengths, shape [B+1]).
Pure PyTorch — works on CPU, MPS, and CUDA without custom kernels.
"""

from __future__ import annotations

import torch


def _segment_ids(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """Map each position to its segment index. Shape (total,)."""
    B = cu_seqlens.shape[0] - 1
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.repeat_interleave(torch.arange(B, device=cu_seqlens.device), lengths.long())


def segment_sum(values: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Sum values within each segment.

    Parameters
    ----------
    values
        Shape ``(total,)`` or ``(total, D)``.
    cu_seqlens
        Cumulative lengths, shape ``(B+1,)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` or ``(B, D)``.
    """
    B = cu_seqlens.shape[0] - 1
    total = values.shape[0]
    seg_ids = _segment_ids(cu_seqlens, total)

    if values.dim() == 1:
        out = torch.zeros(B, dtype=values.dtype, device=values.device)
        out.scatter_reduce_(0, seg_ids, values, reduce="sum", include_self=False)
    else:
        D = values.shape[1]
        out = torch.zeros(B, D, dtype=values.dtype, device=values.device)
        idx = seg_ids.unsqueeze(1).expand(-1, D)
        out.scatter_reduce_(0, idx, values, reduce="sum", include_self=False)
    return out


def segment_max(values: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Max within each segment.

    Parameters
    ----------
    values
        Shape ``(total,)``.
    cu_seqlens
        Cumulative lengths, shape ``(B+1,)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)``.
    """
    B = cu_seqlens.shape[0] - 1
    total = values.shape[0]
    seg_ids = _segment_ids(cu_seqlens, total)
    out = torch.full((B,), float("-inf"), dtype=values.dtype, device=values.device)
    out.scatter_reduce_(0, seg_ids, values, reduce="amax", include_self=False)
    return out


def segment_softmax(values: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax within each segment.

    Parameters
    ----------
    values
        Shape ``(total,)``.
    cu_seqlens
        Cumulative lengths, shape ``(B+1,)``.

    Returns
    -------
    torch.Tensor
        Shape ``(total,)``, sums to 1 within each segment.
    """
    B = cu_seqlens.shape[0] - 1
    total = values.shape[0]
    seg_ids = _segment_ids(cu_seqlens, total)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]

    seg_max = segment_max(values, cu_seqlens)
    expanded_max = seg_max.repeat_interleave(lengths.long())
    exp_values = (values - expanded_max).exp()

    seg_sum = torch.zeros(B, dtype=values.dtype, device=values.device)
    seg_sum.scatter_reduce_(0, seg_ids, exp_values, reduce="sum", include_self=False)
    expanded_sum = seg_sum.repeat_interleave(lengths.long())

    return exp_values / expanded_sum
