"""Typed, lazy-computed context objects passed to diagnostic codes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

# ─────────────────────────────────────────────────────────────────────────────
# Parameter-group helpers (used by lazy properties and by codes_optim)
# ─────────────────────────────────────────────────────────────────────────────


def _group_names_from_patterns(patterns: list[tuple[str, str]]) -> list[str]:
    return [name for name, _ in patterns]


def _match_group(name_lower: str, patterns: list[tuple[str, str]]) -> str:
    for gn, pat in patterns:
        if re.search(pat, name_lower):
            return gn
    return "other"


def _compute_grouped_norms(
    model: nn.Module,
    patterns: list[tuple[str, str]],
    source: str = "param",
) -> dict[str, float]:
    """L2 norm grouped by name pattern.

    Parameters
    ----------
    model
        The model whose parameters to measure.
    patterns
        List of (group_name, regex_pattern) tuples.
    source
        ``"param"`` for parameter norms, ``"grad"`` for gradient norms.
    """
    names = [*_group_names_from_patterns(patterns), "other"]
    sq: dict[str, float] = dict.fromkeys(names, 0.0)
    for name, p in model.named_parameters():
        if source == "grad":
            if p.grad is None:
                continue
            val = float(p.grad.norm(2).item() ** 2)
        else:
            if not p.requires_grad:
                continue
            val = float(p.data.norm(2).item() ** 2)
        sq[_match_group(name.lower(), patterns)] += val
    return {k: float(np.sqrt(v)) for k, v in sq.items()}


@dataclass(slots=True)
class StepContext:
    """Immutable snapshot of one training step, with lazily derived fields.

    Raw fields are set by the container at step-end. Derived fields (label
    stats, grad norms, per-class loss) are computed on first access and cached
    for the lifetime of the context -- so multiple codes reading the same
    derived field pay only once.
    """

    # ── Raw (set by container) ───────────────────────────────────────────────
    step: int = 0
    loss: float = 0.0
    aux_losses: dict[str, float] = field(default_factory=dict)
    batch: dict[str, Any] = field(default_factory=dict)
    grad_norm: float = 0.0
    lr_dense: float = 0.0
    lr_sparse: float = None
    fwd_time: float = None
    bwd_time: float = None
    model: nn.Module = None
    logits: torch.Tensor = None
    dense_optimizer: torch.optim.Optimizer = None
    scaler: torch.amp.GradScaler = None
    # ── Timing (set by container) ─────────────────────────────────────────────
    data_time: float = 0.0
    step_start_time: float = 0.0

    # ── Lazy cache slots ─────────────────────────────────────────────────────
    _label_stats: dict[str, float] = field(default=None, init=False, repr=False)
    _per_class_loss: dict[str, float] = field(default=None, init=False, repr=False)

    @property
    def batch_size(self) -> int:
        """Number of samples in the current batch."""
        label = self.batch.get("label")
        return label.shape[0] if label is not None else 0

    @property
    def label_stats(self) -> dict[str, float]:
        """Mean, sum, and count over batch labels."""
        if self._label_stats is None:
            label = self.batch.get("label")
            if label is None:
                self._label_stats = {"mean": 0.0, "sum": 0.0, "count": 0}
            else:
                arr = label.float().cpu().numpy()
                self._label_stats = {
                    "mean": float(np.mean(arr)),
                    "sum": float(np.sum(arr)),
                    "count": len(arr),
                }
        return self._label_stats

    @property
    def per_class_loss(self) -> dict[str, float]:
        """Per-class BCE split by label; None values when logits unavailable."""
        if self._per_class_loss is None:
            label = self.batch.get("label")
            if self.logits is None or label is None:
                self._per_class_loss = {"pos": None, "neg": None}
            else:
                with torch.no_grad():
                    logits_cpu = self.logits.detach().cpu().view(-1)
                    label_cpu = label.detach().cpu().view(-1)
                    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits_cpu, label_cpu.float(), reduction="none"
                    )
                    mask_pos = label_cpu == 1
                    pos = float(per_sample[mask_pos].mean()) if mask_pos.any() else None
                    neg = float(per_sample[~mask_pos].mean()) if (~mask_pos).any() else None
                self._per_class_loss = {"pos": pos, "neg": neg}
        return self._per_class_loss


@dataclass(slots=True)
class EpochContext:
    """Snapshot of one epoch's end state."""

    epoch: int = 0
    num_epochs: int = 0
    train_loss: float = None
    val_auc: float = 0.0
    val_logloss: float = 0.0
    model: nn.Module = None
    train_time: float = 0.0
    val_time: float = 0.0
    val_data_time: float = 0.0
    val_fwd_time: float = 0.0
    n_val_batches: int = 0
    per_domain_aucs: dict[str, float] = None
    calibration: tuple[float, float] = None
    val_probs: np.ndarray = None
    val_logits: np.ndarray = None
    val_labels: np.ndarray = None
    val_losses: np.ndarray = None
    sparse_optimizer: torch.optim.Optimizer = None
    dense_optimizer: torch.optim.Optimizer = None
    scaler: torch.amp.GradScaler = None
    oob_stats: dict[str, Any] = None
    val_seq_metadata: dict[str, np.ndarray] = None

    # Container-computed derived state
    step: int = 0
    eta_sec: float = 0.0
    best_val_auc: float = 0.0
    best_val_epoch: int = 0
    delta: float = 0.0
    epochs_since_improvement: int = 0
    total_samples_seen: int = 0
