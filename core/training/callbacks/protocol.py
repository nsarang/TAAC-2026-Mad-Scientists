"""Observer protocol for training loop hooks."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn


@runtime_checkable
class ObserverProtocol(Protocol):
    """Interface for observer callbacks invoked by the training loop.

    Implementors receive lifecycle events but do not influence control
    flow (no return values).  The ``Diagnostics`` class in this package
    is the primary implementation.
    """

    def on_train_begin(self) -> None:
        """Called once before the first epoch."""
        ...

    def on_epoch_begin(self) -> None:
        """Called at the start of each epoch."""
        ...

    def on_step_begin(self) -> None:
        """Called before each training step."""
        ...

    def on_step_end(
        self,
        *,
        step: int,
        loss: float,
        aux_losses: dict[str, float] = None,
        batch: dict[str, Any],
        grad_norm: float,
        lr_dense: float,
        lr_sparse: float = None,
        fwd_time: float = None,
        bwd_time: float = None,
        model: nn.Module = None,
        logits: torch.Tensor = None,
        dense_optimizer: torch.optim.Optimizer = None,
        scaler: torch.amp.GradScaler = None,
    ) -> None:
        """Called after each training step."""
        ...

    def on_step_eval(
        self,
        *,
        step: int,
        val_auc: float,
        val_logloss: float,
        val_probs: np.ndarray = None,
        val_logits: np.ndarray = None,
        val_labels: np.ndarray = None,
        val_losses: np.ndarray = None,
        val_seq_metadata: dict[str, np.ndarray] = None,
    ) -> None:
        """Called after step-level validation."""
        ...

    def on_epoch_end(
        self,
        *,
        epoch: int,
        num_epochs: int,
        train_loss: float,
        val_auc: float,
        val_logloss: float,
        model: nn.Module,
        train_time: float,
        val_time: float,
        per_domain_aucs: dict[str, float] = None,
        calibration: tuple[float, float] = None,
        val_probs: np.ndarray = None,
        val_logits: np.ndarray = None,
        val_labels: np.ndarray = None,
        val_losses: np.ndarray = None,
        val_seq_metadata: dict[str, np.ndarray] = None,
        sparse_optimizer: torch.optim.Optimizer = None,
        dense_optimizer: torch.optim.Optimizer = None,
        scaler: torch.amp.GradScaler = None,
        oob_stats: dict[str, Any] = None,
        n_val_batches: int = None,
    ) -> None:
        """Called after each epoch's validation."""
        ...

    def on_reinit(
        self,
        *,
        epoch: int,
        reinit_count: int,
        kept_count: int,
        restored_optim: int,
    ) -> None:
        """Called after sparse embedding reinitialization."""
        ...

    def on_train_end(
        self,
        *,
        ckpt_path: str = None,
        early_stopped: bool = False,
        early_stop_epoch: int = 0,
        model: nn.Module = None,
    ) -> None:
        """Called once after the last epoch."""
        ...
