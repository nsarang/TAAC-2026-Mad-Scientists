"""Base class defining the training loop's model contract."""

from __future__ import annotations

import torch
from torch import nn


class TrainableModel(nn.Module):
    """Interface expected by the DragonChariot training loop.

    Subclasses must implement get_sparse_params / get_dense_params.
    All other methods have no-op defaults.
    """

    def get_sparse_params(self) -> list[nn.Parameter]:
        """Return embedding parameters managed by the sparse optimizer."""
        raise NotImplementedError

    def get_dense_params(self) -> list[nn.Parameter]:
        """Return parameters managed by the dense optimizer."""
        raise NotImplementedError

    def update_learning_rate(self, lr: float = None) -> None:
        """Sync embedding LR for fused-optimizer backends.

        When `lr` is None, the model uses its configured default.
        No-op by default.
        """

    def reinit_high_cardinality_params(self, cardinality_threshold: int) -> set[int]:
        """Reinit tables above threshold; return their data_ptrs."""
        return set()

    def snapshot_low_cardinality_embs(self, vocab_threshold: int) -> dict[str, torch.Tensor]:
        """Clone weights for embedding tables with vocab <= threshold."""
        return {}

    def restore_emb_snapshot(self, snapshot: dict[str, torch.Tensor]) -> set[int]:
        """Restore previously snapshotted weights; return restored data_ptrs."""
        return set()

    def pretext_trainable_params(self) -> set[int]:
        """Return data_ptrs of params unfrozen during the pretext phase."""
        return set()
