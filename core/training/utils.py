"""Training utilities."""

from __future__ import annotations

import numpy as np

from core.evaluation.metrics import binary_auc, sigmoid


class ReservoirSampler:
    """Fixed-size reservoir of (logit, label) pairs using Algorithm R."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._logits = np.empty(capacity, dtype=np.float64)
        self._labels = np.empty(capacity, dtype=np.float64)
        self._fill = 0
        self.n_seen = 0

    def update(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Ingest a batch into the reservoir."""
        n = len(logits)
        if n == 0:
            return
        if self._fill < self.capacity:
            take = min(self.capacity - self._fill, n)
            self._logits[self._fill : self._fill + take] = logits[:take]
            self._labels[self._fill : self._fill + take] = labels[:take]
            self._fill += take
            self.n_seen += take
            if take < n:
                self.update(logits[take:], labels[take:])
            return
        idx = np.random.randint(0, self.n_seen + 1 + np.arange(n))
        mask = idx < self.capacity
        self._logits[idx[mask]] = logits[mask]
        self._labels[idx[mask]] = labels[mask]
        self.n_seen += n

    def compute_auc(self) -> float | None:
        """Return AUC from the reservoir, or None if insufficient data."""
        if self._fill == 0:
            return None
        labels = self._labels[: self._fill]
        if len(np.unique(labels)) < 2:
            return None
        probs = sigmoid(self._logits[: self._fill])
        return binary_auc(labels, probs)

    def reset(self) -> None:
        """Clear for next epoch."""
        self._fill = 0
        self.n_seen = 0
