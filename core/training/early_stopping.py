"""Early stopping tracker for validation-metric-monitored training."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from torch import nn

LOG = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a higher-is-better validation metric plateaus.

    Parameters
    ----------
    patience
        Non-improving validation calls tolerated before ``early_stop`` flips.
    delta
        Minimum absolute improvement to count as progress.
    checkpoint_path
        When set, ``model.state_dict()`` is persisted here on every
        improvement. Parent directories are created automatically.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.0,
        checkpoint_path: str = None,
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path

        self.best_score: float = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_extra_metrics: dict[str, Any] = None

    def is_improved(self, score: float) -> bool:
        """Return True when `score` beats the current best by at least `delta`."""
        if self.best_score is None:
            return True
        return score > self.best_score + self.delta

    def __call__(
        self,
        score: float,
        model: nn.Module,
        extra_metrics: dict[str, Any] = None,
    ) -> bool:
        """Feed a validation score. Returns True when the score is a new best.

        On improvement the model state_dict is deep-copied in memory and,
        when ``checkpoint_path`` is set, persisted to disk.
        """
        improved = self.is_improved(score)

        if self.best_score is None:
            self.best_score = score
            self.best_extra_metrics = extra_metrics
            if self.checkpoint_path:
                self._save(model)
            return True

        if not improved:
            self.counter += 1
            LOG.info(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False

        LOG.info("EarlyStopping counter reset")
        self.best_score = score
        self.best_extra_metrics = extra_metrics
        self.counter = 0
        if self.checkpoint_path:
            self._save(model)
        return True

    def _save(self, model: nn.Module) -> None:
        assert self.checkpoint_path is not None
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
