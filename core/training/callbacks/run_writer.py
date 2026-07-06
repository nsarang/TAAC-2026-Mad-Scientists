"""Observer that persists training run data to a SQLiteStore."""

from __future__ import annotations

from typing import Any

from torch import nn

from core.utils.logging import SQLiteStore


class RunWriter:
    """Observer that writes training metrics and artifacts to a SQLiteStore."""

    def __init__(self, store: SQLiteStore, config: dict = None) -> None:
        self._store = store
        self._config = config

    def on_train_begin(self) -> None:
        """Persist config metadata at training start."""
        if self._config is not None:
            self._store.log_metadata("config", self._config)

    def on_epoch_begin(self) -> None:
        """No-op."""

    def on_step_begin(self) -> None:
        """No-op."""

    def on_step_end(self, **kw: Any) -> None:
        """No-op."""

    def on_step_eval(
        self,
        *,
        step: int,
        val_auc: float,
        val_logloss: float,
        **kw: Any,
    ) -> None:
        """Write step-level validation metrics."""
        self._store.log_metrics(
            {"val/step_auc": val_auc, "val/step_logloss": val_logloss},
            step=step,
        )

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
        **kw: Any,
    ) -> None:
        """Write epoch-level training and validation metrics."""
        metrics: dict[str, float] = {
            "train/loss": train_loss,
            "val/auc": val_auc,
            "val/logloss": val_logloss,
            "train/time": train_time,
            "val/time": val_time,
        }
        self._store.log_metrics(metrics, step=epoch)

    def on_reinit(self, **kw: Any) -> None:
        """No-op."""

    def on_train_end(self, **kw: Any) -> None:
        """Flush and close the store."""
        self._store.close()
