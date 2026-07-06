"""Compass diagnostic code: logs all compass metrics to TensorBoard."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.training.callbacks.diagnostics.base import DiagBase, _parse_context, _try_numeric
from core.training.callbacks.diagnostics.context import EpochContext, StepContext


class CompassCode(DiagBase):
    """Compass guide metrics — logs all numeric keys from intervene() to TB."""

    code = "COMPASS"
    config_key = "compass"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._step_metrics: list[dict[str, float]] = []
        self._epoch_metrics: list[dict[str, float]] = []

    def receive_metrics(self, metrics: dict[str, float]) -> None:
        """Called by the training loop after compass.intervene().

        Not part of the standard DiagBase step() flow — the compass
        metrics come from outside the normal observer protocol since
        the compass operates between backward and optimizer.step().
        """
        if metrics:
            self._step_metrics.append(metrics)
            self._epoch_metrics.append(metrics)

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings."""
        if phase == "step":
            return self._collect_step(ctx)
        if phase == "epoch":
            return self._collect_epoch(ctx)
        return []

    def _collect_step(self, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        if not self._step_metrics:
            return []

        m = self._step_metrics[-1]
        step = ctx.step if isinstance(ctx, StepContext) else 0

        if self.writer:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"compass/{k}", v, step)

        parts = [f"{k}={v:.8g}" for k, v in m.items() if isinstance(v, (int, float))]
        return [",".join(parts)]

    def _collect_epoch(self, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        if not self._epoch_metrics:
            return []

        epoch = ctx.epoch if isinstance(ctx, EpochContext) else 0

        # Aggregate all numeric keys across the epoch
        all_keys: set[str] = set()
        for m in self._epoch_metrics:
            all_keys.update(k for k, v in m.items() if isinstance(v, (int, float)))

        means: dict[str, float] = {}
        for k in sorted(all_keys):
            vals = [m[k] for m in self._epoch_metrics if k in m]
            if vals:
                means[k] = float(np.mean(vals))

        if self.writer:
            for k, v in means.items():
                self.writer.add_scalar(f"compass_epoch/{k}_mean", v, epoch)

        parts = [f"{k}={v:.8g}" for k, v in means.items()]
        parts.append(f"n_steps={len(self._epoch_metrics)}")
        return [",".join(parts)]

    def flush(self) -> None:
        """Clear step-level accumulator after emission."""
        self._step_metrics.clear()

    def epoch_reset(self) -> None:
        """Clear epoch-level accumulator at epoch start."""
        self._epoch_metrics.clear()

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        entry: dict[str, Any] = {}
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                entry[k] = _try_numeric(v)
        ctx = _parse_context(context)
        step = ctx.get("step", ctx.get("epoch", 0))
        accum.setdefault("steps", {})[step] = entry
