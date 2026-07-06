"""Exponential Moving Average / Stochastic Weight Averaging of model parameters."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn

LOG = logging.getLogger(__name__)


class ModelEMA:
    """Maintains a moving average of model parameters (EMA or SWA).

    NOTE: only shadows dense parameters. Sparse embedding tables are excluded
    via `skip_params` at construction time (passed from the engine as the set
    of sparse param data_ptrs). Reinit of embeddings does not require any
    shadow reset because those params are never tracked.

    Shadow weights are updated after each optimizer step. Two modes:

    - **ema**: ``shadow = decay * shadow + (1 - decay) * param``
    - **swa**: ``shadow += (param - shadow) / n`` (uniform running mean),
      collection starts after `start_step` and triggers every `collect_every` steps.

    Parameters
    ----------
    model
        The model whose parameters to shadow. Handles ``torch.compile``
        wrapped models via ``_orig_mod``.
    decay
        EMA decay factor (only used in ema mode).
    mode
        ``"ema"`` or ``"swa"``.
    start_step
        First step at which to begin averaging. Before this, the shadow
        tracks the model exactly (no exponential decay or collection).
        Applies to both EMA and SWA modes.
    collect_every
        SWA only: collect a snapshot every N steps.
    skip_params
        Set of ``data_ptr()`` values for parameters to exclude from
        shadowing (e.g. sparse embedding tables).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        mode: str = "ema",
        start_step: int = 0,
        collect_every: int = 1,
        collect_every_epoch: int = None,
        start_epoch: int = 0,
        skip_params: set[int] = None,
    ) -> None:
        if mode not in ("ema", "swa"):
            raise ValueError(f"Unknown averaging mode: {mode!r}, expected 'ema' or 'swa'")
        self.decay = decay
        self.mode = mode
        self.start_step = start_step
        self.collect_every = collect_every
        self.collect_every_epoch = collect_every_epoch
        self._start_epoch = start_epoch
        self._epoch_mode = collect_every_epoch is not None
        self._current_epoch = 0
        self._n_collected = 0
        self._step = 0
        self._model = model
        self._shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        skip = skip_params or set()
        src = getattr(model, "_orig_mod", model)
        for name, param in src.named_parameters():
            if param.requires_grad and param.data_ptr() not in skip:
                self._shadow[name] = param.data.clone()

    @classmethod
    def from_config(
        cls, cfg, model: nn.Module, steps_per_epoch: int, skip_params: set[int] = None
    ) -> "ModelEMA":
        """Construct from an OmegaConf ema config node.

        Parameters
        ----------
        cfg
            The ``train.ema`` config namespace. Expected keys:
            ``mode``, ``decay``, ``window_fraction``, ``start_epoch``, ``collect_every``.
        model
            Model to shadow.
        steps_per_epoch
            Used to resolve ``window_fraction`` → decay and ``start_epoch`` → start_step.
        skip_params
            Parameter data_ptr set to exclude (sparse embeddings).
        """
        mode = cfg.mode

        if mode == "swa":
            collect_every_epoch = getattr(cfg, "collect_every_epoch", None)
            start_epoch = cfg.start_epoch
            if collect_every_epoch is not None:
                instance = cls(
                    model,
                    mode="swa",
                    collect_every_epoch=collect_every_epoch,
                    start_epoch=start_epoch,
                    skip_params=skip_params,
                )
                LOG.info(
                    f"SWA enabled (epoch-level): start_epoch={start_epoch}, "
                    f"collect_every_epoch={collect_every_epoch}"
                )
            else:
                start_step = int(start_epoch * steps_per_epoch)
                instance = cls(
                    model,
                    mode="swa",
                    start_step=start_step,
                    collect_every=cfg.collect_every,
                    skip_params=skip_params,
                )
                LOG.info(
                    f"SWA enabled (step-level): start_step={start_step} "
                    f"(epoch {start_epoch}), collect_every={cfg.collect_every} steps"
                )
        else:
            has_decay = cfg.decay is not None
            has_fraction = cfg.window_fraction is not None
            if has_decay and has_fraction:
                raise ValueError(
                    "ema config specifies both `decay` and `window_fraction`; use one or the other"
                )
            if not has_decay and not has_fraction:
                raise ValueError("ema config must specify either `decay` or `window_fraction`")
            if has_fraction:
                window_steps = cfg.window_fraction * steps_per_epoch
                decay = 1.0 - 1.0 / max(1.0, window_steps)
                LOG.info(
                    f"EMA enabled with decay={decay:.6f} "
                    f"(window_fraction={cfg.window_fraction}, steps_per_epoch={steps_per_epoch})"
                )
            else:
                decay = cfg.decay
                LOG.info(f"EMA enabled with decay={decay}")
            start_epoch = cfg.start_epoch
            start_step = int(start_epoch * steps_per_epoch) if start_epoch else 0
            if start_step > 0:
                LOG.info(f"  EMA start deferred to step {start_step} (epoch {start_epoch})")
            instance = cls(
                model, decay=decay, mode="ema", start_step=start_step, skip_params=skip_params
            )

        n_shadow = len(instance._shadow)
        n_total = sum(1 for p in model.parameters() if p.requires_grad)
        LOG.info(f"  shadowing {n_shadow}/{n_total} params (sparse excluded)")
        return instance

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow weights toward current model parameters.

        In both modes, before ``start_step`` the shadow simply copies the
        current params (no averaging). After ``start_step``:

        - **EMA**: exponential moving average with configured decay.
        - **SWA epoch_mode**: collection deferred to ``notify_epoch_end``.
        - **SWA step_mode**: collects running-average snapshot every
          ``collect_every`` steps.
        """
        self._step += 1
        if self.mode == "ema":
            self._update_ema()
        else:
            self._update_swa()

    def _update_ema(self) -> None:
        src = getattr(self._model, "_orig_mod", self._model)
        if self._step < self.start_step:
            # Before start: shadow tracks model exactly (no averaging)
            for name, param in src.named_parameters():
                if name in self._shadow:
                    self._shadow[name].copy_(param.data)
            return
        d = self.decay
        for name, param in src.named_parameters():
            if name in self._shadow:
                self._shadow[name].lerp_(param.data, 1.0 - d)

    def _update_swa(self) -> None:
        if self._epoch_mode:
            # Epoch-level: collection happens only in notify_epoch_end
            return
        if self._step < self.start_step:
            # Before collection starts, keep shadow = current params
            src = getattr(self._model, "_orig_mod", self._model)
            for name, param in src.named_parameters():
                if name in self._shadow:
                    self._shadow[name].copy_(param.data)
            return
        if (self._step - self.start_step) % self.collect_every != 0:
            return
        self._collect_snapshot()

    def _collect_snapshot(self) -> None:
        self._n_collected += 1
        src = getattr(self._model, "_orig_mod", self._model)
        n = self._n_collected
        for name, param in src.named_parameters():
            if name in self._shadow:
                self._shadow[name].add_((param.data - self._shadow[name]) / n)

    def notify_epoch_end(self, epoch: int) -> None:
        """Called by the training loop at epoch boundary.

        In epoch-level SWA mode, collects a snapshot if the epoch is past
        `start_epoch` and aligned with `collect_every_epoch`.
        """
        self._current_epoch = epoch
        if self.mode != "swa" or not self._epoch_mode:
            return
        if epoch < self._start_epoch:
            return
        epochs_since_start = epoch - self._start_epoch
        if epochs_since_start % self.collect_every_epoch != 0:
            return
        self._collect_snapshot()
        LOG.info(f"SWA: collected epoch {epoch} snapshot (n={self._n_collected})")

    def apply(self) -> None:
        """Swap model parameters with shadow (save originals for restore).

        In SWA mode, if no snapshots have been collected yet, this is a no-op.
        """
        if self.mode == "swa" and self._n_collected == 0:
            self._backup.clear()
            return
        src = getattr(self._model, "_orig_mod", self._model)
        self._backup.clear()
        for name, param in src.named_parameters():
            if name in self._shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self._shadow[name])

    def restore(self) -> None:
        """Restore original model parameters from backup."""
        src = getattr(self._model, "_orig_mod", self._model)
        for name, param in src.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Context manager that temporarily swaps in averaged parameters."""
        self.apply()
        try:
            yield
        finally:
            self.restore()

    @property
    def n_collected(self) -> int:
        """Number of snapshots collected (SWA mode)."""
        return self._n_collected

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable state."""
        return {
            "decay": self.decay,
            "mode": self.mode,
            "shadow": self._shadow,
            "n_collected": self._n_collected,
            "step": self._step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from a previously saved dict."""
        self.decay = state["decay"]
        self.mode = state.get("mode", "ema")
        self._n_collected = state.get("n_collected", 0)
        self._step = state.get("step", 0)
        for name, tensor in state["shadow"].items():
            if name in self._shadow:
                self._shadow[name].copy_(tensor)
