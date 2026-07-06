"""Save/load training checkpoints with best-model tracking."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

LOG = logging.getLogger(__name__)
from torch import nn


class CheckpointManager:
    """Tracks best model and saves/loads training state.

    When `ckpt_name_template` is set, checkpoints are saved as named
    sub-directories (``<name>/model.pt``, ``<name>.best_model/model.pt``)
    matching the AngelML convention. Otherwise falls back to flat
    ``last.pt``/``best.pt`` files.

    Parameters
    ----------
    checkpoint_dir
        Root directory for checkpoint files.
    config_dict
        Resolved config as a plain dict. Saved as ``config.yaml`` sidecar in
        named checkpoint directories.
    ckpt_name_template
        Pre-resolved template with Python format placeholders for runtime
        values (``{step}``, ``{epoch}``). When None, uses flat
        ``last.pt``/``best.pt``.
    metric
        Metric key in the validation metrics dict to track.
    mode
        ``"max"`` (higher is better) or ``"min"`` (lower is better).
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        config_dict: dict[str, Any] = None,
        ckpt_name_template: str = None,
        schema_path: str | Path = None,
        metric: str = "auc",
        mode: str = "max",
        amp_dtype: torch.dtype = None,
        fit_state: dict[str, Any] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config_dict = config_dict
        self.ckpt_name_template = ckpt_name_template
        self.schema_path = Path(schema_path) if schema_path else None
        self.metric = metric
        self.mode = mode
        self.amp_dtype = amp_dtype
        self.fit_state = fit_state
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.best_dir: Path = None

    @property
    def _named(self) -> bool:
        return self.ckpt_name_template is not None

    @property
    def best_path(self) -> Path:
        """Path to the best checkpoint file."""
        if self._named and self.best_dir is not None:
            return self.best_dir / "model.pt"
        return self.checkpoint_dir / "best.pt"

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def _get_model_state(self, model: nn.Module) -> dict[str, Any]:
        if hasattr(model, "_orig_mod"):
            return model._orig_mod.state_dict()
        return model.state_dict()

    def _resolve_name(self, runtime: dict[str, Any]) -> str:
        return self.ckpt_name_template.format(**runtime)

    def _write_sidecar(self, ckpt_dir: Path) -> None:
        if self.config_dict is not None:
            with open(ckpt_dir / "config.yaml", "w") as f:
                yaml.safe_dump(self.config_dict, f, default_flow_style=False, sort_keys=False)
        if self.schema_path and self.schema_path.exists():
            shutil.copy2(self.schema_path, ckpt_dir / "schema.json")

    def _remove_old_best_dirs(self) -> None:
        for old_dir in self.checkpoint_dir.glob("*.best_model"):
            shutil.rmtree(old_dir)
            LOG.info(f"Removed old best_model dir: {old_dir}")

    def _build_state(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        state = {
            "epoch": epoch,
            "model_state_dict": self._get_model_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
        }
        if self.amp_dtype is not None:
            state["amp_dtype"] = str(self.amp_dtype).removeprefix("torch.")
        if self.fit_state:
            state["fit_state"] = self.fit_state
        return state

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str, Any],
        runtime: dict[str, Any] = None,
        ema=None,
    ) -> Path:
        """Save training state and update best model if metric improved.

        Parameters
        ----------
        epoch
            Current epoch number.
        model
            Model whose state dict is saved.
        optimizer
            Optimizer whose state dict is saved.
        metrics
            Dict of metric values (must include the tracked metric).
        runtime
            Required for named checkpoints. Dict with ``step``, ``epoch``, etc.
        ema
            Optional :class:`ModelEMA` whose shadow weights are saved alongside
            the checkpoint.

        Returns
        -------
        Path
            Path to the saved checkpoint.
        """
        tracked = metrics.get(self.metric, 0.0)
        if isinstance(tracked, dict):
            tracked = tracked.get("value", 0.0)

        improved = self._is_better(tracked)
        if improved:
            self.best_value = tracked
            self.best_epoch = epoch

        state = self._build_state(epoch, model, optimizer, metrics)
        if ema is not None:
            state["ema_state_dict"] = ema.state_dict()

        if not self._named:
            last_path = self.checkpoint_dir / "last.pt"
            torch.save(state, last_path)
            if improved:
                shutil.copy2(last_path, self.checkpoint_dir / "best.pt")
                LOG.info(f"New best {self.metric}={self.best_value:.4f} at epoch {epoch}")
            return last_path

        name = self._resolve_name(runtime or {})
        ckpt_dir = self.checkpoint_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, ckpt_dir / "model.pt")
        self._write_sidecar(ckpt_dir)
        LOG.info(f"Saved checkpoint to {ckpt_dir / 'model.pt'}")

        if improved:
            self._remove_old_best_dirs()
            best_dir = self.checkpoint_dir / (name + ".best_model")
            shutil.copytree(ckpt_dir, best_dir)
            self.best_dir = best_dir
            LOG.info(f"New best {self.metric}={self.best_value:.4f} at epoch {epoch} -> {best_dir}")

        return ckpt_dir

    @staticmethod
    def resolve_path(ckpt_dir: Path) -> Path:
        """Find the best available checkpoint file in a directory.

        Resolution order: best.pt > model.pt > *.best_model/model.pt >
        last.pt > */model.pt (latest).
        """
        ckpt_dir = Path(ckpt_dir)
        best_pt = ckpt_dir / "best.pt"
        if best_pt.exists():
            return best_pt
        model_pt = ckpt_dir / "model.pt"
        if model_pt.exists():
            return model_pt
        candidates = sorted(ckpt_dir.glob("*.best_model/model.pt"))
        if candidates:
            return candidates[-1]
        last_pt = ckpt_dir / "last.pt"
        if last_pt.exists():
            return last_pt
        pts = sorted(ckpt_dir.glob("*/model.pt"))
        if not pts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        return pts[-1]

    @staticmethod
    def load(
        path: Path | str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load a checkpoint into model (and optionally optimizer).

        Handles both flat checkpoints (``last.pt``/``best.pt``) and named
        checkpoints (``<dir>/model.pt``). Both contain full training state.

        Returns
        -------
        dict
            The full checkpoint dict.
        """
        path = Path(path)
        raw = torch.load(path, map_location=device, weights_only=True)

        state_dict = raw.get("model_state_dict", raw)
        if hasattr(model, "_orig_mod"):
            model._orig_mod.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        if optimizer is not None and "optimizer_state_dict" in raw:
            optimizer.load_state_dict(raw["optimizer_state_dict"])
        return raw

    def resume(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device = "cpu",
    ) -> int:
        """Resume from last checkpoint if one exists.

        Returns
        -------
        int
            The next epoch to train (1 if no checkpoint found).
        """
        if self._named:
            candidates = sorted(
                (
                    p
                    for p in self.checkpoint_dir.glob("*/model.pt")
                    if ".best_model" not in p.parent.name
                ),
                key=lambda p: p.stat().st_mtime,
            )
            if not candidates:
                LOG.info("No checkpoint found, starting from epoch 1")
                return 1
            last_path = candidates[-1]
        else:
            last_path = self.checkpoint_dir / "last.pt"
            if not last_path.exists():
                LOG.info("No checkpoint found, starting from epoch 1")
                return 1

        checkpoint = self.load(last_path, model, optimizer, device)
        self.best_value = checkpoint.get("best_value", self.best_value)
        self.best_epoch = checkpoint.get("best_epoch", self.best_epoch)
        next_epoch = checkpoint["epoch"] + 1
        LOG.info(
            f"Resumed from epoch {checkpoint['epoch']}, best {self.metric}={self.best_value:.4f}"
        )
        return next_epoch
