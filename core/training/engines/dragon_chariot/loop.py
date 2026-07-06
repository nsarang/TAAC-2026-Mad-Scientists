"""Training loop for DragonChariot: dual-optimizer, step-level
validation, sparse reinit, diagnostics hooks, and early stopping.

Adapted from core/training/loop.py (V1) with dead weight removed and
interfaces adapted for DragonChariot's dict-based batch format.
"""

from __future__ import annotations

import inspect
import logging
import time
from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms.join import Join
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config.loader import instantiate_config
from core.data.loader import batch_to_device
from core.evaluation.metrics import binary_auc, sigmoid
from core.models.modules.primitives import RMSNorm
from core.training.callbacks.diagnostics.codes_eval import compute_calibration
from core.training.callbacks.protocol import ObserverProtocol
from core.training.checkpoint import CheckpointManager
from core.training.early_stopping import EarlyStopping
from core.training.ema import ModelEMA
from core.utils.distributed import broadcast_bool, is_distributed

LOG = logging.getLogger(__name__)


def build_optimizers(
    model: nn.Module,
    dense_optimizer: dict[str, Any],
    sparse_optimizer: dict[str, Any],
    dense_param_overrides: dict[str, dict[str, Any]] = None,
    exclude_wd_on_bias_norm: bool = False,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Create dense + optional sparse optimizers with layered param grouping.

    The grouping mechanisms compose in order:

    1. ``dense_param_overrides`` — longest-prefix match assigns each param
       to a named bucket with per-bucket optimizer kwargs.
    2. ``exclude_wd_on_bias_norm`` — splits each bucket into decay/no-decay
       sub-groups (bias and norm weights get weight_decay=0).
    The resulting param_groups list is the cartesian product of these two
    axes, pruned to non-empty groups.

    Parameters
    ----------
    model
        The model whose parameters are optimized.
    dense_optimizer
        Dict with ``_cls`` (dotted class path) and ``_init`` (lr, betas, etc).
    sparse_optimizer
        Dict with ``_cls`` and ``_init`` for sparse (embedding) parameters.
    dense_param_overrides
        Per-prefix overrides for dense param groups. Keys are parameter name
        prefixes; values are dicts of optimizer kwargs (e.g. ``weight_decay``).
        Longest matching prefix wins. When ``None`` or empty, all dense params
        share a single group with default kwargs.
    exclude_wd_on_bias_norm
        When True, bias parameters and LayerNorm/RMSNorm weights get
        ``weight_decay=0`` regardless of the group default.
    """
    if sparse_params := model.get_sparse_params():
        sparse_ptrs = {p.data_ptr() for p in sparse_params}
        LOG.info(
            f"Sparse: {len(sparse_params)} tensors, "
            f"{sum(p.numel() for p in sparse_params):,} params"
        )
        sparse_opt: torch.optim.Optimizer = instantiate_config(
            sparse_optimizer, params=sparse_params
        )
    else:
        sparse_opt = None
        sparse_ptrs: set[int] = set()
        LOG.info("Sparse: 0 tensors, 0 params — no sparse optimizer")

    overrides = dense_param_overrides or {}
    sorted_prefixes = sorted(overrides.keys(), key=len, reverse=True)

    groups: dict[str, list[nn.Parameter]] = {"__default__": []}
    for prefix in sorted_prefixes:
        groups[prefix] = []

    for name, param in model.named_parameters():
        if param.data_ptr() in sparse_ptrs:
            continue
        matched = False
        for prefix in sorted_prefixes:
            if name.startswith(prefix):
                groups[prefix].append(param)
                matched = True
                break
        if not matched:
            groups["__default__"].append(param)

    no_decay_ptrs: set[int] = set()
    if exclude_wd_on_bias_norm:
        for module in model.modules():
            if isinstance(module, (nn.LayerNorm, RMSNorm)):
                for p in module.parameters():
                    no_decay_ptrs.add(p.data_ptr())
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                no_decay_ptrs.add(param.data_ptr())

    dense_kwargs = dict(dense_optimizer["_init"])
    param_groups = []
    for key, params in groups.items():
        if not params:
            continue
        base_kwargs = {**dense_kwargs}
        if key != "__default__":
            base_kwargs.update(overrides[key])

        buckets: list[tuple[list[nn.Parameter], dict[str, Any]]] = []
        if no_decay_ptrs:
            decay = [p for p in params if p.data_ptr() not in no_decay_ptrs]
            no_decay = [p for p in params if p.data_ptr() in no_decay_ptrs]
            if decay:
                buckets.append((decay, {**base_kwargs}))
            if no_decay:
                buckets.append((no_decay, {**base_kwargs, "weight_decay": 0.0}))
            n_nd = sum(p.numel() for p in no_decay)
        else:
            buckets.append((params, {**base_kwargs}))
            n_nd = 0

        for bucket_params, bucket_kwargs in buckets:
            param_groups.append({"params": bucket_params, **bucket_kwargs})

        n_params = sum(p.numel() for p in params)
        nd_suffix = f" ({n_nd:,} no-decay)" if n_nd else ""
        LOG.debug(
            "Dense group '%s': %d tensors, %s params%s, %s",
            key,
            len(params),
            f"{n_params:,}",
            nd_suffix,
            base_kwargs,
        )

    dense_opt = instantiate_config(dense_optimizer, params=param_groups)

    n_dense = sum(p.numel() for g in dense_opt.param_groups for p in g["params"])
    LOG.info(
        "Optimizers: dense=%s (%d params, %d groups), sparse=%s",
        type(dense_opt).__name__,
        n_dense,
        len(dense_opt.param_groups),
        type(sparse_opt).__name__ if sparse_opt else "none",
    )
    return dense_opt, sparse_opt


def rebuild_sparse_optimizer(
    model: nn.Module,
    sparse_optimizer: torch.optim.Optimizer,
    sparse_optimizer_cfg: dict[str, Any],
    reinit_ptrs: set[int] = None,
) -> tuple[torch.optim.Optimizer, int]:
    """Rebuild sparse optimizer, preserving state for non-reinitialized params.

    Used both at initial construction (when switching optimizer class) and
    after high-cardinality embedding reinitialization.

    Parameters
    ----------
    model
        Model with ``get_sparse_params()`` method.
    sparse_optimizer
        Current sparse optimizer whose state to partially preserve.
    sparse_optimizer_cfg
        Config dict for instantiating the new optimizer.
    reinit_ptrs
        Set of ``data_ptr()`` for parameters that were reinitialized.
        State for these is NOT preserved. Pass None or empty set to
        preserve all state (useful for optimizer class migration).

    Returns
    -------
    tuple
        ``(new_optimizer, restored_count)``
    """
    reinit_ptrs = reinit_ptrs or set()

    old_state: dict[int, Any] = {}
    for group in sparse_optimizer.param_groups:
        for p in group["params"]:
            if p in sparse_optimizer.state:
                old_state[p.data_ptr()] = sparse_optimizer.state[p]

    sparse_params = model.get_sparse_params()
    new_opt = instantiate_config(sparse_optimizer_cfg, params=sparse_params)

    restored = 0
    for p in sparse_params:
        if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
            new_opt.state[p] = old_state[p.data_ptr()]
            restored += 1

    LOG.info(
        "Rebuilt sparse optimizer: %d params, %d states restored",
        len(sparse_params),
        restored,
    )
    return new_opt, restored


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_schedule: dict[str, Any],
    steps_per_epoch: int,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build LR scheduler as a flat SequentialLR composition.

    Composes up to 3 phases depending on schedule type:
    - ``one_cycle``: standalone OneCycleLR (total_steps + 1 to avoid
      boundary error on final .step() call).
    - ``cosine``: [warmup] → CosineAnnealingLR.
    - ``constant_cooldown``: [warmup] → ConstantLR → linear cooldown.
    - ``constant``: [warmup] only (returns None if no warmup).

    All multi-phase schedules are flattened into a single SequentialLR to
    avoid nested SequentialLR (PyTorch compat issue with nested .step(0)).

    Parameters
    ----------
    optimizer
        The optimizer whose LR groups are scheduled.
    lr_schedule
        Dict with ``type``, ``warmup_steps``, and schedule-specific keys.
    steps_per_epoch
        Number of training steps per epoch.
    epochs
        Total number of training epochs.
    """
    schedule = lr_schedule["type"]
    warmup_steps = lr_schedule["warmup_steps"]
    total_steps = steps_per_epoch * epochs

    if schedule == "constant" and warmup_steps <= 0:
        return None

    if schedule == "one_cycle":
        # +1: OneCycleLR raises on the final .step() call at the boundary
        return OneCycleLR(
            optimizer,
            total_steps=total_steps + 1,
            **lr_schedule["kwargs"],
        )

    # Build a flat sequence of schedulers to avoid nesting SequentialLR
    # inside SequentialLR (PyTorch compat: nested .step(0) call fails).
    schedulers = []
    milestones = []
    cursor = 0

    if warmup_steps > 0:
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        )
        cursor += warmup_steps
        milestones.append(cursor)

    if schedule == "cosine":
        base_lr = optimizer.param_groups[0]["lr"]
        eta_min = base_lr * lr_schedule["cosine_min_lr_ratio"]
        schedulers.append(
            CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=eta_min,
            )
        )
    elif schedule == "constant_cooldown":
        post_warmup = max(1, total_steps - warmup_steps)
        cooldown_steps = max(1, int(post_warmup * lr_schedule["cooldown_fraction"]))
        constant_steps = post_warmup - cooldown_steps
        schedulers.append(ConstantLR(optimizer, factor=1.0, total_iters=constant_steps))
        cursor += constant_steps
        milestones.append(cursor)
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=max(lr_schedule["min_lr_ratio"], 1e-6),
                total_iters=cooldown_steps,
            )
        )
    elif schedule == "constant":
        if not schedulers:
            return None
    else:
        raise ValueError(f"Unknown lr_schedule type: {schedule}")

    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones[: len(schedulers) - 1],
    )


class Trainer:
    """Training loop with dual optimizers, early stopping, and observer callbacks.

    Parameters
    ----------
    model
        DragonChariot model. Must expose ``get_sparse_params()`` /
        ``get_dense_params()``.
    train_loader, valid_loader
        Data iterators. Batches are dicts of tensors with a ``'label'`` key.
    loss_fn
        ``nn.Module`` called as ``loss_fn(logits, labels)``.
    dense_optimizer
        Dict with ``_cls`` (dotted class path) and ``_init`` (optimizer kwargs).
    sparse_optimizer
        Same shape as `dense_optimizer`, for embedding parameters.
    dense_param_overrides
        Per-prefix overrides for dense param groups.
    lr_schedule
        Dict with ``type`` and ``warmup_steps``.
    reinit
        Dict with ``after_epoch`` and ``cardinality_threshold``.
    observers
        List of ``ObserverProtocol`` implementations.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        loss_fn: nn.Module,
        *,
        dense_optimizer: dict[str, Any],
        sparse_optimizer: dict[str, Any],
        dense_param_overrides: dict[str, dict[str, Any]] = None,
        lr_schedule: dict[str, Any],
        epochs: int = 1,
        grad_clip_norm: float = 1.0,
        device: str | torch.device = "cuda",
        mid_epoch_evals: int = 0,
        early_stopping: EarlyStopping = None,
        reinit: dict[str, Any],
        amp_dtype: torch.dtype = None,
        checkpoint_mgr: CheckpointManager = None,
        observers: list[ObserverProtocol] = None,
        progress_bar: bool = True,
        exclude_wd_on_bias_norm: bool = False,
        ema: ModelEMA = None,
        pretrain_phase: dict[str, Any] = None,
        pretrain_loader: DataLoader = None,
        local_rank: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
        val_loader_swapper: Callable[[int], DataLoader | None] = None,
    ) -> None:
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = loss_fn
        self._loss_accepts_weight = "sample_weight" in inspect.signature(loss_fn.forward).parameters
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.device = torch.device(device)
        self.mid_epoch_evals = mid_epoch_evals
        self.early_stopping = early_stopping
        self.sparse_optimizer_cfg = sparse_optimizer
        self.checkpoint_mgr = checkpoint_mgr
        self._observers = observers or []

        self.reinit_cfg = reinit
        self.pretrain_phase_cfg = pretrain_phase
        self.pretrain_loader = pretrain_loader
        self._val_loader_swapper = val_loader_swapper
        self._val_loader_swapped = False
        self._total_step = 0
        self._lowcard_snapshot: dict[str, torch.Tensor] = {}

        self.dense_optimizer, self.sparse_optimizer = build_optimizers(
            model,
            dense_optimizer,
            sparse_optimizer,
            dense_param_overrides=dense_param_overrides,
            exclude_wd_on_bias_norm=exclude_wd_on_bias_norm,
        )
        if self.sparse_optimizer is not None and hasattr(self.sparse_optimizer, "register_hooks"):
            self.sparse_optimizer.register_hooks(model)

        self.scheduler = (
            build_scheduler(self.dense_optimizer, lr_schedule, len(train_loader), epochs)
            if lr_schedule is not None
            else None
        )
        self.ema = ema

        self.progress_bar = progress_bar

        self.amp_dtype = amp_dtype
        self.scaler = torch.amp.GradScaler(
            enabled=(amp_dtype in (torch.float16, torch.bfloat16) and self.device.type == "cuda")
        )

    def train(self) -> dict[str, Any]:
        """Run the full training loop. Returns ``{'best_auc': float, 'history': list}``."""
        self.model.train()
        total_step = 0
        best_auc = 0.0
        history: list[dict[str, Any]] = []

        for obs in self._observers:
            obs.on_train_begin()

        for epoch in range(1, self.epochs + 1):
            for loader in (self.train_loader, self.pretrain_loader):
                if loader is not None and hasattr(loader.dataset, "set_epoch"):
                    loader.dataset.set_epoch(epoch - 1)
            self.maybe_run_pretrain_phase(epoch)
            for obs in self._observers:
                obs.on_epoch_begin()

            pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                dynamic_ncols=True,
                disable=not self.progress_bar or self.global_rank != 0,
            )
            loss_sum = 0.0
            step = -1
            train_start = time.time()

            join_ctx = Join([self.model]) if is_distributed() else nullcontext()
            data_start = time.perf_counter()
            with join_ctx:
                for step, batch in pbar:
                    batch = batch_to_device(batch, self.device)
                    data_time = time.perf_counter() - data_start

                    for obs in self._observers:
                        obs.on_step_begin()

                    loss, grad_norm, fwd_time, bwd_time, logits, aux_losses = self.train_step(batch)
                    total_step += 1
                    self._total_step = total_step

                    loss_sum += loss

                    observer_loss = float(np.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))
                    if not np.isfinite(loss):
                        LOG.warning(
                            "Non-finite train loss encountered; reporting 0.0 to observers."
                        )
                    observer_aux_losses: dict[str, float] = {}
                    for key, value in aux_losses.items():
                        if not np.isfinite(value):
                            LOG.warning(
                                "Non-finite aux loss '%s' encountered; reporting 0.0 to observers.",
                                key,
                            )
                        observer_aux_losses[key] = float(
                            np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                        )

                    step_kw = dict(
                        step=total_step,
                        loss=observer_loss,
                        aux_losses=observer_aux_losses,
                        batch=batch,
                        grad_norm=grad_norm,
                        lr_dense=self.dense_optimizer.param_groups[0]["lr"],
                        lr_sparse=(
                            self.sparse_optimizer.param_groups[0]["lr"]
                            if self.sparse_optimizer is not None
                            else None
                        ),
                        fwd_time=fwd_time,
                        bwd_time=bwd_time,
                        data_time=data_time,
                        step_start_time=data_start,
                        model=self.model,
                        logits=logits,
                        dense_optimizer=self.dense_optimizer,
                        scaler=self.scaler,
                    )
                    for obs in self._observers:
                        obs.on_step_end(**step_kw)

                    pbar.set_postfix({"loss": f"{loss:.4f}"})
                    self.maybe_step_eval(step + 1, total_step, epoch)
                    data_start = time.perf_counter()

            train_time = time.time() - train_start
            avg_loss = loss_sum / max(1, step + 1)
            LOG.info(f"Epoch {epoch}, avg loss: {avg_loss:.4f}")

            self._maybe_activate_cached_val_loader(epoch)

            # All ranks evaluate (redundant work) so no rank idles at the barrier.
            # TODO (nsarang): add distributed val aggregation to avoid redundant eval.
            val_start = time.time()
            (
                val_auc,
                val_logloss,
                val_probs,
                val_logits,
                val_labels,
                val_losses,
                val_data_time,
                val_fwd_time,
                n_val_batches,
            ) = self.evaluate()
            val_time = time.time() - val_start
            self.model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.global_rank == 0:
                LOG.info(
                    f"Epoch {epoch}/{self.epochs} | AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}"
                )

            if self.early_stopping and self.global_rank == 0:
                self.early_stopping(val_auc, self.model)

            if self.checkpoint_mgr and self.global_rank == 0:
                metrics = {"auc": val_auc, "logloss": val_logloss}
                runtime = {"step": total_step, "epoch": epoch, **metrics}
                self.checkpoint_mgr.save(
                    epoch,
                    self.model,
                    self.dense_optimizer,
                    metrics,
                    runtime=runtime,
                    ema=self.ema,
                )
            if is_distributed():
                dist.barrier(device_ids=[self.local_rank])

            if self.global_rank == 0 and np.isfinite(val_auc):
                best_auc = max(best_auc, val_auc)

            epoch_kw = dict(
                epoch=epoch,
                num_epochs=self.epochs,
                train_loss=avg_loss,
                val_auc=val_auc,
                val_logloss=val_logloss,
                model=self.model,
                train_time=train_time,
                val_time=val_time,
                val_data_time=val_data_time,
                val_fwd_time=val_fwd_time,
                n_val_batches=n_val_batches,
                calibration=compute_calibration(val_probs, val_labels)
                if val_probs is not None
                else None,
                val_probs=val_probs,
                val_logits=val_logits,
                val_labels=val_labels,
                val_losses=val_losses,
                sparse_optimizer=self.sparse_optimizer,
                dense_optimizer=self.dense_optimizer,
                scaler=self.scaler,
            )
            for obs in self._observers:
                obs.on_epoch_end(**epoch_kw)

            # Rank 0 runs diagnostics (repr_probe KMeans etc.) while rank 1 has no
            # observers and would race ahead to the EMA broadcast. Sync here.
            if is_distributed():
                dist.barrier(device_ids=[self.local_rank])

            if self.ema is not None:
                self.ema.notify_epoch_end(epoch)

            should_stop = bool(self.early_stopping and self.early_stopping.early_stop)
            should_stop = broadcast_bool(should_stop, src=0)
            if should_stop:
                LOG.info(f"Early stopping at epoch {epoch}")
                return self.finish(
                    early_stopped=True, early_stop_epoch=epoch, best_auc=best_auc, history=history
                )

            if self.global_rank == 0:
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "val_auc": val_auc,
                        "val_logloss": val_logloss,
                    }
                )

            self.maybe_reinit_sparse(epoch)

        return self.finish(
            early_stopped=False, early_stop_epoch=0, best_auc=best_auc, history=history
        )

    def train_step(
        self,
        batch: dict[str, Any],
    ) -> tuple[float, float, float, float, torch.Tensor, dict[str, float]]:
        """Execute one training step: forward, backward, optimizer step.

        Expects batch already on device (caller handles batch_to_device).
        """
        labels = batch["label"].float()

        self.dense_optimizer.zero_grad(set_to_none=True)
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad(set_to_none=True)

        # Sync TBE LR before forward (TBE uses LR set at forward-time for backward)
        if self.sparse_optimizer is not None:
            self.model.update_learning_rate(self.sparse_optimizer.param_groups[0]["lr"])
        else:
            self.model.update_learning_rate()

        use_cuda_events = torch.cuda.is_available()
        fwd_start = fwd_end = bwd_end = None
        if use_cuda_events:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            fwd_start.record()
        else:
            t_fwd_start = time.perf_counter()

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        ):
            logits, aux = self.model(batch, labels=labels)
            if self._loss_accepts_weight:
                loss = self.loss_fn(logits, labels, sample_weight=batch.get("sample_weight"))
            else:
                loss = self.loss_fn(logits, labels)
            for aux_loss in aux.values():
                loss = loss + aux_loss

        if use_cuda_events:
            fwd_end.record()
        else:
            t_fwd_end = time.perf_counter()

        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.dense_optimizer)
        if self.sparse_optimizer is not None:
            self.scaler.unscale_(self.sparse_optimizer)
        clip_params = [
            p for p in self.model.parameters() if p.grad is not None and not p.grad.is_sparse
        ]
        grad_norm = float(nn.utils.clip_grad_norm_(clip_params, self.grad_clip_norm, foreach=False))

        self.scaler.step(self.dense_optimizer)
        if self.sparse_optimizer is not None:
            self.scaler.step(self.sparse_optimizer)
        old_scale = self.scaler.get_scale()
        self.scaler.update()
        optimizer_stepped = self.scaler.get_scale() >= old_scale

        if use_cuda_events:
            bwd_end.record()

        fwd_time: float = 0.0
        bwd_time: float = 0.0
        if use_cuda_events:
            torch.cuda.synchronize()
            fwd_time = fwd_start.elapsed_time(fwd_end) / 1000
            bwd_time = fwd_end.elapsed_time(bwd_end) / 1000
        else:
            t_bwd_end = time.perf_counter()
            fwd_time = t_fwd_end - t_fwd_start
            bwd_time = t_bwd_end - t_fwd_end

        if self.scheduler is not None and optimizer_stepped:
            self.scheduler.step()

        if self.ema is not None and optimizer_stepped:
            self.ema.update()

        aux_losses = {k: float(v) for k, v in aux.items()}
        return loss.item(), grad_norm, fwd_time, bwd_time, logits.detach(), aux_losses

    def _maybe_activate_cached_val_loader(self, epoch: int) -> None:
        if self._val_loader_swapper is None or self._val_loader_swapped:
            return
        try:
            maybe_loader = self._val_loader_swapper(epoch)
        except Exception:
            LOG.exception("Failed to activate cached validation loader; keeping original loader")
            self._val_loader_swapper = None
            return
        if maybe_loader is not None:
            self.valid_loader = maybe_loader
            self._val_loader_swapped = True

    @torch.no_grad()
    def evaluate(
        self,
    ) -> tuple[
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        int,
    ]:
        """Run validation and return metrics + arrays."""
        eval_model = (
            self.model.module if is_distributed() and hasattr(self.model, "module") else self.model
        )

        if len(self.valid_loader) == 0:
            empty = np.array([], dtype=np.float32)
            return (float("nan"), float("inf"), empty, empty, empty, empty, 0.0, 0.0, 0)

        if self.ema is not None:
            self.ema.apply()
        self.model.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        val_data_time = 0.0
        val_fwd_time = 0.0

        data_start = time.perf_counter()
        for batch in self.valid_loader:
            val_data_time += time.perf_counter() - data_start
            batch = batch_to_device(batch, self.device)
            labels = batch["label"].float()
            fwd_start = time.perf_counter()
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                logits, _ = eval_model(batch)
            val_fwd_time += time.perf_counter() - fwd_start
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            data_start = time.perf_counter()

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        logits_np = logits_cat.float().numpy()
        probs = sigmoid(logits_np)
        labels_np = labels_cat.float().numpy()

        nan_mask = torch.isnan(logits_cat)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            LOG.warning(f"{n_nan}/{len(probs)} predictions are NaN, filtering")
            valid = ~nan_mask
            valid_np = valid.numpy()
            logits_cat = logits_cat[valid]
            labels_cat = labels_cat[valid]
            logits_np = logits_np[valid_np]
            probs = probs[valid_np]
            labels_np = labels_np[valid_np]

        # Full-set per-sample losses (for diagnostics — observers see all samples)
        per_sample_losses = (
            F.binary_cross_entropy_with_logits(
                logits_cat, labels_cat.float(), reduction="none"
            ).numpy()
            if len(logits_cat) > 0
            else np.array([], dtype=np.float32)
        )
        metric_probs = probs
        metric_labels = labels_np
        metric_losses = per_sample_losses

        auc = binary_auc(metric_labels, metric_probs)
        logloss = float(metric_losses.mean()) if len(metric_losses) > 0 else float("inf")
        if self.ema is not None:
            self.ema.restore()
        return (
            auc,
            logloss,
            probs,
            logits_np,
            labels_np,
            per_sample_losses,
            val_data_time,
            val_fwd_time,
            len(all_logits),
        )

    def maybe_step_eval(self, step_in_epoch: int, total_step: int, epoch: int) -> None:
        """Run step-level validation if configured."""
        if self.global_rank != 0:
            return
        if self.mid_epoch_evals <= 0:
            return
        steps_per_epoch = len(self.train_loader)
        interval = steps_per_epoch // (self.mid_epoch_evals + 1)
        if interval <= 0 or step_in_epoch % interval != 0:
            return
        if step_in_epoch + interval > steps_per_epoch:
            return

        (
            val_auc,
            val_logloss,
            val_probs,
            val_logits,
            val_labels,
            val_losses,
            *_,
        ) = self.evaluate()
        self.model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        LOG.info(f"Step {total_step} | AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")

        for obs in self._observers:
            obs.on_step_eval(
                step=total_step,
                val_auc=val_auc,
                val_logloss=val_logloss,
                val_probs=val_probs,
                val_logits=val_logits,
                val_labels=val_labels,
                val_losses=val_losses,
            )

    def maybe_reinit_sparse(self, epoch: int) -> None:
        """Reinitialize high-cardinality embeddings and reset optimizer state."""
        if self.sparse_optimizer is None:
            return
        if epoch < self.reinit_cfg["after_epoch"]:
            return
        if self.reinit_cfg["cardinality_threshold"] <= 0:
            return

        reinit_ptrs = self.model.reinit_high_cardinality_params(
            self.reinit_cfg["cardinality_threshold"]
        )

        # Restore snapshotted low-cardinality seq embeddings
        snapshot_ptrs: set[int] = set()
        if self._lowcard_snapshot:
            snapshot_ptrs = self.model.restore_emb_snapshot(self._lowcard_snapshot)
            LOG.info(f"Restored {len(snapshot_ptrs)} seq embeddings from snapshot")

        # Remove old hooks before rebuilding
        if self.sparse_optimizer is not None and hasattr(self.sparse_optimizer, "remove_hooks"):
            self.sparse_optimizer.remove_hooks()

        # Rebuild sparse optimizer, preserving state for kept params
        self.sparse_optimizer, restored = rebuild_sparse_optimizer(
            self.model,
            self.sparse_optimizer,
            self.sparse_optimizer_cfg,
            reinit_ptrs=reinit_ptrs,
        )
        if self.sparse_optimizer is not None and hasattr(self.sparse_optimizer, "register_hooks"):
            self.sparse_optimizer.register_hooks(self.model)

        reinit_count = len(reinit_ptrs)
        kept_count = len(self.model.get_sparse_params()) - reinit_count
        LOG.info(
            f"Reinit: {reinit_count} tables reinitialized, "
            f"{kept_count} kept, {restored} optimizer states restored"
        )
        for obs in self._observers:
            obs.on_reinit(
                epoch=epoch,
                reinit_count=reinit_count,
                kept_count=kept_count,
                restored_optim=restored,
            )

    def maybe_run_pretrain_phase(self, epoch: int) -> None:
        """Run pretext-only training on seq embeddings at epoch boundary.

        Lifecycle:
        1. Freeze all params except seq embeddings + pretext head.
        2. Run N steps of forward/backward using only the pretext aux loss.
        3. Unfreeze all params.
        4. After first pretrain completes, snapshot low-cardinality seq
           embeddings (vocab <= threshold) for later restoration after reinit.
        """
        cfg = self.pretrain_phase_cfg
        if cfg is None or self.pretrain_loader is None:
            return

        if cfg.get("epochs") is not None and epoch > cfg["epochs"]:
            return

        steps = max(1, cfg["steps"] // self.world_size)
        LOG.info(f"Pretrain phase: {steps} steps on seq embeddings + pretext head")

        trainable_ptrs = self.model.pretext_trainable_params()
        frozen = []
        for p in self.model.parameters():
            if p.data_ptr() not in trainable_ptrs:
                p.requires_grad_(False)
                frozen.append(p)

        self.model.train()
        steps_run = 0
        last_loss = 0.0
        join_ctx = Join([self.model]) if is_distributed() else nullcontext()
        with join_ctx:
            for i, batch in zip(range(steps), self.pretrain_loader):
                self.dense_optimizer.zero_grad(set_to_none=True)
                if self.sparse_optimizer is not None:
                    self.sparse_optimizer.zero_grad(set_to_none=True)

                batch = batch_to_device(batch, self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_dtype is not None,
                ):
                    _, aux = self.model(batch)
                    if not aux:
                        break
                    loss = sum(aux.values())

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.dense_optimizer)
                # Sparse params are frozen during pretrain — skip their optimizer
                sparse_active = self.sparse_optimizer is not None and any(
                    p.requires_grad for g in self.sparse_optimizer.param_groups for p in g["params"]
                )
                if sparse_active:
                    self.scaler.unscale_(self.sparse_optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip_norm,
                )
                self.scaler.step(self.dense_optimizer)
                if sparse_active:
                    self.scaler.step(self.sparse_optimizer)
                self.scaler.update()
                steps_run += 1
                last_loss = loss.item()

        # Unfreeze
        for p in frozen:
            p.requires_grad_(True)

        LOG.info(f"Pretrain phase complete, {steps_run} steps, loss: {last_loss:.4f}")

        # Snapshot low-cardinality seq embeddings after first pretrain completes
        snapshot_threshold = self.reinit_cfg["snapshot_vocab_threshold"]
        if snapshot_threshold > 0 and not self._lowcard_snapshot:
            self._lowcard_snapshot = self.model.snapshot_low_cardinality_embs(snapshot_threshold)
            LOG.info(
                f"Snapshot {len(self._lowcard_snapshot)} low-cardinality "
                f"seq embeddings (vocab <= {snapshot_threshold})"
            )

    def finish(
        self,
        *,
        early_stopped: bool,
        early_stop_epoch: int,
        best_auc: float,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Finalize training: notify observers and return results dict."""
        ckpt_path: str = None
        if self.checkpoint_mgr and self.checkpoint_mgr.best_dir:
            ckpt_path = str(self.checkpoint_mgr.best_dir)
        elif self.early_stopping and self.early_stopping.checkpoint_path:
            ckpt_path = self.early_stopping.checkpoint_path
        for obs in self._observers:
            obs.on_train_end(
                ckpt_path=ckpt_path,
                early_stopped=early_stopped,
                early_stop_epoch=early_stop_epoch,
                model=self.model,
            )
        if self.checkpoint_mgr:
            best_auc = max(best_auc, self.checkpoint_mgr.best_value)
        return {"best_auc": best_auc, "history": history}
