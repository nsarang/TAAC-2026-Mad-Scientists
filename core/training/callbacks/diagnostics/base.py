"""DiagBase v2, registry, container, and parser for the DIAG| line protocol.

Available diagnostic codes
--------------------------
config_key       code           description
-----------      ----           -----------
attn             ATTN           Cross-attention: recency mass, entropy, pre-softmax L1
din_attn         DIN_ATTN       DIN attention entropy and time-bucket mass per domain
dataset          DATASET        Dataset file inventory
dense_stats      DENSE_STATS    Dense feature distribution stats
domain_geom      DOMAIN_GEOM    Domain encoder pairwise cosine sim and within-domain diversity
done             DONE           Training completion summary
eff_rank         EFF_RANK       Effective rank of representations at hooked pipeline stages
emb_rank         EMB_RANK       Effective rank of embedding tables (weight matrices) via SVD
emb_util         EMB_UTIL       Embedding table utilization via forward-hook hit counting
env              ENV            Environment and config snapshot
gate_stats       GATE_STATS     Output statistics for gate modules
gdcn_cross       GDCN_CROSS     Field-pair cross-interaction strength (first GDCN layer)
gdcn_gate        GDCN_GATE      Per-layer per-field GDCN instance-gate emphasis
grad_norm        GRAD           Gradient norm statistics per epoch
grad_flow        GRAD_FLOW      Per-layer gradient L2 norms via backward hooks
label_dist       LABEL_DIST     Label rate, running positive rate, per-class loss
layer_health     LAYER_HEALTH   Activation stats and block influence per hooked submodule
logit_dist       LOGIT_DIST     Per-class logit percentiles, median gap, overlap fraction
loss_conc        LOSS_CONC      Loss concentration from hardest val samples; optional hard-sample profile
lr               LR             Learning rate snapshot
metrics          METRICS        Train loss + aux losses (step), val metrics (epoch/step_eval), rolling train AUC
model            MODEL          Model architecture and parameter counts
oob              OOB            Out-of-bounds feature stats
opt_state        OPT_STATE      AdamW variance accumulator stats per parameter group
pred_conf        PRED           Prediction confidence distribution
reinit           REINIT         Embedding reinitialization counts
repr_probe       REPR_PROBE     Hidden-state MI (KMeans vs labels) + user/item relevance cosine
sage             SAGE           SAGE feature-group importance (loss + AUC) per epoch
schema           SCHEMA         Feature schema snapshot
seq_lens         SEQ_LENS       Sequence length distribution and truncation rate per domain
timing           TIMING         Forward/backward/collation timing over warmup window
throughput       TPT            Samples-per-second throughput and timing breakdown
tin_stats        TIN_STATS      TIN filter keep ratio, threshold, score separation, domain budget
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from core.training.callbacks.diagnostics.context import (
    EpochContext,
    StepContext,
)

LOG = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base class — v2
# ─────────────────────────────────────────────────────────────────────────────


class DiagBase:
    """Base class for a diagnostic code (v2).

    Subclasses auto-register in ``_registry`` when they define a non-empty
    ``code``. Key v2 changes:

    - ``init_params`` declares extra constructor kwargs beyond ``writer``.
    - ``step()`` receives a ``StepContext``, not raw ``**kw``.
    - ``collect()`` is idempotent — never clears accumulators.
    - ``flush()`` clears accumulators after emission.
    - ``register_hooks`` / ``remove_hooks`` are explicit base-class methods.
    """

    _registry: ClassVar[dict[str, type[DiagBase]]] = {}

    code: str = ""
    config_key: str = ""
    emit: frozenset[str] = frozenset()
    accumulate: frozenset[str] = frozenset()
    init_params: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.code:
            DiagBase._registry[cls.code] = cls

    def __init__(
        self,
        writer: Any = None,
        accumulate_freq: float = 1.0,
        warmup_steps: int = None,
    ) -> None:
        self.writer = writer
        self.warmup_steps = warmup_steps
        self.hooks_active: bool = True
        self._accumulate_freq: float = float(accumulate_freq)
        self._accumulate_debt: float = 0.0
        self._overhead_total: float = 0.0
        self._overhead_steps: int = 0

    def activate_hooks(self, step: int, is_emit: bool) -> None:
        """Decide whether hooks should fire this step.

        ``accumulate_freq`` is a float in (0, 1] controlling the fraction of
        steps that accumulate. 1.0 = every step, 0.1 = ~10% of steps.
        Hooks always activate on emit steps.
        """
        if "always" not in self.accumulate:
            self.hooks_active = False
            return
        if is_emit:
            self.hooks_active = True
            self._accumulate_debt = 0.0
            return
        self._accumulate_debt += self._accumulate_freq
        if self._accumulate_debt >= 1.0:
            self.hooks_active = True
            self._accumulate_debt -= 1.0
        else:
            self.hooks_active = False

    def _timed_hook(self, hook_fn: Any) -> Any:
        """Wrap a forward hook function with overhead accumulation."""
        code_self = self

        def wrapper(mod, args, output):
            if not code_self.hooks_active:
                return
            t0 = time.perf_counter()
            result = hook_fn(mod, args, output)
            code_self._overhead_total += time.perf_counter() - t0
            return result

        return wrapper

    def _timed_backward_hook(self, hook_fn: Any) -> Any:
        """Wrap a backward hook function with overhead accumulation."""
        code_self = self

        def wrapper(mod, grad_input, grad_output):
            if not code_self.hooks_active:
                return
            t0 = time.perf_counter()
            result = hook_fn(mod, grad_input, grad_output)
            code_self._overhead_total += time.perf_counter() - t0
            return result

        return wrapper

    @staticmethod
    def _tb_step(
        phase: str, ctx: StepContext | EpochContext | dict[str, Any] = None
    ) -> tuple[str, int]:
        """Return ``(tag_prefix, global_step)`` for TensorBoard logging."""
        if phase == "epoch":
            if isinstance(ctx, EpochContext):
                return "epoch", ctx.epoch
            if isinstance(ctx, dict):
                return "epoch", ctx.get("epoch", 0)
            return "epoch", 0
        if isinstance(ctx, StepContext):
            return "step", ctx.step
        if isinstance(ctx, dict):
            return "step", ctx.get("step", 0)
        return "step", 0

    def step(self, ctx: StepContext, emit: bool = False) -> None:
        """Per-step accumulation. Override in subclasses.

        Parameters
        ----------
        ctx
            Typed snapshot of the current training step.
        emit
            When True, emission will follow this step — codes may gate
            expensive GPU-to-CPU transfers behind this flag.
        """

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Return payload strings to be packed into DIAG| lines.

        Must be idempotent — never clears state. All clearing goes
        through ``flush()`` or ``epoch_reset()``.
        """
        return []

    def flush(self) -> None:
        """Clear accumulators after emission. Called by container after emit."""

    def epoch_reset(self) -> None:
        """Reset per-epoch state. Called at epoch start."""

    def register_hooks(self, model: nn.Module) -> None:
        """Attach forward/backward hooks to model. No-op by default."""

    def remove_hooks(self) -> None:
        """Detach all hooks. No-op by default."""

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse one payload segment into *accum*."""


REGISTRY = DiagBase._registry


# ─────────────────────────────────────────────────────────────────────────────
# Container — v2
# ─────────────────────────────────────────────────────────────────────────────


class Diagnostics:
    """Structured diagnostics container implementing ``ObserverProtocol`` (v2).

    Parameters
    ----------
    active_codes
        Set of active code config_keys (e.g. ``{"loss", "metrics", "done"}``).
    log_dir
        Directory for TensorBoard event files.
    warmup_steps
        Default warmup step count for codes that don't declare their own.
    log_every_n_steps
        Frequency of per-step DIAG lines.
    code_config
        Per-code configuration dicts, keyed by config_key.
    static_context
        Engine-provided runtime objects that are fixed for the lifetime of the
        training run (e.g. dataset, prepare_batch, device, seq_domains). Codes
        declare which keys they need via ``init_params`` and receive them as
        constructor kwargs.
    """

    def __init__(
        self,
        active_codes: set[str],
        log_dir: str = None,
        warmup_steps: int = 100,
        log_every_n_steps: int = 50,
        max_line_len: int = 12175,
        code_config: dict[str, dict] = None,
        static_context: dict[str, Any] = None,
    ) -> None:
        self.default_warmup_steps = warmup_steps
        self.log_every_n_steps = log_every_n_steps
        self.max_line_len = max_line_len

        if log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer: Any = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        code_configs = code_config or {}
        init_pool: dict[str, Any] = dict(static_context) if static_context else {}

        registered_keys = {cls.config_key for cls in REGISTRY.values()}
        unknown = active_codes - registered_keys
        if unknown:
            raise ValueError(
                f"active_codes contains unregistered diagnostic codes: {sorted(unknown)}. "
                f"Registered: {sorted(registered_keys)}"
            )

        # TPT sorted last: its self-timing must follow all other codes
        self._codes: dict[str, DiagBase] = {}
        for cls in sorted(
            (c for c in REGISTRY.values() if c.config_key in active_codes),
            key=lambda c: c.code == "TPT",
        ):
            extra = {k: init_pool[k] for k in cls.init_params if k in init_pool}
            cfg = code_configs.get(cls.config_key, {})
            self._codes[cls.code] = cls(writer=self.writer, **cfg, **extra)

        # Container-owned state: timing + convergence
        self.train_start_time = 0.0
        self._step_counter = 0
        self.total_samples_seen = 0
        self.best_val_auc = 0.0
        self.best_val_epoch = 0
        self.prev_val_auc = 0.0
        self.epochs_since_improvement = 0

    # ── Manifest ──────────────────────────────────────────────────────────

    def _active_keys(self) -> list[str]:
        return sorted(c.config_key for c in self._codes.values())

    # ── Per-code warmup ───────────────────────────────────────────────────

    def _warmup_steps_for(self, code: DiagBase) -> int:
        return code.warmup_steps if code.warmup_steps is not None else self.default_warmup_steps

    def _is_warmup_done_for(self, code: DiagBase, step: int) -> bool:
        return step > self._warmup_steps_for(code)

    # ── Line emission ─────────────────────────────────────────────────────

    def _emit(self, event: str, context: str, segments: list[tuple[str, str]]) -> None:
        """Pack code:payload segments into the DIAG| line protocol.

        Format: ``DIAG|EVENT|context;;CODE:payload;;CODE:payload...``
        Splits into multiple lines when a segment would exceed `max_line_len`,
        repeating the header prefix on each continuation line.
        """
        if not segments:
            return
        header = f"DIAG|{event}|{context}"
        line = header
        for code, payload in segments:
            seg = f";;{code}:{payload}"
            if len(line) + len(seg) > self.max_line_len and line != header:
                LOG.info(line)
                line = header + seg
            else:
                line += seg
        if line != header:
            LOG.info(line)

    def _collect_and_emit(
        self,
        event: str,
        context: str,
        phase: str,
        ctx: StepContext | EpochContext | dict[str, Any],
    ) -> None:
        """Collect from all codes for the given phase, emit, then flush."""
        segments: list[tuple[str, str]] = []
        emitted_codes: list[DiagBase] = []
        for cn in sorted(self._codes):
            inst = self._codes[cn]
            if phase not in inst.emit:
                continue
            t0 = time.perf_counter()
            payloads = inst.collect(phase, ctx)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            inst._overhead_total += elapsed_ms / 1000
            if payloads:
                emitted_codes.append(inst)
                for payload in payloads:
                    segments.append((cn, payload))
        if event == "STEP":
            overhead_payload = self._format_overhead()
            if overhead_payload:
                segments.append(("OVERHEAD", overhead_payload))
        self._emit(event, context, segments)
        for inst in emitted_codes:
            inst.flush()

    def _format_overhead(self) -> str:
        """Format per-code avg overhead in ms as a flat key=value string."""
        parts: list[str] = []
        for cn in sorted(self._codes):
            inst = self._codes[cn]
            if inst._overhead_steps == 0:
                continue
            avg_ms = (inst._overhead_total / inst._overhead_steps) * 1000.0
            if avg_ms >= 0.01:
                parts.append(f"{inst.config_key}={avg_ms:.2f}")
        return ",".join(parts) if parts else ""

    # ── Preamble (called from train.py, outside Protocol) ─────────────────

    def emit_preamble(
        self,
        *,
        seed: int = 0,
        config: dict[str, Any] = None,
        model: nn.Module = None,
        data_dir: str = "",
        schema_path: str = "",
    ) -> None:
        """Emit preamble DIAG lines for active preamble codes."""
        kw: dict[str, Any] = {
            "seed": seed,
            "config": config or {},
            "model": model,
            "data_dir": data_dir,
            "schema_path": schema_path,
        }
        self._collect_and_emit("PREAMBLE", "init", "preamble", kw)

        if model is not None:
            for inst in self._codes.values():
                inst.register_hooks(model)

    # ── ObserverProtocol lifecycle ────────────────────────────────────────

    def on_train_begin(self) -> None:
        """Handle training start."""
        self.train_start_time = time.perf_counter()
        manifest = ",".join(self._active_keys())
        LOG.info(f"DIAG|CONFIG|{manifest}")

    def on_epoch_begin(self) -> None:
        """Handle epoch start."""
        for inst in self._codes.values():
            inst.epoch_reset()

    def on_step_begin(self) -> None:
        """Handle step start."""
        self._step_counter += 1
        is_emit = self._step_counter % self.log_every_n_steps == 0
        for inst in self._codes.values():
            inst.activate_hooks(self._step_counter, is_emit)

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
        data_time: float = 0.0,
        step_start_time: float = 0.0,
        model: nn.Module = None,
        logits: torch.Tensor = None,
        dense_optimizer: torch.optim.Optimizer = None,
        scaler: torch.amp.GradScaler = None,
    ) -> None:
        """Handle step completion: dispatch to codes and conditionally emit.

        Assumes ``on_step_begin`` already called ``activate_hooks`` to gate
        accumulation for this step. This method then:
        1. Dispatches ``step()`` to codes — "always" codes run every step,
           "warmup" codes only until their warmup boundary passes.
        2. On emit steps (every ``log_every_n_steps``), collects payloads from
           all codes and emits DIAG|STEP lines with per-code overhead timing.
        """
        ctx = StepContext(
            step=step,
            loss=loss,
            aux_losses=aux_losses or {},
            batch=batch,
            grad_norm=grad_norm,
            lr_dense=lr_dense,
            lr_sparse=lr_sparse,
            data_time=data_time,
            fwd_time=fwd_time,
            bwd_time=bwd_time,
            model=model,
            logits=logits,
            dense_optimizer=dense_optimizer,
            scaler=scaler,
            step_start_time=step_start_time,
        )

        # Container bookkeeping
        self.total_samples_seen += ctx.batch_size

        # Determine if this step triggers emission
        is_emit_step = step % self.log_every_n_steps == 0

        # Dispatch step() to codes
        for inst in self._codes.values():
            if "always" in inst.accumulate:
                t0 = time.perf_counter()
                inst.step(ctx, emit=is_emit_step)
                inst._overhead_total += time.perf_counter() - t0
            elif "warmup" in inst.accumulate and not self._is_warmup_done_for(inst, step):
                t0 = time.perf_counter()
                inst.step(ctx, emit=False)
                inst._overhead_total += time.perf_counter() - t0
            inst._overhead_steps += 1

        # Per-code warmup boundary
        for inst in self._codes.values():
            if "warmup" not in inst.accumulate:
                continue
            ws = self._warmup_steps_for(inst)
            if step == ws:
                segments: list[tuple[str, str]] = []
                if "warmup" in inst.emit:
                    for payload in inst.collect("warmup", ctx):
                        segments.append((inst.code, payload))
                if segments:
                    self._emit("WARMUP", f"step={step}", segments)
                    inst.flush()

        # Per-step emission
        if is_emit_step:
            self._collect_and_emit("STEP", f"step={step}", "step", ctx)

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
        """Log step-level validation metrics via codes that emit step_eval."""
        ctx = EpochContext(
            step=step,
            val_auc=val_auc,
            val_logloss=val_logloss,
            val_probs=val_probs,
            val_logits=val_logits,
            val_labels=val_labels,
            val_losses=val_losses,
            val_seq_metadata=val_seq_metadata,
        )
        self._collect_and_emit("STEP_EVAL", f"step={step}", "step_eval", ctx)

    def on_epoch_end(
        self,
        *,
        epoch: int,
        num_epochs: int,
        train_loss: float = None,
        val_auc: float,
        val_logloss: float = 0.0,
        model: nn.Module = None,
        train_time: float = 0.0,
        val_time: float = 0.0,
        val_data_time: float = 0.0,
        val_fwd_time: float = 0.0,
        n_val_batches: int = 0,
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
    ) -> None:
        """Handle epoch completion and emit epoch diagnostics."""
        elapsed = time.perf_counter() - self.train_start_time
        eta_sec = (elapsed / epoch) * (num_epochs - epoch) if epoch > 0 else 0.0

        # Convergence tracking
        delta = val_auc - self.prev_val_auc
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_val_epoch = epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        self.prev_val_auc = val_auc

        ctx = EpochContext(
            epoch=epoch,
            num_epochs=num_epochs,
            train_loss=train_loss,
            val_auc=val_auc,
            val_logloss=val_logloss,
            model=model,
            train_time=train_time,
            val_time=val_time,
            val_data_time=val_data_time,
            val_fwd_time=val_fwd_time,
            n_val_batches=n_val_batches,
            per_domain_aucs=per_domain_aucs,
            calibration=calibration,
            val_probs=val_probs,
            val_logits=val_logits,
            val_labels=val_labels,
            val_losses=val_losses,
            val_seq_metadata=val_seq_metadata,
            sparse_optimizer=sparse_optimizer,
            dense_optimizer=dense_optimizer,
            scaler=scaler,
            oob_stats=oob_stats,
            eta_sec=eta_sec,
            best_val_auc=self.best_val_auc,
            best_val_epoch=self.best_val_epoch,
            delta=delta,
            epochs_since_improvement=self.epochs_since_improvement,
            total_samples_seen=self.total_samples_seen,
        )

        self._collect_and_emit("EPOCH", f"epoch={epoch}", "epoch", ctx)

    def on_reinit(
        self,
        *,
        epoch: int,
        reinit_count: int,
        kept_count: int,
        restored_optim: int,
    ) -> None:
        """Handle sparse embedding reinitialization."""
        ctx: dict[str, Any] = {
            "reinit_count": reinit_count,
            "kept_count": kept_count,
            "restored_optim": restored_optim,
        }
        self._collect_and_emit("EPOCH", f"epoch={epoch}", "reinit", ctx)

    def on_train_end(
        self,
        *,
        ckpt_path: str = None,
        early_stopped: bool = False,
        early_stop_epoch: int = 0,
        model: nn.Module = None,
    ) -> None:
        """Handle training completion and emit done diagnostics."""
        wall = time.perf_counter() - self.train_start_time
        ctx: dict[str, Any] = {
            "best_auc": self.best_val_auc,
            "best_epoch": self.best_val_epoch,
            "wall_sec": wall,
            "samples": self.total_samples_seen,
            "early_stop": early_stopped,
            "model": model,
        }
        self._collect_and_emit("DONE", "", "done", ctx)

        for inst in self._codes.values():
            inst.remove_hooks()

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────


def _try_numeric(s: str) -> int | float | str:
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _parse_context(ctx: str) -> dict[str, Any]:
    """Parse ``key=val,key=val`` context strings."""
    result: dict[str, Any] = {}
    for part in ctx.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = _try_numeric(v)
    return result


_DIAG_RE = re.compile(r"(DIAG\|.+)$")


def parse_log(raw: str = None, *, path: str = None) -> list[dict[str, Any]]:
    """Parse ``DIAG|`` lines from raw log text into structured dicts.

    Handles both bare ``DIAG|...`` lines and lines prefixed by a logging
    framework (e.g. ``HH:MM:SS module INFO  DIAG|...``).

    Each ``DIAG|CONFIG|...`` line marks the start of a new experiment. The
    returned list contains one dict per experiment found. Lines before
    the first CONFIG (e.g. preamble codes) are merged into that experiment.

    Parameters
    ----------
    raw
        Raw log text. Mutually exclusive with `path`.
    path
        File path to read log text from. Mutually exclusive with `raw`.

    Returns
    -------
    list[dict[str, Any]]
        One dict per experiment, keyed by code name (uppercase).
    """
    if raw is None and path is None:
        raise ValueError("Provide either `raw` or `path`")
    if raw is not None and path is not None:
        raise ValueError("`raw` and `path` are mutually exclusive")
    if path is not None:
        with open(path, encoding="utf-8") as f:
            raw = f.read()

    experiments: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    for line in raw.splitlines():
        m = _DIAG_RE.search(line)
        if m is None:
            continue
        diag_str = m.group(1)

        parts = diag_str.split("|", 2)
        if len(parts) < 3:
            continue

        event = parts[1]
        rest = parts[2]

        if event == "PREAMBLE":
            if results and "_config" in results:
                experiments.append(results)
                results = {}

        if event == "CONFIG":
            results["_config"] = set(rest.split(","))
            continue

        # Split context from first segment
        first_sep = rest.find(";;")
        if first_sep == -1:
            continue
        context = rest[:first_sep]
        segments_str = rest[first_sep + 2 :]

        for segment in segments_str.split(";;"):
            colon_pos = segment.find(":")
            if colon_pos == -1:
                continue
            code = segment[:colon_pos]
            payload = segment[colon_pos + 1 :]

            if code == "OVERHEAD":
                ctx = _parse_context(context)
                step = ctx.get("step", 0)
                entry: dict[str, float] = {}
                for kv in payload.split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        entry[k] = _try_numeric(v)
                results.setdefault("OVERHEAD", {}).setdefault("steps", {})[step] = entry
                continue

            cls = REGISTRY.get(code)
            if cls is None:
                continue

            accum = results.setdefault(code, {})
            cls.parse(payload, context, accum)

    if results:
        experiments.append(results)

    return experiments
