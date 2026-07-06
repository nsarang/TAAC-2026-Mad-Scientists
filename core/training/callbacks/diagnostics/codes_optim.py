"""Optimizer and gradient codes: LR, GRAD, OPT_STATE, REINIT."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from core.training.callbacks.diagnostics.base import DiagBase, _parse_context, _try_numeric
from core.training.callbacks.diagnostics.context import (
    EpochContext,
    StepContext,
    _compute_grouped_norms,
    _group_names_from_patterns,
    _match_group,
)


class LrCode(DiagBase):
    """Learning rate snapshot."""

    code = "LR"
    config_key = "lr"
    emit = frozenset({"step"})
    accumulate = frozenset()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        d = ctx.lr_dense
        s = ctx.lr_sparse
        step = ctx.step
        parts = [f"d={d:.6g}"]
        if s is not None:
            parts.append(f"s={s:.6g}")
        if self.writer:
            self.writer.add_scalar("LR/dense", d, step)
            if s is not None:
                self.writer.add_scalar("LR/sparse", s, step)
        return [",".join(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        entry: dict[str, Any] = {}
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                entry[k] = _try_numeric(v)
        ctx = _parse_context(context)
        step = ctx.get("step", 0)
        accum.setdefault("steps", {})[step] = entry


class GradCode(DiagBase):
    """Gradient norm statistics per epoch."""

    code = "GRAD"
    config_key = "grad_norm"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    def __init__(
        self,
        param_groups: list[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._param_groups_raw = param_groups or []
        self._norms: list[float] = []
        group_names = [*_group_names_from_patterns(self._patterns), "other"]
        self._group_norms: dict[str, list[float]] = {name: [] for name in group_names}
        self._step_cursor: int = 0
        self._group_step_cursors: dict[str, int] = dict.fromkeys(group_names, 0)

    @property
    def _patterns(self) -> list[tuple[str, str]]:
        return [(entry["name"], entry["pattern"]) for entry in self._param_groups_raw]

    def step(self, ctx: StepContext, emit: bool = False) -> None:
        """Accumulate per-step data."""
        self._norms.append(ctx.grad_norm)
        if emit and ctx.model is not None and self._patterns:
            norms = _compute_grouped_norms(ctx.model, self._patterns, "grad")
            for g, v in norms.items():
                self._group_norms.setdefault(g, []).append(v)

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        self._norms.clear()
        self._step_cursor = 0
        for lst in self._group_norms.values():
            lst.clear()
        for g in self._group_step_cursors:
            self._group_step_cursors[g] = 0

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code.

        Step-level: stats over norms accumulated since the last flush.
        Epoch-level: stats over the full epoch.
        Idempotent -- never clears state.
        """
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        if not self._norms:
            return []

        if phase == "epoch":
            arr = np.array(self._norms)
            group_slices = {g: vals for g, vals in self._group_norms.items() if vals}
        else:
            window = self._norms[self._step_cursor :]
            if not window:
                return []
            arr = np.array(window)
            group_slices = {}
            for g, vals in self._group_norms.items():
                c = self._group_step_cursors.get(g, 0)
                if c < len(vals):
                    group_slices[g] = vals[c:]

        parts = [f"mean={arr.mean():.4f},max={arr.max():.4f},p99={np.percentile(arr, 99):.4f}"]
        for g, vals in sorted(group_slices.items()):
            parts.append(f"{g}={np.mean(vals):.4f}")

        if isinstance(ctx, (StepContext, EpochContext)):
            pnorms = (
                _compute_grouped_norms(ctx.model, self._patterns)
                if ctx.model is not None and self._patterns
                else None
            )
        else:
            pnorms = None

        if pnorms is not None:
            for g, v in sorted(pnorms.items()):
                parts.append(f"pnorm_{g}={v:.2f}")

        if self.writer:
            if isinstance(ctx, StepContext) and ctx.grad_norm is not None:
                self.writer.add_scalar("GradNorm/total", ctx.grad_norm, tb_step)
            self.writer.add_scalar(f"GradNorm/{tb_tag_prefix}_mean", arr.mean(), tb_step)
            self.writer.add_scalar(f"GradNorm/{tb_tag_prefix}_max", arr.max(), tb_step)
            self.writer.add_scalar(f"GradNorm/{tb_tag_prefix}_p99", np.percentile(arr, 99), tb_step)
            if pnorms is not None:
                for g, v in pnorms.items():
                    self.writer.add_scalar(f"ParamNorm/{tb_tag_prefix}_{g}", v, tb_step)

        return [",".join(parts)]

    def flush(self) -> None:
        """Advance cursors past the current window."""
        self._step_cursor = len(self._norms)
        for g, vals in self._group_norms.items():
            self._group_step_cursors[g] = len(vals)

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        ctx = _parse_context(context)
        entry: dict[str, Any] = {}
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                entry[k] = _try_numeric(v)
        if "epoch" in ctx:
            accum.setdefault("epochs", {})[ctx["epoch"]] = entry
        else:
            accum.setdefault("steps", {})[ctx.get("step", 0)] = entry


class OptStateCode(DiagBase):
    """AdamW optimizer variance accumulator stats per parameter group."""

    code = "OPT_STATE"
    config_key = "opt_state"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset()

    def __init__(
        self,
        param_groups: list[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._param_groups_raw = param_groups or []

    @property
    def _patterns(self) -> list[tuple[str, str]]:
        return [(entry["name"], entry["pattern"]) for entry in self._param_groups_raw]

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        dense_opt: torch.optim.Optimizer | None = ctx.dense_optimizer
        scaler: torch.amp.GradScaler | None = ctx.scaler
        model: nn.Module | None = ctx.model
        if dense_opt is None or model is None:
            return []

        param_to_name: dict[int, str] = {}
        for name, p in model.named_parameters():
            param_to_name[id(p)] = name

        patterns = self._patterns
        group_names = [*_group_names_from_patterns(patterns), "other"]
        group_sq: dict[str, list[float]] = {n: [] for n in group_names}

        for pg in dense_opt.param_groups:
            for p in pg["params"]:
                state = dense_opt.state.get(p)
                if state is None or "exp_avg_sq" not in state:
                    continue
                v = state["exp_avg_sq"]
                name = param_to_name.get(id(p), "")
                group_sq[_match_group(name.lower(), patterns)].append(float(v.mean().item()))

        parts: list[str] = []
        for g in group_names:
            vals = group_sq[g]
            if not vals:
                continue
            arr = np.array(vals)
            parts.append(
                f"{g}:p1={np.percentile(arr, 1):.6g},"
                f"mean={arr.mean():.6g},"
                f"p99={np.percentile(arr, 99):.6g},"
                f"std={arr.std():.6g}"
            )

        if scaler is not None and scaler.is_enabled():
            scale = float(scaler.get_scale())
            parts.append(f"scaler:scale={scale:.1f}")

        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        if self.writer:
            for g in group_names:
                vals = group_sq[g]
                if vals:
                    arr = np.array(vals)
                    self.writer.add_scalar(
                        f"OptState/{tb_tag_prefix}_{g}_mean", arr.mean(), tb_step
                    )

        return [";".join(parts)] if parts else []

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        ctx = _parse_context(context)
        entry: dict[str, Any] = {}
        for block in payload.split(";"):
            if ":" not in block:
                continue
            name, rest = block.split(":", 1)
            d: dict[str, Any] = {}
            for kv in rest.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    d[k] = _try_numeric(v)
            entry[name] = d
        if "epoch" in ctx:
            accum.setdefault("epochs", {})[ctx["epoch"]] = entry
        else:
            accum.setdefault("steps", {})[ctx.get("step", 0)] = entry


class ReinitCode(DiagBase):
    """Embedding reinitialization counts."""

    code = "REINIT"
    config_key = "reinit"
    emit = frozenset({"reinit"})
    accumulate = frozenset()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        reinit = ctx["reinit_count"]
        kept = ctx["kept_count"]
        restored = ctx["restored_optim"]
        return [f"reinit={reinit},kept={kept},restored_optim={restored}"]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        ctx = _parse_context(context)
        epoch = ctx.get("epoch", 0)
        entry: dict[str, Any] = {}
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                entry[k] = _try_numeric(v)
        accum.setdefault("epochs", {})[epoch] = entry
