"""DIN_ATTN diagnostic: per-domain attention entropy and time-bucket mass."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from core.models.modules.din import TargetAwareDINHead
from core.models.modules.segment_ops import segment_sum
from core.training.callbacks.diagnostics.base import DiagBase, _parse_context, _try_numeric
from core.training.callbacks.diagnostics.codes_model import _find_hook_targets, _flat_kv, _tag
from core.training.callbacks.diagnostics.context import EpochContext, StepContext


class DINAttnCode(DiagBase):
    r"""Per-domain DIN attention diagnostics.

    Captures attention weights stashed by ``TargetAwareDINHead`` (requires
    ``capture_diag=True`` on the module, set automatically on hook registration).

    Emitted keys per matched domain ``d``:

    - ``d_entropy_mean`` / ``d_entropy_std`` — per-sample attention entropy
    - ``d_{window}_mass`` — mean attention mass in each configured time window
    - ``d_pad_mass`` — attention mass on padding positions (padded path only)

    Category / campaign / target match analysis requires threading raw feature
    IDs through the model forward and is not yet implemented.

    Config example::

        din_attn:
          accumulate_freq: 0.2
          targets:
            - "^din_heads\\.seq_[a-d]$"
          time_windows:
            recent: [1, 15]
            mid: [16, 35]
            tail: [36, 63]
    """

    code = "DIN_ATTN"
    config_key = "din_attn"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    def __init__(
        self,
        targets: list[str] = None,
        time_windows: dict[str, list[int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._targets = targets or []
        self._time_windows: dict[str, tuple[int, int]] = {
            k: (int(v[0]), int(v[1])) for k, v in (time_windows or {}).items()
        }
        self._stats: dict[str, list[dict[str, float]]] = {}
        self._din_modules: list[TargetAwareDINHead] = []
        self._handles: list[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        """Attach forward hooks to matched DIN heads and enable their capture flag."""
        if not self._targets:
            return
        for name, mod in _find_hook_targets(model, self._targets):
            if not isinstance(mod, TargetAwareDINHead):
                continue
            mod.capture_diag = True
            self._din_modules.append(mod)
            self._stats[name] = []
            self._handles.append(mod.register_forward_hook(self._timed_hook(self._make_hook(name))))

    def remove_hooks(self) -> None:
        """Detach hooks and clear the capture flag / cached diag on each DIN head."""
        for h in self._handles:
            h.remove()
        for mod in self._din_modules:
            mod.capture_diag = False
            mod._last_diag = None
        self._handles.clear()
        self._din_modules.clear()

    def _make_hook(self, name: str) -> Any:
        stats = self._stats
        time_windows = self._time_windows

        @torch.compiler.disable
        def hook(mod: nn.Module, args: tuple, output: Any) -> None:
            diag = getattr(mod, "_last_diag", None)
            if diag is None:
                return
            entry: dict[str, float] = {}

            entropy = diag["entropy"].float()
            entry["entropy_mean"] = entropy.mean().item()
            entry["entropy_std"] = entropy.std().item() if entropy.numel() > 1 else 0.0

            if diag["jagged"]:
                attn = diag["attn"].float()  # (total_tokens,)
                cu_seqlens = diag["cu_seqlens"]  # (B+1,)
                tb_ids = diag.get("tb_ids")  # (total_tokens,) or None
                if tb_ids is not None and time_windows:
                    for win_name, (lo, hi) in time_windows.items():
                        in_win = ((tb_ids >= lo) & (tb_ids <= hi)).float()
                        per_sample = segment_sum(attn * in_win, cu_seqlens)  # (B,)
                        entry[f"{win_name}_mass"] = per_sample.mean().item()
            else:
                attn = diag["attn"].float()  # [B, L]
                padding_mask = diag["padding_mask"]  # [B, L]
                tb_ids = diag.get("tb_ids")  # [B, L] or None
                pad = padding_mask.bool().float()
                entry["pad_mass"] = (attn * pad).sum(dim=-1).mean().item()
                if tb_ids is not None and time_windows:
                    for win_name, (lo, hi) in time_windows.items():
                        in_win = ((tb_ids >= lo) & (tb_ids <= hi)).float()
                        entry[f"{win_name}_mass"] = (attn * in_win).sum(dim=-1).mean().item()

            stats[name].append(entry)

        return hook

    def epoch_reset(self) -> None:
        """Drop accumulated per-domain stats at the start of an epoch."""
        for v in self._stats.values():
            v.clear()

    def flush(self) -> None:
        """Discard buffered stats without emitting."""
        for v in self._stats.values():
            v.clear()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Average buffered per-domain attention stats and emit one flattened row."""
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        parts: dict[str, float] = {}
        for name, entries in self._stats.items():
            if not entries:
                continue
            t = _tag(name)
            for key in entries[0].keys():
                vals = [e[key] for e in entries if key in e]
                if vals:
                    parts[f"{t}_{key}"] = float(np.mean(vals))
        if not parts:
            return []
        if self.writer:
            for k, v in parts.items():
                self.writer.add_scalar(f"DINAttn/{tb_tag_prefix}_{k}", v, tb_step)
        return [_flat_kv(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Reconstitute an emitted DIN_ATTN row into `accum` keyed by step or epoch."""
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
