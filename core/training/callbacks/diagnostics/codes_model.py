"""Model-instrumentation codes: LAYER_HEALTH, ATTN, GATE_STATS, DOMAIN_GEOM, GRAD_FLOW, TIN_STATS, EMB_UTIL."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import torch
from torch import nn

from core.training.callbacks.diagnostics.base import DiagBase, _parse_context, _try_numeric
from core.training.callbacks.diagnostics.context import EpochContext, StepContext

# ─────────────────────────────────────────────────────────────────────────────
# Hook helpers
# ─────────────────────────────────────────────────────────────────────────────


def _find_hook_targets(
    model: nn.Module,
    patterns: list[str],
) -> list[tuple[str, nn.Module]]:
    """Find named modules matching any of the given regex patterns."""
    compiled = [re.compile(p) for p in patterns]
    results = []
    for n, m in model.named_modules():
        if any(p.search(n) for p in compiled):
            results.append((n, m))
    return results


def _flat_kv(d: dict[str, float]) -> str:
    return ",".join(f"{k}={v:.4f}" for k, v in sorted(d.items()))


def _parse_flat_kv(payload: str, context: str, accum: dict) -> None:
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


def _unpack_output(output: Any) -> torch.Tensor | None:
    # TODO (nsarang): hack for query_generator which returns list[Tensor].
    # Proper fix: per-module unpack strategy instead of guessing from type.
    if output is None:
        return None
    if isinstance(output, list):
        return torch.cat(output, dim=1)
    return output[0] if isinstance(output, tuple) else output


_BLOCK_PREFIX_RE = re.compile(r"^(?:_orig_mod\.)?blocks\.\d+\.")


def _tag(name: str) -> str:
    stripped = _BLOCK_PREFIX_RE.sub("", name)
    return stripped.replace(".", "_")


def _gini(counts: np.ndarray) -> float:
    """Gini coefficient for a 1-D array of non-negative values."""
    if len(counts) == 0:
        return 0.0
    sorted_c = np.sort(counts)
    n = len(sorted_c)
    index = np.arange(1, n + 1)
    total = sorted_c.sum()
    if total == 0:
        return 0.0
    return float((2.0 * (index * sorted_c).sum()) / (n * total) - (n + 1) / n)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER_HEALTH (merges ACTIV + LAYER_SIM)
# ─────────────────────────────────────────────────────────────────────────────


class LayerHealthCode(DiagBase):
    """Activation stats and block influence per hooked submodule.

    Merges the old ACTIV (mean, std, sparsity) and LAYER_SIM (block influence)
    codes into a single forward hook that captures both input and output.
    """

    code = "LAYER_HEALTH"
    config_key = "layer_health"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    def __init__(
        self,
        targets: list[str] = None,
        activation_stats: bool = True,
        block_influence: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._targets = targets or []
        self._activation_stats = activation_stats
        self._block_influence = block_influence
        self._stats: dict[str, list[dict[str, float]]] = {}
        self._handles: list[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        """Attach hooks to block components and upstream modules."""
        if not self._targets:
            return
        targets = _find_hook_targets(model, self._targets)
        for name, mod in targets:
            self._stats[name] = []
            self._handles.append(mod.register_forward_hook(self._timed_hook(self._make_hook(name))))

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str) -> Any:
        stats = self._stats
        do_activ = self._activation_stats
        do_bi = self._block_influence

        @torch.compiler.disable
        def hook(mod: nn.Module, args: tuple, output: Any) -> None:
            out = _unpack_output(output)
            if out is None:
                return
            entry: dict[str, float] = {}
            out = out.detach().float()

            if do_activ:
                entry["mean"] = out.mean().item()
                entry["std"] = out.std().item()
                entry["sparse"] = (out.abs() < 1e-6).float().mean().item()

            # BI needs matching input/output shapes; skip modules whose first arg isn't a tensor
            if do_bi and isinstance(args[0], torch.Tensor):
                inp = args[0].detach().float()
                if inp.shape == out.shape:
                    cos = nn.functional.cosine_similarity(
                        inp.reshape(-1, inp.shape[-1]),
                        out.reshape(-1, out.shape[-1]),
                        dim=-1,
                    )
                    entry["bi"] = 1.0 - cos.mean().item()

            if entry:
                stats[name].append(entry)

        return hook

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        for v in self._stats.values():
            v.clear()

    def flush(self) -> None:
        """Clear windowed accumulators."""
        for v in self._stats.values():
            v.clear()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        parts: dict[str, float] = {}
        for name, entries in self._stats.items():
            if not entries:
                continue
            t = _tag(name)
            for key in ("mean", "std", "sparse", "bi"):
                vals = [e[key] for e in entries if key in e]
                if vals:
                    parts[f"{t}_{key}"] = float(np.mean(vals))
        if not parts:
            return []
        if self.writer:
            for k, v in parts.items():
                self.writer.add_scalar(f"LayerHealth/{tb_tag_prefix}_{k}", v, tb_step)
        return [_flat_kv(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        _parse_flat_kv(payload, context, accum)


# ─────────────────────────────────────────────────────────────────────────────
# ATTN (replaces XATTN_MAP, adds pre_softmax_l1)
# ─────────────────────────────────────────────────────────────────────────────


class AttnCode(DiagBase):
    """Cross-attention diagnostics: recency mass, entropy, and pre-softmax L1.

    Targets modules matching the ``targets`` patterns in code_config.
    Each matched module must expose ``capture_weights``,
    ``last_attn_weights``, and ``last_pre_softmax`` properties.
    """

    code = "ATTN"
    config_key = "attn"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    def __init__(
        self,
        targets: list[str] = None,
        pre_softmax_l1: bool = True,
        recency_mass: bool = True,
        weight_entropy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._targets = targets or []
        self._pre_softmax_l1 = pre_softmax_l1
        self._snapshots: dict[str, list[dict[str, float]]] = {}
        self._cross_attns: list[tuple[str, nn.Module]] = []
        self._handles: list[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        """Attach hooks to model submodules."""
        if not self._targets:
            return
        for name, mod in _find_hook_targets(model, self._targets):
            mod.capture_weights = True
            self._cross_attns.append((name, mod))
            self._snapshots[name] = []
            self._handles.append(mod.register_forward_pre_hook(self._make_pre_hook(mod)))
            self._handles.append(mod.register_forward_hook(self._timed_hook(self._make_hook(name))))

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for _, mod in self._cross_attns:
            mod.capture_weights = False
        self._cross_attns.clear()

    def _make_pre_hook(self, mod: nn.Module) -> Any:
        code_self = self

        def _pre_hook(module: nn.Module, args: tuple) -> None:
            module.capture_weights = code_self.hooks_active

        return _pre_hook

    def _make_hook(self, name: str) -> Any:
        snapshots = self._snapshots
        do_pre_softmax = self._pre_softmax_l1

        @torch.compiler.disable
        def hook(mod: nn.Module, args: tuple, output: Any) -> None:
            entry: dict[str, float] = {}

            w = mod.last_attn_weights
            if w is not None:
                avg = w.float().mean(dim=(0, 1)).mean(dim=0)  # (Lk,)
                Lk = avg.shape[0]
                tail = max(1, Lk // 5)
                entry["recency_mass"] = avg[-tail:].sum().item()
                p = avg.clamp(min=1e-8)
                entry["entropy"] = -(p * p.log()).sum().item()

            if do_pre_softmax:
                pre = mod.last_pre_softmax
                if pre is not None:
                    valid = pre[pre.isfinite()]
                    if valid.numel() > 0:
                        entry["pre_softmax_l1"] = valid.float().abs().mean().item()

            if hasattr(mod, "last_effective_seq_len") and mod.last_effective_seq_len is not None:
                entry["effective_seq_len"] = mod.last_effective_seq_len

            if entry:
                snapshots[name].append(entry)

        return hook

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        for v in self._snapshots.values():
            v.clear()

    def flush(self) -> None:
        """Clear windowed accumulators."""
        for v in self._snapshots.values():
            v.clear()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        parts: dict[str, float] = {}
        for name, entries in self._snapshots.items():
            if not entries:
                continue
            t = _tag(name)
            for key in ("recency_mass", "entropy", "pre_softmax_l1", "effective_seq_len"):
                vals = [e[key] for e in entries if key in e]
                if vals:
                    parts[f"{t}_{key}"] = float(np.mean(vals))
        if not parts:
            return []
        if self.writer:
            for k, v in parts.items():
                self.writer.add_scalar(f"Attn/{tb_tag_prefix}_{k}", v, tb_step)
        return [_flat_kv(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        _parse_flat_kv(payload, context, accum)


# ─────────────────────────────────────────────────────────────────────────────
# GATE_STATS
# ─────────────────────────────────────────────────────────────────────────────


class _GateTarget:
    """Parsed target spec for GateStatsCode."""

    __slots__ = ("labels", "mode", "pattern", "transform")

    def __init__(self, spec: str | dict) -> None:
        if isinstance(spec, str):
            self.pattern = spec
            self.mode = "aggregate"
            self.transform = None
            self.labels: list[str] | None = None
        else:
            self.pattern = spec["pattern"]
            self.mode = spec.get("mode", "aggregate")
            self.transform = spec.get("transform")
            self.labels = spec.get("labels")


class GateStatsCode(DiagBase):
    """Output statistics for gate modules.

    Hooks on modules matching the ``targets`` patterns in code_config.

    Each target can be a regex string (backward-compatible aggregate mode)
    or a dict with fine-grained control::

        targets:
          - "fusion_gate$"                       # aggregate: mean/std/min/max
          - pattern: "weight_projection$"        # per_index: one value per dim
            mode: per_index
            transform: softmax                   # apply before recording
            labels: [unified, context, scale, graph, candidate]
    """

    code = "GATE_STATS"
    config_key = "gate_stats"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    def __init__(
        self,
        targets: list = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._targets = [_GateTarget(t) for t in (targets or [])]
        self._target_map: dict[str, _GateTarget] = {}
        self._stats: dict[str, list[dict[str, float]]] = {}
        self._handles: list[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        """Attach hooks to model submodules."""
        if not self._targets:
            return
        for gt in self._targets:
            for name, mod in _find_hook_targets(model, [gt.pattern]):
                self._stats[name] = []
                self._target_map[name] = gt
                self._handles.append(
                    mod.register_forward_hook(self._timed_hook(self._make_hook(name, gt)))
                )

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str, gt: _GateTarget) -> Any:
        stats = self._stats

        @torch.compiler.disable
        def hook(mod: nn.Module, args: tuple, output: Any) -> None:
            t = _unpack_output(output)
            if t is None:
                return
            t = t.detach().float()
            if gt.transform == "softmax":
                t = torch.softmax(t, dim=-1)
            elif gt.transform == "sigmoid":
                t = torch.sigmoid(t)
            if gt.mode == "per_index":
                d = t.shape[-1]
                entry: dict[str, float] = {}
                for i in range(d):
                    label = gt.labels[i] if gt.labels and i < len(gt.labels) else f"idx{i}"
                    entry[label] = t[..., i].mean().item()
                stats[name].append(entry)
            else:
                stats[name].append(
                    {
                        "mean": t.mean().item(),
                        "std": t.std().item(),
                        "min": t.min().item(),
                        "max": t.max().item(),
                    }
                )

        return hook

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        for v in self._stats.values():
            v.clear()

    def flush(self) -> None:
        """Clear windowed accumulators."""
        for v in self._stats.values():
            v.clear()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        parts: dict[str, float] = {}
        for name, entries in self._stats.items():
            if not entries:
                continue
            t = _tag(name)
            keys = entries[0].keys()
            for key in keys:
                vals = [e[key] for e in entries]
                parts[f"{t}_{key}"] = float(np.mean(vals))
        if not parts:
            return []
        if self.writer:
            for k, v in parts.items():
                self.writer.add_scalar(f"GateStats/{tb_tag_prefix}_{k}", v, tb_step)
        return [_flat_kv(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        _parse_flat_kv(payload, context, accum)


# ─────────────────────────────────────────────────────────────────────────────
# WRITER_STATS
# ─────────────────────────────────────────────────────────────────────────────


class WriterStatsCode(DiagBase):
    """Per-domain ConvGate writer statistics.

    Reads diagnostic scalars stored by ``ConvGate._store_diag`` after each
    forward pass. Emits per-domain gate mean/std and, when ADS-lite FiLM is
    enabled, the abs-mean magnitudes of the FiLM gamma and beta offsets.

    Emitted keys (per domain ``d``):

    - ``seq_local_writer_convs_d_gate_mean``
    - ``seq_local_writer_convs_d_gate_std``
    - ``seq_local_writer_convs_d_film_gamma_abs_mean`` (ADS-lite only)
    - ``seq_local_writer_convs_d_film_beta_abs_mean`` (ADS-lite only)

    Enable in diagnostics config with::

        writer_stats: {}
    """

    code = "WRITER_STATS"
    config_key = "writer_stats"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stats: dict[str, list[dict[str, float]]] = {}
        self._handles: list[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        """Attach hooks to SeqLocalWriter convolution modules with diagnostics state."""
        for name, mod in model.named_modules():
            if re.search(r"seq_local_writer\.convs\.", name) and hasattr(mod, "_diag"):
                self._stats[name] = []
                self._handles.append(
                    mod.register_forward_hook(self._timed_hook(self._make_hook(name)))
                )

    def remove_hooks(self) -> None:
        """Detach all writer-stat hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str) -> Any:
        stats = self._stats

        @torch.compiler.disable
        def hook(mod: nn.Module, args: tuple, output: Any) -> None:
            if mod._diag:
                entry = dict(mod._diag)
                for pname in ("alpha", "beta", "gamma"):
                    p = getattr(mod, pname, None)
                    if isinstance(p, nn.Parameter):
                        entry[pname] = p.detach().float().item()
                stats[name].append(entry)

        return hook

    def epoch_reset(self) -> None:
        """Reset per-epoch writer-stat accumulators."""
        for v in self._stats.values():
            v.clear()

    def flush(self) -> None:
        """Clear writer-stat accumulators after emission."""
        for v in self._stats.values():
            v.clear()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Average collected writer diagnostics and emit one flat payload."""
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        parts: dict[str, float] = {}
        for name, entries in self._stats.items():
            if not entries:
                continue
            t = _tag(name)
            for key in entries[0].keys():
                vals = [e[key] for e in entries]
                parts[f"{t}_{key}"] = float(np.mean(vals))
        if not parts:
            return []
        if self.writer:
            for k, v in parts.items():
                self.writer.add_scalar(f"WriterStats/{tb_tag_prefix}_{k}", v, tb_step)
        return [_flat_kv(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a writer-stat payload into the accumulator."""
        _parse_flat_kv(payload, context, accum)


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN_GEOM
# ─────────────────────────────────────────────────────────────────────────────


class DomainGeomCode(DiagBase):
    """Domain encoder geometry: pairwise cosine similarity and within-domain diversity.

    Forward hooks on domain encoder modules (ns_tokenizers). At collect time,
    computes pairwise cosine similarity between domain outputs and
    within-domain standard deviation (diversity).
    """

    code = "DOMAIN_GEOM"
    config_key = "domain_geom"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    def __init__(
        self,
        targets: list[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._targets = targets or []
        self._outputs: dict[str, list[torch.Tensor]] = {}
        self._handles: list[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        """Attach hooks to model submodules."""
        if not self._targets:
            return
        for name, mod in _find_hook_targets(model, self._targets):
            self._outputs[name] = []
            self._handles.append(mod.register_forward_hook(self._timed_hook(self._make_hook(name))))

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str) -> Any:
        outputs = self._outputs

        @torch.compiler.disable
        def hook(mod: nn.Module, args: tuple, output: Any) -> None:
            t = _unpack_output(output)
            if t is None:
                return
            t = t.detach().float()
            # Store mean-pooled representation per domain per forward pass
            outputs[name].append(t.mean(dim=tuple(range(t.ndim - 1))))

        return hook

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        for v in self._outputs.values():
            v.clear()

    def flush(self) -> None:
        """Clear windowed accumulators."""
        for v in self._outputs.values():
            v.clear()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        parts: dict[str, float] = {}

        domain_means: dict[str, torch.Tensor] = {}
        for name, tensors in self._outputs.items():
            if not tensors:
                continue
            stacked = torch.stack(tensors)  # (N, D)
            domain_means[name] = stacked.mean(dim=0)
            t = _tag(name)
            if stacked.shape[0] > 1:
                parts[f"{t}_diversity"] = stacked.std(dim=0).mean().item()

        # Pairwise cosine similarity between domain mean embeddings
        names = sorted(domain_means.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = domain_means[names[i]].unsqueeze(0)
                b = domain_means[names[j]].unsqueeze(0)
                sim = nn.functional.cosine_similarity(a, b, dim=-1).item()
                ta = _tag(names[i])
                tb_name = _tag(names[j])
                parts[f"cos_{ta}_{tb_name}"] = sim

        if not parts:
            return []
        if self.writer:
            for k, v in parts.items():
                self.writer.add_scalar(f"DomainGeom/{tb_tag_prefix}_{k}", v, tb_step)
        return [_flat_kv(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        _parse_flat_kv(payload, context, accum)


# ─────────────────────────────────────────────────────────────────────────────
# EFF_RANK
# ─────────────────────────────────────────────────────────────────────────────


def _effective_rank(matrix: torch.Tensor) -> float:
    """Compute effective rank of a 2D matrix via SVD entropy."""
    sv = torch.linalg.svdvals(matrix.float())
    sv = sv[sv > 1e-7]
    if sv.numel() == 0:
        return 1.0
    p = sv / sv.sum()
    entropy = -(p * p.log()).sum().item()
    return float(np.exp(entropy))


def _spectral_rank_stats(matrix: torch.Tensor) -> tuple[float, float]:
    """Return entropy effective rank and Information Abundance from one SVD."""
    sv = torch.linalg.svdvals(matrix.float())
    sv = sv[sv > 1e-7]
    if sv.numel() == 0:
        return 1.0, 1.0
    p = sv / sv.sum()
    entropy = -(p * p.log()).sum().item()
    eff_rank = float(np.exp(entropy))
    information_abundance = float(sv.sum().item() / sv.max().item())
    return eff_rank, information_abundance


class EffRankCode(DiagBase):
    """Effective rank of representations at hooked pipeline stages.

    Uses a preallocated ring buffer on-device -- zero allocations in the
    hook path. SVD and the single GPU->CPU transfer happen at collect time.
    """

    code = "EFF_RANK"
    config_key = "eff_rank"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    def __init__(
        self,
        targets: list[str] = None,
        max_samples: int = 2048,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._targets = targets or []
        self._max_samples = max_samples
        # Ring buffer state: preallocated tensor + write cursor
        self._buffers: dict[str, torch.Tensor] = {}
        self._cursors: dict[str, int] = {}
        self._filled: dict[str, bool] = {}
        self._handles: list[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        """Attach forward hooks to targeted modules."""
        if not self._targets:
            return
        for name, mod in _find_hook_targets(model, self._targets):
            self._buffers[name] = None
            self._cursors[name] = 0
            self._filled[name] = False
            self._handles.append(mod.register_forward_hook(self._timed_hook(self._make_hook(name))))

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str) -> Any:
        buffers = self._buffers
        cursors = self._cursors
        filled = self._filled
        max_samples = self._max_samples
        # Cap GPU-side work per hook invocation
        max_per_step = min(256, max_samples)

        @torch.compiler.disable
        def hook(mod: nn.Module, args: tuple, output: Any) -> None:
            t = _unpack_output(output)
            if t is None:
                return
            t = t.detach()
            flat = t.reshape(-1, t.shape[-1])
            n = flat.shape[0]
            # Subsample on GPU to bound per-step cost
            if n > max_per_step:
                indices = torch.randint(n, (max_per_step,), device=flat.device)
                flat = flat[indices]
            else:
                flat = flat[:max_per_step]
            k = flat.shape[0]

            # Lazy-allocate ring buffer on first invocation (need hidden dim)
            buf = buffers[name]
            if buf is None:
                buf = torch.empty(max_samples, flat.shape[-1], device=flat.device, dtype=flat.dtype)
                buffers[name] = buf

            # Write into ring buffer -- no allocation, no cat
            cursor = cursors[name]
            space = max_samples - cursor
            if k <= space:
                buf[cursor : cursor + k] = flat
                cursors[name] = cursor + k
            else:
                buf[cursor:] = flat[:space]
                remainder = k - space
                buf[:remainder] = flat[space:]
                cursors[name] = remainder
                filled[name] = True

            if cursors[name] >= max_samples:
                cursors[name] = 0
                filled[name] = True

        return hook

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        for k in self._cursors:
            self._cursors[k] = 0
            self._filled[k] = False

    def flush(self) -> None:
        """Clear windowed accumulators."""
        for k in self._cursors:
            self._cursors[k] = 0
            self._filled[k] = False

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Compute effective rank via SVD and return payload strings."""
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        parts: dict[str, float] = {}
        for name, buf in self._buffers.items():
            if buf is None:
                continue
            n_valid = self._max_samples if self._filled[name] else self._cursors[name]
            if n_valid < 2:
                continue
            matrix = buf[:n_valid].float().cpu()
            t = _tag(name)
            parts[f"{t}_eff_rank"] = _effective_rank(matrix)
        if not parts:
            return []
        if self.writer:
            for k, v in parts.items():
                self.writer.add_scalar(f"EffRank/{tb_tag_prefix}_{k}", v, tb_step)
        return [_flat_kv(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        _parse_flat_kv(payload, context, accum)


# ─────────────────────────────────────────────────────────────────────────────
# GRAD_FLOW
# ─────────────────────────────────────────────────────────────────────────────


class GradFlowCode(DiagBase):
    """Per-layer gradient L2 norms via backward hooks for gradient-flow heatmaps.

    Uses ``register_full_backward_hook`` on Linear layers and attention modules.
    Accumulates every step until ``dense_log_until``, then only on emit steps.
    """

    code = "GRAD_FLOW"
    config_key = "grad_flow"
    emit = frozenset({"step"})
    accumulate = frozenset({"always"})

    def __init__(
        self,
        targets: list[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._targets = targets or []
        self._grad_norms: dict[str, list[float]] = {}
        self._handles: list[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        """Attach hooks to model submodules."""
        if not self._targets:
            return
        for name, mod in _find_hook_targets(model, self._targets):
            self._grad_norms[name] = []
            self._handles.append(
                mod.register_full_backward_hook(self._timed_backward_hook(self._make_hook(name)))
            )

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str) -> Any:
        grad_norms = self._grad_norms

        @torch.compiler.disable
        def hook(mod: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            if grad_output[0] is not None:
                norm = grad_output[0].detach().float().norm(2).item()
                grad_norms[name].append(norm)

        return hook

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        for v in self._grad_norms.values():
            v.clear()
        self._step_count = 0

    def flush(self) -> None:
        """Clear windowed accumulators."""
        for v in self._grad_norms.values():
            v.clear()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        parts: dict[str, float] = {}
        for name, norms in self._grad_norms.items():
            if not norms:
                continue
            t = _tag(name)
            parts[f"{t}_grad_l2"] = float(np.mean(norms))
        if not parts:
            return []
        if self.writer:
            for k, v in parts.items():
                self.writer.add_scalar(f"GradFlow/{tb_tag_prefix}_{k}", v, tb_step)
        return [_flat_kv(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        _parse_flat_kv(payload, context, accum)


# ─────────────────────────────────────────────────────────────────────────────
# LANE_COSINE
# ─────────────────────────────────────────────────────────────────────────────


class LaneCosineCode(DiagBase):
    """Pairwise cosine similarity between parallel representations.

    Two modes, selected by config:

    **Module mode** — hooks a single module and reads its input list.
    Use when parallel tensors are passed as a list to one module
    (e.g. symbiosis ``lane_mixer``).

    **Targets mode** — hooks multiple modules and compares their outputs.
    Use when parallel tensors are produced by separate modules
    (e.g. hyformer multihead ``base_head``, ``seq_head``, etc.).

    Config examples::

        # Module mode (symbiosis)
        lane_cosine:
          module: "lane_mixer"
          labels: [unified, context, scale, graph, candidate]

        # Targets mode (hyformer)
        lane_cosine:
          accumulate_freq: 0.02
          targets:
            - "base_head$"
            - "seq_head$"
            - "domain_head$"
            - "fusion_head$"
    """

    code = "LANE_COSINE"
    config_key = "lane_cosine"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    def __init__(
        self,
        module: str = None,
        labels: list[str] = None,
        targets: list[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._module_name = module
        self._labels = labels
        self._targets = targets or []
        self._target_names: list[str] = []
        self._last_output: dict[str, torch.Tensor | None] = {}
        self._entries: list[dict[str, float]] = []
        self._handles: list[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        """Attach forward hooks in module or targets mode."""
        if self._module_name:
            mod = None
            for name, m in model.named_modules():
                if name == self._module_name:
                    mod = m
                    break
            if mod is None:
                return
            self._handles.append(
                mod.register_forward_hook(self._timed_hook(self._make_module_hook()))
            )
        elif self._targets:
            for name, mod in _find_hook_targets(model, self._targets):
                self._target_names.append(name)
                self._last_output[name] = None
                self._handles.append(
                    mod.register_forward_hook(self._timed_hook(self._make_target_hook(name)))
                )

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @staticmethod
    def _pairwise_cosine(tensors: list[torch.Tensor], names: list[str]) -> dict[str, float]:
        n = len(tensors)
        entry: dict[str, float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                a = tensors[i].float()
                b = tensors[j].float()
                if a.shape != b.shape:
                    continue
                cos = (
                    torch.cosine_similarity(
                        a.reshape(-1, a.shape[-1]),
                        b.reshape(-1, b.shape[-1]),
                        dim=-1,
                    )
                    .mean()
                    .item()
                )
                entry[f"{names[i]}_x_{names[j]}"] = cos
        return entry

    def _make_module_hook(self) -> Any:
        entries = self._entries
        labels = self._labels
        pairwise = self._pairwise_cosine

        @torch.compiler.disable
        def hook(mod: nn.Module, args: tuple, output: Any) -> None:
            lanes = args[0]
            n = len(lanes)
            names = labels if labels and len(labels) == n else [f"lane{i}" for i in range(n)]
            detached = [t.detach() for t in lanes]
            entry = pairwise(detached, names)
            if entry:
                entries.append(entry)

        return hook

    def _make_target_hook(self, name: str) -> Any:
        last_output = self._last_output

        @torch.compiler.disable
        def hook(mod: nn.Module, args: tuple, output: Any) -> None:
            t = _unpack_output(output)
            if t is not None:
                last_output[name] = t.detach()

        return hook

    def step(self, ctx: StepContext | dict[str, Any], emit: bool = False) -> None:
        """Compute pairwise cosine from buffered target outputs."""
        super().step(ctx, emit=emit)
        if not self._target_names or not self.hooks_active:
            return
        outputs = self._last_output
        names = self._target_names
        if any(outputs[k] is None for k in names):
            return
        entry = self._pairwise_cosine(
            [outputs[k] for k in names],
            [_tag(k) for k in names],
        )
        if entry:
            self._entries.append(entry)
        for k in names:
            outputs[k] = None

    def epoch_reset(self) -> None:
        """Reset per-epoch cosine similarity entries."""
        self._entries.clear()

    def flush(self) -> None:
        """Clear accumulated entries after emission."""
        self._entries.clear()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Average pairwise cosine similarities and return as flat key=value payload."""
        if not self._entries:
            return []
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        parts: dict[str, float] = {}
        keys = self._entries[0].keys()
        for key in keys:
            vals = [e[key] for e in self._entries if key in e]
            if vals:
                parts[key] = float(np.mean(vals))
        if not parts:
            return []
        if self.writer:
            for k, v in parts.items():
                self.writer.add_scalar(f"LaneCosine/{tb_tag_prefix}_{k}", v, tb_step)
        return [_flat_kv(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse flat key=value cosine similarity payload."""
        _parse_flat_kv(payload, context, accum)


# ─────────────────────────────────────────────────────────────────────────────
# TIN_STATS
# ─────────────────────────────────────────────────────────────────────────────


class TinStatsCode(DiagBase):
    """TIN filter health: keep ratio, threshold drift, score separation,
    domain budget share, and position distribution of kept events.

    Reads ``_last_stats`` dict stashed by ``TINFilter.forward()``.
    No-op when the model has no ``tin_filter``.
    """

    code = "TIN_STATS"
    config_key = "tin_stats"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entries: list[dict[str, float]] = []
        self._handles: list[Any] = []
        self._tin_ref: Any = None

    def register_hooks(self, model: nn.Module) -> None:
        """Attach a forward hook to the tin_filter module if present."""
        tin = getattr(model, "tin_filter", None)
        if tin is None:
            return
        self._tin_ref = tin
        self._handles.append(tin.register_forward_hook(self._timed_hook(self._make_hook())))

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._tin_ref = None

    def _make_hook(self) -> Any:
        entries = self._entries
        tin = self._tin_ref

        @torch.compiler.disable
        def hook(mod: nn.Module, args: tuple, output: Any) -> None:
            stats = getattr(tin, "_last_stats", None)
            if not stats:
                return
            flat: dict[str, float] = {
                "keep_ratio": stats["keep_ratio"],
                "score_sep": stats["score_sep"],
                "kept_score_mean": stats["kept_score_mean"],
                "kept_score_std": stats["kept_score_std"],
                "disc_score_mean": stats["disc_score_mean"],
                "disc_score_std": stats["disc_score_std"],
            }
            for domain, dstats in stats.get("domains", {}).items():
                flat[f"{domain}_keep"] = dstats["keep"]
                flat[f"{domain}_budget"] = dstats["budget"]
                flat[f"{domain}_pos_norm"] = dstats["pos_norm"]
            entries.append(flat)

        return hook

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        self._entries.clear()

    def flush(self) -> None:
        """Clear windowed accumulators."""
        self._entries.clear()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        if not self._entries:
            return []
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        parts: dict[str, float] = {}
        keys = self._entries[0].keys()
        for key in keys:
            vals = [e[key] for e in self._entries if key in e]
            if vals:
                parts[key] = float(np.mean(vals))
        if not parts:
            return []
        if self.writer:
            for k, v in parts.items():
                self.writer.add_scalar(f"TinStats/{tb_tag_prefix}_{k}", v, tb_step)
        return [_flat_kv(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        _parse_flat_kv(payload, context, accum)


# ─────────────────────────────────────────────────────────────────────────────
# EMB_UTIL (v2 upgrade with Gini coefficient and configurable percentiles)
# ─────────────────────────────────────────────────────────────────────────────


class EmbUtilCode(DiagBase):
    """Embedding table utilization via forward-hook hit counting.

    Registers ``forward_pre_hook`` on every ``nn.Embedding`` to count how
    many times each row (excluding padding row 0) is looked up per epoch.
    At collect time, emits hit rate, percentile distribution, and Gini
    coefficient over hit counts.
    """

    code = "EMB_UTIL"
    config_key = "emb_util"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    def __init__(
        self,
        hit_percentiles: list[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._percentiles = (
            hit_percentiles if hit_percentiles is not None else [10, 25, 50, 75, 90, 95, 99]
        )
        self._hit_counts: dict[str, torch.Tensor] = {}
        self._hook_handles: list[Any] = []
        self._hooked = False

    def register_hooks(self, model: nn.Module) -> None:
        """Attach hooks to model submodules."""
        if self._hooked:
            return
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Embedding):
                continue
            self._hit_counts[name] = torch.zeros(mod.num_embeddings, dtype=torch.long)
            handle = mod.register_forward_pre_hook(self._make_counter(name))
            self._hook_handles.append(handle)
        self._hooked = True

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._hooked = False

    def _make_counter(self, name: str) -> Any:
        hit_counts = self._hit_counts
        code_self = self

        @torch.compiler.disable
        def _hook(module: nn.Module, args: tuple) -> None:
            if not code_self.hooks_active:
                return
            idx = args[0].detach().reshape(-1).long().cpu()
            valid = idx[idx > 0]
            if valid.numel() > 0:
                hit_counts[name].scatter_add_(0, valid, torch.ones_like(valid))

        return _hook

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        for c in self._hit_counts.values():
            c.zero_()

    def flush(self) -> None:
        """Clear windowed accumulators."""

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        if not self._hit_counts:
            return []

        total_rows = 0
        total_hit = 0
        all_counts: list[np.ndarray] = []
        per_table: list[list[Any]] = []

        for name in sorted(self._hit_counts):
            counts = self._hit_counts[name].numpy()
            data = counts[1:]
            n_hit = int((data > 0).sum())
            total_rows += len(data)
            total_hit += n_hit
            per_table.append([name, len(data), n_hit])
            all_counts.append(data)

        hit_rate = total_hit / total_rows if total_rows > 0 else 0.0
        summary = f"total={total_rows},hit={total_hit},hit_rate={hit_rate:.4f}"

        merged = np.concatenate(all_counts) if all_counts else np.array([])
        hit_vals = merged[merged > 0]
        if len(hit_vals) > 0:
            pct_parts = [f"p{p}={int(np.percentile(hit_vals, p))}" for p in self._percentiles]
            pct_parts.append(f"max={int(hit_vals.max())}")
            summary += "," + ",".join(pct_parts)
            summary += f",gini={_gini(merged):.4f}"

        payloads = [summary]
        payloads.append(f"tables:{json.dumps(per_table, separators=(',', ':'))}")

        if self.writer:
            self.writer.add_scalar(f"EmbUtil/{tb_tag_prefix}_hit_rate", hit_rate, tb_step)
            if len(hit_vals) > 0:
                self.writer.add_scalar(
                    f"EmbUtil/{tb_tag_prefix}_hits_p50", float(np.percentile(hit_vals, 50)), tb_step
                )
                self.writer.add_scalar(
                    f"EmbUtil/{tb_tag_prefix}_hits_p99", float(np.percentile(hit_vals, 99)), tb_step
                )
                self.writer.add_scalar(f"EmbUtil/{tb_tag_prefix}_gini", _gini(merged), tb_step)
        return payloads

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        ctx = _parse_context(context)
        if payload.startswith("tables:"):
            tables = json.loads(payload[7:])
            if "epoch" in ctx:
                accum.setdefault("epoch_tables", {})[ctx["epoch"]] = tables
            else:
                accum.setdefault("step_tables", {})[ctx.get("step", 0)] = tables
        else:
            entry: dict[str, Any] = {}
            for kv in payload.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    entry[k] = _try_numeric(v)
            if "epoch" in ctx:
                accum.setdefault("epochs", {})[ctx["epoch"]] = entry
            else:
                accum.setdefault("steps", {})[ctx.get("step", 0)] = entry


# ─────────────────────────────────────────────────────────────────────────────
# EMB_RANK — effective rank of embedding tables (weights), not activations
# ─────────────────────────────────────────────────────────────────────────────


def _iter_embedding_weights(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Yield ``(name, weight)`` for every embedding table across storage formats.

    Covers module-backed tables by walking ``named_modules`` -- ``nn.Embedding``
    (padded path and the torchrec EmbeddingCollection sub-modules) and
    ``nn.EmbeddingBag`` (torchrec EmbeddingBagCollection static path) -- plus
    FBGEMM TBE tables whose weights are fused buffers rather than modules,
    surfaced through the embedder's ``tbe_weight_views()``. Weights are
    de-duplicated by storage pointer so tied tables are measured once.
    """
    pairs: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Embedding, nn.EmbeddingBag)):
            w = mod.weight
            if w.data_ptr() in seen:
                continue
            seen.add(w.data_ptr())
            pairs.append((name, w))
        elif getattr(mod, "use_tbe", False) and hasattr(mod, "tbe_weight_views"):
            for tname, w in mod.tbe_weight_views().items():
                if w.data_ptr() in seen:
                    continue
                seen.add(w.data_ptr())
                pairs.append((f"{name}.tbe.{tname}", w))
    return pairs


class EmbRankCode(DiagBase):
    """Effective rank of embedding *tables* (weight matrices).

    Complements EMB_UTIL (per-row hit counting) by asking whether a table's
    columns span its full embedding dimension or have collapsed onto a
    low-rank subspace -- a signal of an under-trained or redundant table. SVD
    runs at collect time on the selected tables only.

    Targeted: ``targets`` is a list of regexes matched against table names, so
    you measure specific tables (e.g. the high-cardinality item table) rather
    than every table in the model. With no targets the code is a no-op.
    """

    code = "EMB_RANK"
    config_key = "emb_rank"
    emit = frozenset({"epoch"})
    accumulate = frozenset()

    def __init__(
        self,
        targets: list[str] = None,
        max_rows: int = 50000,
        skip_padding_row: bool = True,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._targets = targets or []
        self._max_rows = max_rows
        self._skip_padding_row = skip_padding_row
        self._seed = seed

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """SVD each selected table at epoch end and emit effective rank + ratio."""
        if not self._targets:
            return []
        model = ctx.get("model") if isinstance(ctx, dict) else getattr(ctx, "model", None)
        if model is None:
            return []
        tb_tag_prefix, tb_step = self._tb_step(phase, ctx)
        compiled = [re.compile(p) for p in self._targets]
        # Local RNG: row subsampling must not perturb the global torch RNG stream.
        rng = np.random.default_rng(self._seed)
        parts: dict[str, float] = {}
        for name, weight in _iter_embedding_weights(model):
            if not any(p.search(name) for p in compiled):
                continue
            w = weight.detach()
            # Padding row 0 is a near-zero duplicate that biases the spectrum.
            if self._skip_padding_row and w.shape[0] > 1:
                w = w[1:]
            n_rows = w.shape[0]
            dim = w.shape[-1]
            if n_rows < 2 or dim < 1:
                continue
            if n_rows > self._max_rows:
                idx = rng.choice(n_rows, size=self._max_rows, replace=False)
                w = w[torch.from_numpy(idx).to(w.device)]
            er, ia = _spectral_rank_stats(w.float().cpu())
            t = _tag(name)
            parts[f"{t}_eff_rank"] = er
            parts[f"{t}_ratio"] = er / dim
            parts[f"{t}_ia"] = ia
        if not parts:
            return []
        if self.writer:
            for k, v in parts.items():
                self.writer.add_scalar(f"EmbRank/{tb_tag_prefix}_{k}", v, tb_step)
        return [_flat_kv(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        _parse_flat_kv(payload, context, accum)
