"""Data-health codes: DENSE_STATS, SEQ_LENS, LABEL_DIST, OOB."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import numpy as np

from core.training.callbacks.diagnostics.base import DiagBase, _try_numeric
from core.training.callbacks.diagnostics.context import EpochContext, StepContext


class DenseStatsCode(DiagBase):
    """Dense feature distribution stats."""

    code = "DENSE_STATS"
    config_key = "dense_stats"
    emit = frozenset({"warmup"})
    accumulate = frozenset({"warmup"})
    init_params: ClassVar[tuple[str, ...]] = ("schema",)

    _DEFAULT_FILTERS: ClassVar[dict[str, str]] = {
        "user": "entity = 'user' and dtype = 'numerical' and scope = 'static' and source = 'original'",
        "item": "entity = 'item' and dtype = 'numerical' and scope = 'static' and source = 'original'",
    }

    def __init__(
        self,
        schema=None,
        filters: dict[str, str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._schema = schema
        self._filters = filters or self._DEFAULT_FILTERS
        self._buffers: dict[str, list[np.ndarray]] = {}

    def step(self, ctx: StepContext, emit: bool = False) -> None:
        """Accumulate per-step data."""
        if not self._schema:
            return
        for tag, expr in self._filters.items():
            extracted = self._schema.extract(ctx.batch, expr=expr, cat=True)
            if extracted is None:
                continue
            arr = extracted.cpu().numpy() if hasattr(extracted, "cpu") else extracted
            if arr.size > 0:
                self._buffers.setdefault(tag, []).append(arr)

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        if not self._buffers:
            return []
        payloads = []
        for tag, bufs in self._buffers.items():
            if not bufs:
                continue
            all_dense = np.concatenate(bufs, axis=0)
            ncols = all_dense.shape[1]
            stds = np.std(all_dense, axis=0)
            sparsities = np.mean(all_dense == 0, axis=0)
            n_const = int(np.sum(stds < 1e-6))
            n_high_sparse = int(np.sum(sparsities > 0.9)) - n_const
            prefix = f"{tag}_" if tag != "user" else ""
            payloads.append(f"{prefix}cols={ncols},constant={n_const},high_sparse={n_high_sparse}")
            per_col = [
                [
                    round(float(np.min(all_dense[:, c])), 4),
                    round(float(np.max(all_dense[:, c])), 4),
                    round(float(np.mean(all_dense[:, c])), 4),
                    round(float(stds[c]), 4),
                    round(float(sparsities[c]), 4),
                ]
                for c in range(ncols)
            ]
            full_tag = f"{tag}_full" if tag != "user" else "full"
            payloads.append(f"{full_tag}:{json.dumps(per_col, separators=(',', ':'))}")
        return payloads

    def flush(self) -> None:
        """Clear windowed accumulators after emission."""
        self._buffers.clear()

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        self._buffers.clear()

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        colon = payload.find(":")
        if colon > 0 and "=" not in payload[:colon]:
            key = payload[:colon]
            accum[key] = json.loads(payload[colon + 1 :])
        else:
            for kv in payload.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    accum[k] = _try_numeric(v)


class SeqLensCode(DiagBase):
    """Sequence length distribution and truncation rate per domain."""

    code = "SEQ_LENS"
    config_key = "seq_lens"
    emit = frozenset({"warmup"})
    accumulate = frozenset({"warmup"})
    init_params: ClassVar[tuple[str, ...]] = ("seq_domains",)

    def __init__(
        self,
        seq_domains: list[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_domains = seq_domains or []
        self._buffers: dict[str, list[np.ndarray]] = {d: [] for d in self.seq_domains}
        self._raw_buffers: dict[str, list[np.ndarray]] = {d: [] for d in self.seq_domains}

    def step(self, ctx: StepContext, emit: bool = False) -> None:
        """Accumulate per-step data."""
        for d in self.seq_domains:
            key = f"{d}_len"
            if key in ctx.batch:
                self._buffers[d].append(ctx.batch[key].cpu().numpy())
            raw_key = f"{d}_raw_len"
            if raw_key in ctx.batch:
                self._raw_buffers[d].append(ctx.batch[raw_key].cpu().numpy())

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        parts: list[str] = []
        for d in self.seq_domains:
            if not self._buffers[d]:
                continue
            all_lens = np.concatenate(self._buffers[d])
            empty = float(np.mean(all_lens == 0))
            domain_parts = (
                f"{d}:p25={np.percentile(all_lens, 25):.0f},"
                f"p50={np.percentile(all_lens, 50):.0f},"
                f"p75={np.percentile(all_lens, 75):.0f},"
                f"p95={np.percentile(all_lens, 95):.0f},"
                f"max={int(np.max(all_lens))},"
                f"empty={empty:.4f}"
            )
            if self._raw_buffers[d]:
                all_raw = np.concatenate(self._raw_buffers[d])
                nonempty = all_raw > 0
                if nonempty.any():
                    truncated = all_raw[nonempty] > all_lens[nonempty]
                    trunc_rate = float(np.mean(truncated))
                    raw_nonempty = all_raw[nonempty]
                    domain_parts += (
                        f",trunc_rate={trunc_rate:.4f}"
                        f",raw_p25={np.percentile(raw_nonempty, 25):.0f}"
                        f",raw_p50={np.percentile(raw_nonempty, 50):.0f}"
                        f",raw_p75={np.percentile(raw_nonempty, 75):.0f}"
                        f",raw_p95={np.percentile(raw_nonempty, 95):.0f}"
                        f",raw_max={int(np.max(raw_nonempty))}"
                    )
            parts.append(domain_parts)
        return [";".join(parts)] if parts else []

    def flush(self) -> None:
        """Clear windowed accumulators after emission."""
        for d in self.seq_domains:
            self._buffers[d].clear()
            self._raw_buffers[d].clear()

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        for d in self.seq_domains:
            self._buffers[d].clear()
            self._raw_buffers[d].clear()

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        for domain_block in payload.split(";"):
            if ":" not in domain_block:
                continue
            domain, rest = domain_block.split(":", 1)
            d: dict[str, Any] = {}
            for kv in rest.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    d[k] = _try_numeric(v)
            accum.setdefault("domains", {})[domain] = d


class LabelDistCode(DiagBase):
    """Label rate, running positive rate, and per-class loss."""

    code = "LABEL_DIST"
    config_key = "label_dist"
    emit = frozenset({"warmup", "step"})
    accumulate = frozenset({"always"})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rates: list[float] = []
        self._loss_pos: list[float] = []
        self._loss_neg: list[float] = []
        self._running_sum: float = 0.0
        self._running_count: int = 0

    def step(self, ctx: StepContext, emit: bool = False) -> None:
        """Accumulate per-step data."""
        stats = ctx.label_stats
        self._rates.append(stats["mean"])
        self._running_sum += stats["sum"]
        self._running_count += stats["count"]

        pcl = ctx.per_class_loss
        if pcl["pos"] is not None:
            self._loss_pos.append(pcl["pos"])
        if pcl["neg"] is not None:
            self._loss_neg.append(pcl["neg"])

    @property
    def running_label_rate(self) -> float:
        """Running label positive rate across all steps seen."""
        if self._running_count == 0:
            return 0.0
        return self._running_sum / self._running_count

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        if phase == "step":
            rate = self.running_label_rate
            _, step = self._tb_step(phase, ctx)
            if self.writer:
                self.writer.add_scalar("DataHealth/running_label_rate", rate, step)
            return [f"running_label_rate={rate:.4f}"]

        if not self._rates:
            return []
        rates = np.array(self._rates)
        parts = [f"mean={rates.mean():.4f},var={rates.var():.6f}"]
        if self._loss_pos:
            parts.append(f"loss_pos={np.mean(self._loss_pos):.4f}")
        if self._loss_neg:
            parts.append(f"loss_neg={np.mean(self._loss_neg):.4f}")
        return [",".join(parts)]

    def flush(self) -> None:
        """Clear windowed accumulators after emission."""
        self._rates.clear()
        self._loss_pos.clear()
        self._loss_neg.clear()

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        self._rates.clear()
        self._loss_pos.clear()
        self._loss_neg.clear()

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                accum[k] = _try_numeric(v)


class OobCode(DiagBase):
    """Out-of-bounds feature stats."""

    code = "OOB"
    config_key = "oob"
    emit = frozenset({"epoch"})
    accumulate = frozenset()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        if not isinstance(ctx, EpochContext):
            return ["clean=true"]
        oob_stats = ctx.oob_stats
        if not oob_stats:
            return ["clean=true"]
        return [json.dumps(oob_stats, separators=(",", ":"))]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        if payload == "clean=true":
            accum["clean"] = True
        else:
            accum["stats"] = json.loads(payload)
