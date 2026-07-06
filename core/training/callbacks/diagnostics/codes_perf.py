"""Performance codes: TPT (throughput), TIMING."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from core.training.callbacks.diagnostics.base import DiagBase, _parse_context, _try_numeric
from core.training.callbacks.diagnostics.context import EpochContext, StepContext


class ThroughputCode(DiagBase):
    """Samples-per-second throughput and timing breakdown (averaged over window)."""

    code = "TPT"
    config_key = "throughput"
    emit = frozenset({"step", "epoch"})
    accumulate = frozenset({"always"})

    _FIELDS = ("samples", "data_time", "fwd_time", "bwd_time", "total_time")
    _IO_FIELDS = ("io_ext_ms", "asm_ms", "n_rgs", "rows_decomp", "unique_rgs")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._buf: dict[str, list[float]] = {f: [] for f in self._FIELDS}
        self._io_buf: dict[str, list[float]] = {f: [] for f in self._IO_FIELDS}

    def step(self, ctx: StepContext, emit: bool = False) -> None:
        """Append per-step timing to buffers.

        step_start_time is stamped before data loading, so (now - step_start)
        already includes data_time, fwd, bwd, and diagnostics overhead.
        """
        now = time.perf_counter()
        total = now - ctx.step_start_time
        self._buf["samples"].append(ctx.batch_size)
        self._buf["data_time"].append(ctx.data_time)
        self._buf["fwd_time"].append(ctx.fwd_time or 0.0)
        self._buf["bwd_time"].append(ctx.bwd_time or 0.0)
        self._buf["total_time"].append(total)

        meta = ctx.batch.get("_meta_timing")
        if meta is not None:
            v = meta.float()
            self._io_buf["io_ext_ms"].append(v[0].item())
            self._io_buf["asm_ms"].append(v[1].item())
            self._io_buf["n_rgs"].append(v[3].item())
            self._io_buf["rows_decomp"].append(v[4].item())
            self._io_buf["unique_rgs"].append(v[5].item())

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Emit averaged throughput and GPU stats."""
        if phase == "epoch":
            return self._collect_epoch(ctx)
        step = ctx.step
        n = len(self._buf["total_time"])
        data_s = sum(self._buf["data_time"]) / n
        fwd_s = sum(self._buf["fwd_time"]) / n
        bwd_s = sum(self._buf["bwd_time"]) / n
        total_s = sum(self._buf["total_time"]) / n
        avg_samples = sum(self._buf["samples"]) / n
        sps = avg_samples / total_s if total_s > 0 else 0.0
        st_ps = 1.0 / total_s if total_s > 0 else 0.0

        misc_s = max(0.0, total_s - data_s - fwd_s - bwd_s)

        parts = [
            f"sps={sps:.1f}",
            f"st_ps={st_ps:.2f}",
            f"data_s={data_s:.4f}",
            f"fwd_s={fwd_s:.4f}",
            f"bwd_s={bwd_s:.4f}",
            f"misc_s={misc_s:.4f}",
            f"total_s={total_s:.4f}",
        ]
        if torch.cuda.is_available():
            alloc_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            sm_util = torch.cuda.utilization(0)
            mem_util = torch.cuda.memory_usage(0)
            parts += [
                f"gpu_alloc_gb={alloc_gb:.2f}",
                f"gpu_reserved_gb={reserved_gb:.2f}",
                f"gpu_sm_util={sm_util}",
                f"gpu_mem_util={mem_util}",
            ]
            if self.writer:
                self.writer.add_scalar("GPU/memory_allocated_gb", alloc_gb, step)
                self.writer.add_scalar("GPU/memory_reserved_gb", reserved_gb, step)
                self.writer.add_scalar("GPU/sm_utilization", sm_util, step)
                self.writer.add_scalar("GPU/mem_utilization", mem_util, step)
        if self.writer:
            self.writer.add_scalar("Throughput/samples_per_sec", sps, step)
            self.writer.add_scalar("Throughput/steps_per_sec", st_ps, step)
            self.writer.add_scalar("Throughput/data_sec", data_s, step)
            self.writer.add_scalar("Throughput/fwd_sec", fwd_s, step)
            self.writer.add_scalar("Throughput/bwd_sec", bwd_s, step)
            self.writer.add_scalar("Throughput/misc_sec", misc_s, step)
            self.writer.add_scalar("Throughput/total_sec", total_s, step)

        # DataIO metrics from worker-side _meta_timing
        io_buf = self._io_buf
        if io_buf["io_ext_ms"]:
            io_ms = np.mean(io_buf["io_ext_ms"])
            asm_ms = np.mean(io_buf["asm_ms"])
            avg_rgs = np.mean(io_buf["n_rgs"])
            avg_samples = sum(self._buf["samples"]) / n
            avg_decomp = np.mean(io_buf["rows_decomp"])
            util_pct = avg_samples / avg_decomp * 100 if avg_decomp > 0 else 0
            max_unique = max(io_buf["unique_rgs"])
            parts += [
                f"io_ext_ms={io_ms:.1f}",
                f"asm_ms={asm_ms:.1f}",
                f"n_rgs={avg_rgs:.1f}",
                f"util={util_pct:.0f}%",
                f"unique_rgs={max_unique:.0f}",
            ]
            if self.writer:
                self.writer.add_scalar("Throughput/io_extract_ms", io_ms, step)
                self.writer.add_scalar("Throughput/io_assemble_ms", asm_ms, step)
                self.writer.add_scalar("Throughput/io_rgs_per_batch", avg_rgs, step)
                self.writer.add_scalar("Throughput/io_row_util_pct", util_pct, step)
                self.writer.add_scalar("Throughput/io_unique_rgs", max_unique, step)

        return [",".join(parts)]

    def _collect_epoch(self, ctx: EpochContext) -> list[str]:
        """Emit val throughput breakdown at epoch end (per-batch averages)."""
        if ctx.val_time <= 0 or ctx.n_val_batches <= 0:
            return []
        n_samples = len(ctx.val_probs) if ctx.val_probs is not None else 0
        if n_samples == 0:
            return []
        n = ctx.n_val_batches
        val_sps = n_samples / ctx.val_time
        val_data_s = ctx.val_data_time / n
        val_fwd_s = ctx.val_fwd_time / n
        parts = [
            f"val_sps={val_sps:.1f}",
            f"val_data_s={val_data_s:.4f}",
            f"val_fwd_s={val_fwd_s:.4f}",
        ]
        if self.writer:
            self.writer.add_scalar("Throughput/val_samples_per_sec", val_sps, ctx.epoch)
            self.writer.add_scalar("Throughput/val_data_sec", val_data_s, ctx.epoch)
            self.writer.add_scalar("Throughput/val_fwd_sec", val_fwd_s, ctx.epoch)
        return [",".join(parts)]

    def flush(self) -> None:
        """Clear all timing buffers after emission."""
        for lst in self._buf.values():
            lst.clear()
        for lst in self._io_buf.values():
            lst.clear()

    def epoch_reset(self) -> None:
        """Reset buffers at epoch boundary."""
        self.flush()

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


class TimingCode(DiagBase):
    """Forward/backward/collation timing over the warmup window."""

    code = "TIMING"
    config_key = "timing"
    emit = frozenset({"warmup"})
    accumulate = frozenset({"warmup"})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fwd: list[float] = []
        self._bwd: list[float] = []
        self._col: list[float] = []

    def step(self, ctx: StepContext, emit: bool = False) -> None:
        """Accumulate per-step timing data."""
        if ctx.fwd_time is not None:
            self._fwd.append(ctx.fwd_time)
        if ctx.bwd_time is not None:
            self._bwd.append(ctx.bwd_time)
        # Collation = total step minus data, fwd, bwd (diagnostics overhead etc.)
        now = time.perf_counter()
        total = now - ctx.step_start_time
        col = total - ctx.data_time - (ctx.fwd_time or 0.0) - (ctx.bwd_time or 0.0)
        if col > 0:
            self._col.append(col)

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code. Idempotent -- does not clear."""
        parts: list[str] = []
        if self._fwd:
            parts.append(f"fwd_ms={np.mean(self._fwd) * 1000:.1f}")
        if self._bwd:
            parts.append(f"bwd_ms={np.mean(self._bwd) * 1000:.1f}")
        if self._col:
            parts.append(f"col_ms={np.mean(self._col) * 1000:.1f}")
        if torch.cuda.is_available():
            parts.append(f"peak_gpu_gb={torch.cuda.max_memory_allocated() / (1024**3):.2f}")
        return [",".join(parts)] if parts else []

    def flush(self) -> None:
        """Clear windowed accumulators after emission."""
        self._fwd.clear()
        self._bwd.clear()
        self._col.clear()

    def epoch_reset(self) -> None:
        """Clear all buffers at epoch start."""
        self._fwd.clear()
        self._bwd.clear()
        self._col.clear()

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                accum[k] = _try_numeric(v)
