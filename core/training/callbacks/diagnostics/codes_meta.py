"""Preamble and completion codes: ENV, MODEL, DATASET, SCHEMA, DONE."""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from core.config.loader import to_dict
from core.training.callbacks.diagnostics.base import DiagBase, _try_numeric
from core.training.callbacks.diagnostics.context import EpochContext, StepContext


class EnvCode(DiagBase):
    """Environment and config snapshot."""

    code = "ENV"
    config_key = "env"
    emit = frozenset({"preamble"})
    accumulate = frozenset()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        seed = ctx["seed"]
        config = ctx["config"]
        parts = [f"torch={torch.__version__}"]
        if torch.cuda.is_available():
            parts.append(f"cuda={torch.version.cuda}")
            parts.append(f"gpu={torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            mem = (getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)) / (
                1024**3
            )
            parts.append(f"vram_gb={mem:.2f}")
        parts.append(f"seed={seed}")
        payloads = [",".join(parts)]
        if config:
            data = to_dict(config) if isinstance(config, DictConfig) else config
            payloads.append(f"config:{json.dumps(data, separators=(',', ':'))}")
        return payloads

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        if payload.startswith("config:"):
            raw = payload[7:]
            try:
                accum.setdefault("config", json.loads(raw))
            except (json.JSONDecodeError, ValueError):
                accum.setdefault("config", raw)
        else:
            for kv in payload.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    accum[k] = v


class ModelCode(DiagBase):
    """Model architecture and parameter counts."""

    code = "MODEL"
    config_key = "model"
    emit = frozenset({"preamble"})
    accumulate = frozenset()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        model: nn.Module = ctx["model"]
        total = sum(p.numel() for p in model.parameters())
        emb = 0
        tables: list[list[Any]] = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Embedding):
                n = mod.weight.numel()
                emb += n
                tables.append([name, list(mod.weight.shape)])
        dense = total - emb
        payloads = [f"total={total},emb={emb},dense={dense},tables={len(tables)}"]
        payloads.append(f"emb_full:{json.dumps(tables, separators=(',', ':'))}")
        return payloads

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        if payload.startswith("emb_full:"):
            accum["emb_full"] = json.loads(payload[9:])
        else:
            for kv in payload.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    accum[k] = _try_numeric(v)


class DatasetCode(DiagBase):
    """Dataset file inventory."""

    code = "DATASET"
    config_key = "dataset"
    emit = frozenset({"preamble"})
    accumulate = frozenset()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        import pyarrow.parquet as pq

        data_dir: str = ctx["data_dir"]
        pq_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        rgs = 0
        rows = 0
        total_bytes = 0
        for f in pq_files:
            meta = pq.ParquetFile(f).metadata
            for i in range(meta.num_row_groups):
                rg = meta.row_group(i)
                rgs += 1
                rows += rg.num_rows
                total_bytes += sum(
                    rg.column(c).total_compressed_size for c in range(rg.num_columns)
                )
        return [
            f"path={data_dir},files={len(pq_files)},rows={rows},row_groups={rgs},bytes={total_bytes}"
        ]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                accum[k] = _try_numeric(v)


class SchemaCode(DiagBase):
    """Feature schema snapshot."""

    code = "SCHEMA"
    config_key = "schema"
    emit = frozenset({"preamble"})
    accumulate = frozenset()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        schema_path: str = ctx["schema_path"]
        with open(schema_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [json.dumps(raw, separators=(",", ":"))]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        accum["schema"] = json.loads(payload)


class DoneCode(DiagBase):
    """Training completion summary."""

    code = "DONE"
    config_key = "done"
    emit = frozenset({"done"})
    accumulate = frozenset()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Collect payload strings for this code."""
        parts = [
            f"best_auc={ctx.get('best_auc', 0.0):.6f}",
            f"best_epoch={ctx.get('best_epoch', 0)}",
            f"wall_sec={ctx.get('wall_sec', 0.0):.0f}",
            f"samples={ctx.get('samples', 0)}",
            f"early_stop={str(ctx.get('early_stop', False)).lower()}",
        ]
        return [",".join(parts)]

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse a payload segment into the accumulator."""
        for kv in payload.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                if v == "true":
                    accum[k] = True
                elif v == "false":
                    accum[k] = False
                else:
                    accum[k] = _try_numeric(v)
