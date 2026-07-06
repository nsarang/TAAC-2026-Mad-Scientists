"""GDCN-instrumentation codes: GDCN_GATE (field-wise gate vectors) and
GDCN_CROSS (field-pair cross-interaction strength).

Both operate at the granularity of individual fields (each of the ~90 raw
features), not hierarchy groups -- aggregation/normalization are left to the
offline step. Each emits a one-shot field manifest (per-field names, and dims
for the cross matrix) so the log is self-describing. They are no-ops when the
model has no ``gdcn_source``.

GDCN_GATE captures each gated cross layer's gate vector via a forward hook on
its gate projection and averages within each field -> a per-layer, per-field
emphasis vector, emitted every epoch.

GDCN_CROSS reads the first cross layer's weight, partitions it into
field-by-field blocks, and reports every block's Frobenius norm -- the full
per-field N x N interaction matrix, every epoch. The matrix is optionally
uint16+zlib+base64 compressed (default on) to keep the per-epoch log small.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch import nn

from core.training.callbacks.diagnostics.base import DiagBase, _parse_context
from core.training.callbacks.diagnostics.codec import decode_array, encode_array
from core.training.callbacks.diagnostics.context import EpochContext, StepContext


def _gdcn_source(model: nn.Module) -> nn.Module | None:
    src = getattr(model, "gdcn_source", None)
    return src if src is not None else None


def _unwrap(module: nn.Module) -> nn.Module:
    """Return the eager module behind a torch.compile OptimizedModule wrapper."""
    return getattr(module, "_orig_mod", module)


# ─────────────────────────────────────────────────────────────────────────────
# GDCN_GATE — per-field instance-gate emphasis per cross layer
# ─────────────────────────────────────────────────────────────────────────────


class GdcnGateCode(DiagBase):
    """Per-layer, per-field mean of the GDCN instance gate (every epoch).

    Higher values mean the layer's gate keeps (emphasises) that field; lower
    means it suppresses it. Emits one ``gate:l{layer}`` vector of length N per
    layer plus a ``fields`` manifest of field names.
    """

    code = "GDCN_GATE"
    config_key = "gdcn_gate"
    emit = frozenset({"epoch"})
    accumulate = frozenset({"always"})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._offsets: list[int] = []
        self._names: list[str] = []
        self._groups: list[str] = []
        self._per_field: dict[int, list[np.ndarray]] = {}
        self._handles: list[Any] = []

    def register_hooks(self, model: nn.Module) -> None:
        """Hook each cross layer's gate projection (pre-sigmoid)."""
        src = _gdcn_source(model)
        if src is None:
            return
        layout = src.field_layout()
        self._offsets = layout["offsets"]
        self._names = layout["names"]
        self._groups = layout["groups"]
        net = _unwrap(src.network)
        for i, layer in enumerate(net.layers):
            gate_mod = getattr(layer, "gate_V", None) or getattr(layer, "gate_W", None)
            if gate_mod is None:
                # cross_experts > 1 has no single gate projection to tap.
                continue
            self._per_field[i] = []
            self._handles.append(
                gate_mod.register_forward_hook(self._timed_hook(self._make_hook(i)))
            )

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, layer_idx: int) -> Any:
        offsets = self._offsets
        store = self._per_field

        @torch.compiler.disable
        def hook(mod: nn.Module, args: tuple, output: Any) -> None:
            gate = torch.sigmoid(output.detach().float())  # [B, D] pre-sigmoid -> gate
            means = np.array(
                [
                    gate[:, offsets[f] : offsets[f + 1]].mean().item()
                    for f in range(len(offsets) - 1)
                ]
            )
            store[layer_idx].append(means)

        return hook

    def epoch_reset(self) -> None:
        """Reset per-epoch accumulators."""
        for v in self._per_field.values():
            v.clear()

    def flush(self) -> None:
        """Clear windowed accumulators."""
        for v in self._per_field.values():
            v.clear()

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Average the gate over the epoch and emit a per-field vector per layer."""
        if not self._per_field:
            return []
        payloads: list[str] = []
        for layer_idx, frames in sorted(self._per_field.items()):
            if not frames:
                continue
            field_mean = np.stack(frames).mean(axis=0)  # [n_fields]
            payloads.append(
                f"gate:l{layer_idx}:"
                + json.dumps([round(float(v), 5) for v in field_mean], separators=(",", ":"))
            )
        if not payloads:
            return []
        payloads.insert(
            0,
            "fields:"
            + json.dumps({"names": self._names, "groups": self._groups}, separators=(",", ":")),
        )
        return payloads

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse the field manifest and per-layer gate vectors."""
        ctx = _parse_context(context)
        epoch = ctx.get("epoch", ctx.get("step", 0))
        if payload.startswith("fields:"):
            accum["fields"] = json.loads(payload[len("fields:") :])
        elif payload.startswith("gate:"):
            _, layer, vec = payload.split(":", 2)
            accum.setdefault("epochs", {}).setdefault(epoch, {})[layer] = json.loads(vec)


# ─────────────────────────────────────────────────────────────────────────────
# GDCN_CROSS — field-pair cross-interaction strength (first layer)
# ─────────────────────────────────────────────────────────────────────────────


def _first_layer_cross_matrix(layer: nn.Module) -> torch.Tensor | None:
    """Reconstruct the [D, D] effective cross weight of one GDCN layer."""
    if hasattr(layer, "cross_W"):
        return layer.cross_W.weight.detach()
    if hasattr(layer, "cross_V"):
        return (layer.cross_V.weight @ layer.cross_U.weight).detach()
    if hasattr(layer, "cross_W_experts"):
        return torch.stack([m.weight.detach() for m in layer.cross_W_experts]).mean(0)
    if hasattr(layer, "cross_V_experts"):
        mats = [
            (v.weight @ u.weight).detach()
            for u, v in zip(layer.cross_U_experts, layer.cross_V_experts)
        ]
        return torch.stack(mats).mean(0)
    return None


class GdcnCrossCode(DiagBase):
    """Full per-field field-pair interaction matrix from the first GDCN layer.

    Reconstructs the first layer's ``[D, D]`` cross matrix and reports the
    Frobenius norm of every field-by-field block -- the raw N x N matrix over
    all fields, no hierarchy aggregation, every epoch. Emits a ``fields``
    manifest (per-field name and dim) so grouping and dimension normalization
    can be done offline. The matrix is uint16+zlib+base64 compressed when
    ``compress`` is set (default), else emitted as plain JSON.
    """

    code = "GDCN_CROSS"
    config_key = "gdcn_cross"
    emit = frozenset({"epoch"})
    accumulate = frozenset()

    def __init__(self, compress: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._compress = compress

    def collect(self, phase: str, ctx: StepContext | EpochContext | dict[str, Any]) -> list[str]:
        """Emit the full per-field Frobenius interaction matrix plus a manifest."""
        model = ctx.get("model") if isinstance(ctx, dict) else getattr(ctx, "model", None)
        if model is None:
            return []
        src = _gdcn_source(model)
        if src is None:
            return []
        net = _unwrap(src.network)
        if not getattr(net, "layers", None):
            return []
        weight = _first_layer_cross_matrix(net.layers[0])
        if weight is None:
            return []

        layout = src.field_layout()
        offsets = layout["offsets"]
        w = weight.float().cpu().numpy()
        n = len(offsets) - 1
        field_frob = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                block = w[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]]
                field_frob[i, j] = float(np.linalg.norm(block))

        payloads = [
            "fields:"
            + json.dumps(
                {"names": layout["names"], "groups": layout["groups"], "dims": layout["dims"]},
                separators=(",", ":"),
            )
        ]
        # Chunk rows so each payload stays under the DIAG line cap. A row costs
        # ~2 bytes/value compressed (uint16+zlib+base64) but ~12 chars/value as
        # plain JSON, so the row budget must differ by mode. Floor of 1 row
        # keeps a single field row (< line cap for any realistic field count).
        max_rows = max(1, 6000 // (n * 2)) if self._compress else max(1, 9000 // (n * 12))
        for c, start in enumerate(range(0, n, max_rows)):
            block = field_frob[start : start + max_rows]
            if self._compress:
                payloads.append(f"frobz:{c}:" + encode_array(block, "uint16"))
            else:
                payloads.append(
                    f"frob:{c}:" + json.dumps(block.round(6).tolist(), separators=(",", ":"))
                )
        return payloads

    @staticmethod
    def parse(payload: str, context: str, accum: dict) -> None:
        """Parse the field manifest and the per-epoch interaction matrix."""
        ctx = _parse_context(context)
        epoch = ctx.get("epoch", 0)
        if payload.startswith("fields:"):
            accum["fields"] = json.loads(payload[len("fields:") :])
        elif payload.startswith("frobz:"):
            _, chunk, blob = payload.split(":", 2)
            accum.setdefault("epochs", {}).setdefault(epoch, {})[int(chunk)] = decode_array(blob)
        elif payload.startswith("frob:"):
            _, chunk, rows = payload.split(":", 2)
            accum.setdefault("epochs", {}).setdefault(epoch, {})[int(chunk)] = np.array(
                json.loads(rows)
            )
