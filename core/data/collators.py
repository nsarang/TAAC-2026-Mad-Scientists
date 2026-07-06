"""Terminal block that converts numpy batch dicts to model-ready tensors.

Single Collator class, parametric on format:
- "padded": pads per-feature arrays to batch max length → [B, max_L]
- "flat": concatenates per-feature arrays → [total_len]

Acts as the terminal block in the pipeline. Takes a numpy batch dict
(with per-feature seq arrays) and returns a torch tensor dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from core.data.blocks import BatchTransform
from core.data.schema import FeatureSchema


class Collator(BatchTransform):
    """Format-parametric collator. Terminal block."""

    type_key = "collate"

    def __init__(self, schema: FeatureSchema, format: str = "padded") -> None:
        self._schema = schema
        self._format = format

        # Cache everything needed for __call__
        self._static_keys = schema.static_batch_keys()
        self._domains = sorted({s.domain for s in schema.query("scope = 'seq'")})
        self._domain_feats: dict[str, list[str]] = {}
        for d in self._domains:
            self._domain_feats[d] = [
                s.batch_key for s in schema.query(f"scope = 'seq' and domain = '{d}'")
            ]

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Convert numpy batch dict to model-ready torch tensors."""
        result: dict[str, Any] = {}

        for key in self._static_keys:
            if key in batch:
                result[key] = torch.from_numpy(batch[key])

        for domain in self._domains:
            lengths = batch[f"{domain}_len"]
            B = len(lengths)

            if B == 0:
                continue

            if self._format == "padded":
                max_len = max(1, int(lengths.max()))
                total = int(lengths.sum())
                # Bool mask built once per domain, reused across features
                col_idx = np.arange(max_len, dtype=np.int32)
                mask = col_idx[np.newaxis, :] < lengths[:, np.newaxis]
                for feat_key in self._domain_feats[domain]:
                    arrays = batch[feat_key]
                    dt = arrays[0].dtype
                    padded = np.zeros((B, max_len), dtype=dt)
                    if total > 0:
                        padded[mask] = np.concatenate(arrays)
                    result[feat_key] = torch.from_numpy(padded)
            else:
                total_len = int(lengths.sum())
                for feat_key in self._domain_feats[domain]:
                    arrays = batch[feat_key]
                    if total_len > 0:
                        result[feat_key] = torch.from_numpy(np.concatenate(arrays))
                    else:
                        result[feat_key] = torch.zeros(0, dtype=torch.int32)

            result[f"{domain}_len"] = torch.from_numpy(lengths)

        # Pass through non-feature arrays (label, impression timestamp, etc.)
        # These aren't registered in FeatureSchema — they're targets/metadata.
        for key, val in batch.items():
            if key not in result and isinstance(val, np.ndarray):
                result[key] = torch.from_numpy(val)

        return result

    def compute(self, batch: dict[str, Any]) -> None:
        """Convert numpy batch to model-ready torch tensors.

        Reads
        -----
        All keys produced by upstream blocks (static arrays + seq lists).

        Writes
        ------
        Replaces entire batch contents with torch tensors. Static keys
        become ``[B, ...]`` tensors; sequence keys become padded or flat
        tensors depending on format.
        """
        result = self(batch)
        batch.clear()
        batch.update(result)
