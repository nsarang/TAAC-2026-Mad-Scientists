"""Feature blocks for dataset v2.

Single ABC with __init_subclass__ auto-registration keyed by `type_key`.
Blocks that need a pre-scan fitting phase implement `fit_columns`,
`partial_fit`, and `finish_fit`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from core.data.schema import Dtype, Entity, FeatureSchema, FeatureSpec, Source


class BatchTransform(ABC):
    """Produces or transforms features from batch context.

    Subclasses that need a pre-scan fitting phase override `fit_columns`,
    `partial_fit`, and `finish_fit`. The dataset detects blocks with
    non-empty `fit_columns` and runs them through the shared I/O scan
    before any `compute` calls happen.
    """

    registry: ClassVar[dict[str, type[BatchTransform]]] = {}
    type_key: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.type_key:
            BatchTransform.registry[cls.type_key] = cls

    def output_specs(self) -> list[FeatureSpec]:
        """Declare features this block produces (with layout fields set)."""
        return []

    def fit_columns(self) -> list[str]:
        """Parquet columns needed for fitting. Empty means no fit needed."""
        return []

    @property
    def fit_saturated(self) -> bool:
        """True when this block has collected enough data and needs no more I/O."""
        return False

    def partial_fit(self, batch: pa.RecordBatch) -> None:
        """Process one RecordBatch during the pre-scan."""
        return

    def finish_fit(self) -> None:
        """Compute final statistics after all RGs have been consumed."""
        return

    def fit_state(self) -> dict[str, Any] | None:
        """Return fitted state for checkpoint serialization. None if unfitted."""
        return None

    def load_fit_state(self, state: dict[str, Any]) -> None:  # noqa: B027
        """Restore fitted state from a checkpoint, bypassing the I/O scan."""

    @abstractmethod
    def compute(self, batch: dict[str, Any]) -> None:
        """Compute or transform features in the batch dict.

        Mutates `batch` in-place: reads input keys, writes output keys.
        Subclass docstrings document which keys are read and written.
        """


class RSSCBlock(BatchTransform):
    """Robust-scale with smooth clipping for dense continuous features.

    Applies `z / sqrt(1 + (z/clip)^2)` where `z = (x - median) / IQR`.
    Output is bounded in `(-clip, +clip)` with smooth saturation.

    Fitting phase scans training data to compute per-column median and IQR.
    Overwrites the targeted columns in-place via `FeatureSchema.update`.

    Parameters
    ----------
    schema
        FeatureSchema — used to discover and locate target features.
    clip_value
        Soft saturation bound.
    pre_transform
        Variance-stabilizing transform applied before robust scaling.
        ``"log1p"`` — log1p(max(x, 0)), good for large positive skew.
        ``"asinh"`` — arcsinh(x), handles full real line gracefully.
        None — no pre-transform (raw values).
    pattern
        DSL filter expression selecting features to normalize.
        Defaults to all static numerical user features.
    max_rows
        Cap on rows collected during fit (memory bound).
    """

    type_key = "rssc"

    _DEFAULT_EXPR = (
        "entity = 'user' and dtype = 'numerical' and scope = 'static' and source != 'metadata'"
    )

    _VALID_TRANSFORMS = (None, "log1p", "asinh")

    def __init__(
        self,
        schema: "FeatureSchema",
        clip_value: float = 3.0,
        pre_transform: str = None,
        pattern: str = None,
        max_rows: int = 5_000_000,
    ) -> None:
        if pre_transform not in self._VALID_TRANSFORMS:
            raise ValueError(
                f"pre_transform must be one of {self._VALID_TRANSFORMS}, got {pre_transform!r}"
            )
        self._schema = schema
        self._clip_value = clip_value
        self._pre_transform = pre_transform
        self._max_rows = max_rows

        # Resolve pattern: if it looks like a glob (no operators), wrap in name matches
        if pattern is not None:
            if any(kw in pattern for kw in ("=", "!=", " and ", " or ", " in ", " matches ")):
                self._expr = pattern
            else:
                self._expr = (
                    f"name matches '{pattern}' and scope = 'static' and source != 'metadata'"
                )
        else:
            self._expr = self._DEFAULT_EXPR

        self._specs = schema.query(self._expr)
        self._total_dim = sum(s.dim for s in self._specs)

        self._fit_col_names = [s.source_col for s in self._specs]

        # Fit state
        self._collected: list[np.ndarray] = []
        self._rows_seen = 0

        # Set after finish_fit
        self._median: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit_columns(self) -> list[str]:
        """Parquet columns needed for median/IQR computation."""
        return self._fit_col_names

    @property
    def fit_saturated(self) -> bool:
        """Whether enough rows have been collected for statistics."""
        return self._rows_seen >= self._max_rows

    def partial_fit(self, batch: pa.RecordBatch) -> None:
        """Accumulate raw values for robust-scaling statistics."""
        if self._rows_seen >= self._max_rows:
            return
        B = batch.num_rows
        buf = np.zeros((B, self._total_dim), dtype=np.float32)
        row_idx = np.arange(B, dtype=np.intp)
        offset = 0
        for spec, col_name in zip(self._specs, self._fit_col_names):
            dim = spec.dim
            col_idx = batch.schema.get_field_index(col_name)
            if col_idx < 0:
                offset += dim
                continue
            col = batch.column(col_idx)
            offs = col.offsets.to_numpy()
            vals = col.values.to_numpy()
            if len(vals) == 0:
                offset += dim
                continue
            # source_offset handles split features that start mid-column
            src_off = spec.source_offset
            starts = offs[row_idx] + src_off
            lengths = offs[row_idx + 1] - offs[row_idx] - src_off
            use = np.minimum(np.maximum(lengths, 0), dim)
            idx_2d = starts[:, None] + np.arange(dim)[None, :]
            mask = np.arange(dim)[None, :] < use[:, None]
            idx_2d = np.where(mask, idx_2d, 0)
            chunk = vals[idx_2d]
            chunk[~mask] = 0.0
            buf[:, offset : offset + dim] = chunk
            offset += dim

        self._apply_pre_transform(buf)
        self._collected.append(buf)
        self._rows_seen += B

    def finish_fit(self) -> None:
        """Compute per-column median and IQR from accumulated data."""
        if not self._collected:
            raise RuntimeError("RSSCBlock: no data collected during fit")

        all_data = np.concatenate(self._collected, axis=0)
        self._collected.clear()

        quantiles = np.percentile(all_data, [25, 50, 75], axis=0)
        q25 = quantiles[0]
        median = quantiles[1].astype(np.float32)
        q75 = quantiles[2]
        iqr = q75 - q25

        EPS = 1e-6
        scale = np.where(iqr > EPS, iqr, 0.0)
        for j in range(self._total_dim):
            if scale[j] == 0.0:
                range_j = float(all_data[:, j].max()) - float(all_data[:, j].min())
                fallback = 0.5 * range_j
                if fallback > EPS:
                    scale[j] = fallback
                else:
                    scale[j] = 1.0

        self._median = median
        self._scale = scale.astype(np.float32)

    def fit_state(self) -> dict[str, Any] | None:
        """Serialize median and scale arrays for checkpointing."""
        if self._median is None:
            return None
        return {
            "median": self._median.tolist(),
            "scale": self._scale.tolist(),
        }

    def load_fit_state(self, state: dict[str, Any]) -> None:
        """Restore median/scale from checkpoint."""
        self._median = np.array(state["median"], dtype=np.float32)
        self._scale = np.array(state["scale"], dtype=np.float32)
        self._rows_seen = self._max_rows

    def _apply_pre_transform(self, arr: np.ndarray) -> None:
        """Apply the configured variance-stabilizing transform in-place."""
        if self._pre_transform == "log1p":
            np.maximum(arr, 0, out=arr)
            np.log1p(arr, out=arr)
        elif self._pre_transform == "asinh":
            np.arcsinh(arr, out=arr)

    def compute(self, batch: dict[str, Any]) -> None:
        """Apply log1p (if enabled) + robust-scale + smooth-clip.

        Reads
        -----
        Columns selected by ``self._expr`` from their parent batch key
        (e.g. ``user_cont[..., 2:5]`` or ``item_cont[..., 0:4]``).

        Writes
        ------
        Same columns overwritten in-place with normalized values bounded
        in ``(-clip_value, +clip_value)``.
        """
        data = self._schema.extract(batch, expr=self._expr, cat=True)
        if data is None:
            raise RuntimeError(
                f"RSSCBlock: no features matched expr {self._expr!r}. "
                f"Check that the pattern selects registered features."
            )
        out = data.copy()
        self._apply_pre_transform(out)
        z = (out - self._median) / self._scale
        out = z / np.sqrt(1.0 + (z / self._clip_value) ** 2)
        self._schema.update(batch, self._expr, out)


class DenseSeqStatsBlock(BatchTransform):
    """Per-domain sequence summary statistics as dense features.

    Configurable set of features per domain. With defaults (all add_* False),
    produces 2 dims/domain (presence + log_len) matching the original behavior.
    Enable additional flags for richer representations.

    Parameters
    ----------
    schema
        FeatureSchema — used to discover sequence domains.
    add_seq_len_bucket_norm
        Emit normalized bucket index of sequence length.
    add_last_gap_log
        Emit log-normalized seconds since last event.
    add_gap_missing
        Emit 1.0 when timestamp data is unavailable.
    add_recent_flags
        Emit binary flags for gap <= each threshold.
    recent_thresholds_sec
        Time thresholds in seconds for recent flags.
    len_bucket_edges
        Edges for length bucketing.
    max_gap_sec
        Gap values clamped to this before log transform.
    empty_gap_value
        Value written to gap_missing column when ts unavailable.
    """

    type_key = "dense_seq_stats"

    def __init__(
        self,
        schema: "FeatureSchema",
        add_seq_len_bucket_norm: bool = False,
        add_last_gap_log: bool = False,
        add_gap_missing: bool = False,
        add_recent_flags: bool = False,
        recent_thresholds_sec: list[int] = None,
        len_bucket_edges: list[int] = None,
        max_seq_len_norm: int = 4000,
        max_gap_sec: int = 31536000,
        empty_gap_value: float = 1.0,
    ) -> None:
        self._domains = schema.seq_domains
        self._add_seq_len_bucket_norm = add_seq_len_bucket_norm
        self._add_last_gap_log = add_last_gap_log
        self._add_gap_missing = add_gap_missing
        self._add_recent_flags = add_recent_flags
        self._thresholds = sorted(recent_thresholds_sec or [300, 3600, 86400])
        self._bucket_edges = np.asarray(
            len_bucket_edges or [1, 10, 50, 100, 200, 500, 1000], dtype=np.int64
        )
        self._bucket_den = float(max(len(self._bucket_edges), 1))
        self._max_seq_len_norm = float(max(1, max_seq_len_norm))
        self._len_log_den = math.log1p(self._max_seq_len_norm)
        self._max_gap_sec = max(1, max_gap_sec)
        self._gap_log_den = math.log1p(self._max_gap_sec)
        self._empty_gap_value = empty_gap_value

    @property
    def _dim_per_domain(self) -> int:
        d = 3  # presence + empty_flag + len_log always
        if self._add_seq_len_bucket_norm:
            d += 1
        if self._add_last_gap_log:
            d += 1
        if self._add_gap_missing:
            d += 1
        if self._add_recent_flags:
            d += len(self._thresholds)
        return d

    def output_specs(self) -> list[FeatureSpec]:
        """Output features.

        dense_seq_stats : float32, shape `[B, dim_per_domain * N_domains]`
            Per-domain columns in order: presence, empty_flag, len_log,
            [len_bucket_norm], [last_gap_log], [gap_missing], [recent_flags...].
            Length is `log1p(min(len, max_seq_len_norm)) / log1p(max_seq_len_norm)`.
        """
        return [
            FeatureSpec(
                name="dense_seq_stats",
                dtype=Dtype.NUMERICAL,
                entity=Entity.USER,
                dim=self._dim_per_domain * len(self._domains),
                source=Source.DERIVED,
                batch_key="dense_seq_stats",
            )
        ]

    def compute(self, batch: dict[str, Any]) -> None:
        """Compute configurable per-domain sequence statistics.

        Reads
        -----
        {domain}_len : np.ndarray, shape ``[B]``
        {domain}_ts : list[np.ndarray], optional
        timestamp : np.ndarray, shape ``[B]``

        Writes
        ------
        dense_seq_stats : np.ndarray, shape ``[B, dim_per_domain * N_domains]``
        """
        lengths_0 = batch[f"{self._domains[0]}_len"]
        B = len(lengths_0)
        total_dim = self._dim_per_domain * len(self._domains)
        out = np.zeros((B, total_dim), dtype=np.float32)

        col = 0
        for domain in self._domains:
            lengths = batch[f"{domain}_len"].astype(np.int64)
            present = lengths > 0

            # presence
            out[:, col] = present.astype(np.float32)
            col += 1

            # empty_flag (complement of presence)
            out[:, col] = (~present).astype(np.float32)
            col += 1

            # len_log: log1p(min(len, max_seq_len_norm)) / log1p(max_seq_len_norm)
            capped = np.minimum(lengths.astype(np.float32), self._max_seq_len_norm)
            out[:, col] = np.log1p(capped) / self._len_log_den
            col += 1

            if self._add_seq_len_bucket_norm:
                bucket_idx = np.searchsorted(self._bucket_edges, lengths, side="right").astype(
                    np.float32
                )
                out[:, col] = bucket_idx / self._bucket_den
                col += 1

            # Gap-based features need timestamps
            needs_gap = self._add_last_gap_log or self._add_gap_missing or self._add_recent_flags
            if needs_gap:
                ts_key = f"{domain}_ts"
                has_ts = ts_key in batch
                imp_ts = batch.get("timestamp")

                gaps = np.full(B, -1, dtype=np.int64)
                if has_ts and imp_ts is not None:
                    ts_arrays = batch[ts_key]
                    total_len = int(lengths.sum())
                    if total_len > 0:
                        flat_ts = np.concatenate(ts_arrays).astype(np.int64)
                        flat_rows = np.repeat(np.arange(B, dtype=np.int32), lengths)
                        # Scatter-max: find max valid timestamp per row
                        # np.maximum.at is not fast, use a sort-based approach
                        # Since we need max per group, just use bincount trick:
                        # multiply by validity, take max via reduceat
                        valid = flat_ts > 0
                        if valid.any():
                            # For each row, find its max valid ts
                            # Set invalid to -1 so they don't win the max
                            max_ts = np.full(B, -1, dtype=np.int64)
                            np.maximum.at(max_ts, flat_rows[valid], flat_ts[valid])
                            has_valid = max_ts > 0
                            gaps[has_valid] = np.maximum(
                                imp_ts[has_valid].astype(np.int64) - max_ts[has_valid], 0
                            )

                if self._add_last_gap_log:
                    valid_mask = gaps >= 0
                    out[:, col] = self._empty_gap_value
                    capped_gaps = np.minimum(gaps, self._max_gap_sec).astype(np.float32)
                    out[valid_mask, col] = np.log1p(capped_gaps[valid_mask]) / self._gap_log_den
                    col += 1

                if self._add_gap_missing:
                    out[:, col] = (gaps < 0).astype(np.float32) * self._empty_gap_value
                    col += 1

                if self._add_recent_flags:
                    for thr in self._thresholds:
                        valid_mask = gaps >= 0
                        out[valid_mask, col] = (gaps[valid_mask] <= thr).astype(np.float32)
                        col += 1

        batch["dense_seq_stats"] = out


class TimeDeltaBucketBlock(BatchTransform):
    """Discretize recency of each sequence event relative to impression time.

    For each domain, computes `impression_ts - event_ts` and buckets the
    result into 63 log-spaced bins (5s granularity at the short end, up to
    1 year at the long end). Produces a per-event categorical feature that
    the model can embed alongside the event's content features.

    Parameters
    ----------
    schema
        FeatureSchema — used to discover sequence domains.
    """

    type_key = "time_bucket"

    # fmt: off
    BUCKET_BOUNDARIES: ClassVar[np.ndarray] = np.array(
        [
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
            120, 180, 240, 300, 360, 420, 480, 540, 600,
            900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
            5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
            32400, 43200, 54000, 64800, 75600, 86400,
            172800, 259200, 345600, 432000, 518400, 604800,
            1123200, 1641600, 2160000, 2592000,
            4320000, 6048000, 7776000, 11664000, 15552000, 31536000,
        ],
        dtype=np.int64,
    )
    # fmt: on

    def __init__(self, schema: "FeatureSchema") -> None:
        self._domains = schema.seq_domains

    def output_specs(self) -> list[FeatureSpec]:
        """Output features (one per domain).

        {domain}_time_bucket : int32, shape `[B, variable_seq_len]`
            Per-event bucket ID in `[0, 63]`. 0 means missing timestamp.
            Bucket 1 is the most recent (<=5s), bucket 63 is the oldest
            (<=1 year).
        """
        return [
            FeatureSpec(
                name=f"{domain}_time_bucket",
                dtype=Dtype.CATEGORICAL,
                entity=Entity.USER,
                dim=1,
                vocab_size=len(self.BUCKET_BOUNDARIES) + 1,
                domain=domain,
                source=Source.DERIVED,
                batch_key=f"{domain}_time_bucket",
            )
            for domain in self._domains
        ]

    def compute(self, batch: dict[str, Any]) -> None:
        """Bucket impression_ts - event_ts into time-delta bins.

        Reads
        -----
        timestamp : np.ndarray, shape ``[B]``
        {domain}_ts : list[np.ndarray]
        {domain}_len : np.ndarray, shape ``[B]``

        Writes
        ------
        {domain}_time_bucket : list[np.ndarray]
            Per-event bucket ID in ``[0, 63]``.
        """
        timestamp = batch["timestamp"]

        for domain in self._domains:
            ts_key = f"{domain}_ts"
            if ts_key not in batch:
                continue
            ts_arrays = batch[ts_key]
            lengths = batch[f"{domain}_len"]
            n_samples = len(ts_arrays)

            # Vectorized: concat all ts, broadcast impression timestamps,
            # single searchsorted, then split back by sample lengths.
            total_len = int(lengths.sum())
            if total_len == 0:
                batch[f"{domain}_time_bucket"] = [np.zeros(0, dtype=np.int32)] * n_samples
                continue

            all_ts = np.concatenate(ts_arrays)
            imp_ts_expanded = np.repeat(timestamp[:n_samples], lengths)
            time_diff = np.maximum(imp_ts_expanded - all_ts, 0)
            buckets = (
                np.searchsorted(self.BUCKET_BOUNDARIES, time_diff).clip(
                    0, len(self.BUCKET_BOUNDARIES) - 1
                )
                + 1
            ).astype(np.int32)
            buckets[all_ts == 0] = 0
            splits = np.cumsum(lengths[:-1])
            batch[f"{domain}_time_bucket"] = np.split(buckets, splits)


class SeqCompressBlock(BatchTransform):
    """Compress sequences via RLE or adaptive category-budget allocation.

    For RLE domains (typically B/C): collapses consecutive events with
    identical (action, item) pairs into a single representative event,
    keeping the last timestamp of each run.

    For budget domains (typically A/D): adaptively allocates the
    sequence cap across unique group-key tuples. Given a cap of C and
    protect_recent of P, the block distributes C-P slots across unique
    groups (defined by `group_key` indices), preferring events with
    diverse `prefer_diverse` feature values within each group.
    Sequences already at or below cap are left untouched.

    Emits a per-event "run bucket" categorical encoding the count of
    events collapsed into each surviving position.

    Must run before ``SeqTruncateBlock`` so truncation operates on
    already-compressed sequences.

    Parameters
    ----------
    schema
        FeatureSchema — used to discover sequence domains.
    action_slots
        Per-domain index into sideinfo features (0-based, excluding ts)
        identifying the action column for RLE identity.
    item_slots
        Per-domain index into sideinfo features identifying the item/content
        column used alongside action to define RLE identity.
    protect_recent
        Per-domain number of most-recent events to leave uncompressed.
        Events within this window are never merged. Domains not listed
        default to 0 (compress everything).
    budget_domains
        Domains using adaptive category-budget compression. Each entry
        maps domain name to a config dict with keys:

        - ``group_key``: list[int] — sideinfo slot indices that define groups
        - ``prefer_diverse``: int — sideinfo slot index to diversify within groups
        - ``cap``: int — target sequence length (same as SeqTruncateBlock ceiling)
    count_boundaries
        Boundaries for the run-count bucket. Defaults to
        [2, 5, 10, 25, 50, 100, 200] producing 9 buckets:
        0=pad, 1=single event, 2=count 2, 3=count 3-5, 4=count 6-10,
        5=count 11-25, 6=count 26-50, 7=count 51-100, 8=count 101-200,
        9=count >200.
    """

    type_key = "seq_compress"

    _DEFAULT_COUNT_BOUNDARIES: ClassVar[np.ndarray] = np.array(
        [2, 5, 10, 25, 50, 100, 200], dtype=np.int64
    )

    def __init__(
        self,
        schema: "FeatureSchema",
        action_slots: dict[str, int] = None,
        item_slots: dict[str, int] = None,
        protect_recent: dict[str, int] = None,
        budget_domains: dict[str, dict] = None,
        count_boundaries: list[int] = None,
    ) -> None:
        self._domains = schema.seq_domains
        self._action_slots = action_slots or {}
        self._item_slots = item_slots or {}
        self._protect_recent = {k: int(v) for k, v in (protect_recent or {}).items()}
        self._budget_domains = budget_domains or {}
        self._boundaries = np.asarray(
            count_boundaries or self._DEFAULT_COUNT_BOUNDARIES, dtype=np.int64
        )
        # vocab = pad(0) + single(1) + len(boundaries) + overflow(1)
        self._vocab_size = len(self._boundaries) + 2

        # Resolve sideinfo batch keys per domain from schema
        self._domain_sideinfo: dict[str, list[str]] = {}
        for domain in self._domains:
            specs = schema.query(f"scope = 'seq' and domain = '{domain}' and source = 'original'")
            self._domain_sideinfo[domain] = [s.batch_key for s in specs]

        for domain, k in self._protect_recent.items():
            if k < 0:
                raise ValueError(f"protect_recent[{domain}] must be >= 0, got {k}")
        for domain, cfg in self._budget_domains.items():
            cap = int(cfg["cap"])
            if cap < 0:
                raise ValueError(f"budget_domains[{domain}].cap must be >= 0, got {cap}")
            pr = self._protect_recent.get(domain, 0)
            if pr > cap:
                raise ValueError(
                    f"protect_recent[{domain}]={pr} cannot exceed cap={cap} "
                    f"for budget domain '{domain}'"
                )

    def output_specs(self) -> list[FeatureSpec]:
        """Declare per-domain run-count bucket categoricals."""
        specs = []
        for domain in self._domains:
            if domain not in self._action_slots and domain not in self._budget_domains:
                continue
            specs.append(
                FeatureSpec(
                    name=f"{domain}_run_bucket",
                    dtype=Dtype.CATEGORICAL,
                    entity=Entity.USER,
                    dim=1,
                    vocab_size=self._vocab_size,
                    domain=domain,
                    source=Source.DERIVED,
                    batch_key=f"{domain}_run_bucket",
                )
            )
        return specs

    def compute(self, batch: dict[str, Any]) -> None:
        """Compress sequences in-place via RLE or budget allocation.

        Reads
        -----
        {domain}_ts : list[np.ndarray]
        {domain}_len : np.ndarray, shape ``[B]``
        {domain}_f* : list[np.ndarray]

        Writes
        ------
        {domain}_ts : list[np.ndarray] (compressed, overwritten)
        {domain}_len : np.ndarray (updated)
        {domain}_f* : list[np.ndarray] (compressed, overwritten)
        {domain}_run_bucket : list[np.ndarray] (new)
        """
        for domain in self._domains:
            if domain in self._budget_domains:
                self._compress_budget(batch, domain)
            elif domain in self._action_slots:
                self._compress_rle(batch, domain)

    def _compress_rle(self, batch: dict[str, Any], domain: str) -> None:
        """RLE compress a domain's sequences on (action, item) identity."""
        ts_key = f"{domain}_ts"
        len_key = f"{domain}_len"
        if ts_key not in batch:
            return

        ts_arrays = batch[ts_key]
        lengths = batch[len_key]
        B = len(ts_arrays)

        action_slot = self._action_slots[domain]
        item_slot = self._item_slots[domain]
        sideinfo_keys = self._domain_sideinfo[domain]
        action_key = sideinfo_keys[action_slot]
        item_key = sideinfo_keys[item_slot]

        protect_k = self._protect_recent.get(domain, 0)

        # Collect all per-event array keys for this domain (exclude ts and len)
        prefix = f"{domain}_"
        seq_keys = [
            k
            for k in batch
            if k.startswith(prefix) and isinstance(batch[k], list) and k not in (len_key, ts_key)
        ]

        new_ts = []
        new_lengths = np.empty(B, dtype=lengths.dtype)
        new_arrays: dict[str, list] = {k: [] for k in seq_keys}
        new_buckets = []

        for i in range(B):
            n = int(lengths[i])
            if n <= 1:
                new_ts.append(ts_arrays[i])
                new_lengths[i] = n
                for k in seq_keys:
                    new_arrays[k].append(batch[k][i])
                bucket = np.ones(n, dtype=np.int32) if n == 1 else np.zeros(0, dtype=np.int32)
                new_buckets.append(bucket)
                continue

            ts = ts_arrays[i]
            acts = batch[action_key][i]
            items = batch[item_key][i]

            # Sort ascending by timestamp for RLE
            order = np.argsort(ts, kind="stable")
            sorted_acts = acts[order]
            sorted_items = items[order]

            # Vectorized run detection: position starts a new run when
            # action or item differs from previous
            new_run = np.empty(n, dtype=np.bool_)
            new_run[0] = True
            new_run[1:] = (sorted_acts[1:] != sorted_acts[:-1]) | (
                sorted_items[1:] != sorted_items[:-1]
            )

            # Protect recent K: last K positions (highest ts) always kept
            if protect_k > 0 and n > protect_k:
                new_run[n - protect_k :] = True

            # Run boundaries
            run_starts = np.flatnonzero(new_run)
            n_compressed = len(run_starts)

            # End of each run = start of next run - 1; last run ends at n-1
            run_ends = np.empty(n_compressed, dtype=np.intp)
            run_ends[:-1] = run_starts[1:] - 1
            run_ends[-1] = n - 1

            # Keep the last event of each run (most recent timestamp within run)
            # Compose order[run_ends] to go from sorted-index -> original-index
            keep_orig = order[run_ends]

            # Run counts: number of events per run
            run_counts = run_ends - run_starts + 1

            # Bucket counts: 1=single event, 2..N+1=count bins
            buckets = (
                np.searchsorted(self._boundaries, run_counts, side="right").astype(np.int32) + 1
            )
            buckets[run_counts == 1] = 1

            new_ts.append(ts[keep_orig])
            new_lengths[i] = n_compressed

            # Single fancy-index per key (fused sort+select)
            for k in seq_keys:
                new_arrays[k].append(batch[k][i][keep_orig])

            new_buckets.append(buckets)

        # Write back
        batch[ts_key] = new_ts
        batch[len_key] = new_lengths
        for k in seq_keys:
            batch[k] = new_arrays[k]
        batch[f"{domain}_run_bucket"] = new_buckets

    @staticmethod
    def _allocate_budget(
        group_counts: np.ndarray,
        group_max_ts: np.ndarray,
        budget: int,
    ) -> np.ndarray:
        """Distribute `budget` slots across groups respecting capacity.

        Phase 1: one slot per group (diversity floor), capped by budget.
        Phase 2: distribute remaining slots to groups with residual
        capacity, preferring groups with more events then more recent.
        """
        n_groups = len(group_counts)
        alloc = np.zeros(n_groups, dtype=np.int32)
        if budget <= 0 or n_groups == 0:
            return alloc

        if budget >= n_groups:
            # Phase 1: every group gets at least 1
            alloc[:] = 1
        else:
            # Not enough slots for all groups — pick most-recent ones
            top = np.argsort(-group_max_ts)[:budget]
            alloc[top] = 1
            return alloc

        remaining = budget - int(alloc.sum())
        capacity = group_counts.astype(np.int32) - alloc

        while remaining > 0:
            eligible = np.flatnonzero(capacity > 0)
            if len(eligible) == 0:
                break
            # Distribute one slot each to top-priority eligible groups
            order = eligible[np.lexsort((-group_max_ts[eligible], -capacity[eligible]))]
            k = min(remaining, len(order))
            chosen = order[:k]
            alloc[chosen] += 1
            capacity[chosen] -= 1
            remaining -= k

        return alloc

    def _compress_budget(self, batch: dict[str, Any], domain: str) -> None:
        """Adaptive category-budget compression.

        Allocates cap - protect_recent slots across groups, respecting
        each group's actual event count as a ceiling. Selects events
        within each group to maximize diversity of the `prefer_diverse`
        feature. Sequences at or below cap pass through.
        """
        ts_key = f"{domain}_ts"
        len_key = f"{domain}_len"
        if ts_key not in batch:
            return

        ts_arrays = batch[ts_key]
        lengths = batch[len_key]
        B = len(ts_arrays)

        cfg = self._budget_domains[domain]
        group_key_indices = cfg["group_key"]
        diverse_index = cfg["prefer_diverse"]
        cap = cfg["cap"]

        sideinfo_keys = self._domain_sideinfo[domain]
        protect_k = self._protect_recent.get(domain, 0)

        prefix = f"{domain}_"
        seq_keys = [
            k
            for k in batch
            if k.startswith(prefix) and isinstance(batch[k], list) and k not in (len_key, ts_key)
        ]

        new_ts = []
        new_lengths = np.empty(B, dtype=lengths.dtype)
        new_arrays: dict[str, list] = {k: [] for k in seq_keys}
        new_buckets = []

        for i in range(B):
            n = int(lengths[i])

            # Pass through short sequences and empty sequences
            if n <= cap:
                new_ts.append(ts_arrays[i])
                new_lengths[i] = n
                for k in seq_keys:
                    new_arrays[k].append(batch[k][i])
                new_buckets.append(np.ones(max(n, 0), dtype=np.int32))
                continue

            ts = ts_arrays[i]

            # Sort ascending by timestamp
            order = np.argsort(ts, kind="stable")

            # Split into older portion and protected tail
            split = n - protect_k
            older_idx = order[:split]
            protected_idx = order[split:]
            budget = cap - protect_k

            # No budget for older portion: keep protected tail only.
            if budget <= 0:
                keep_orig = protected_idx[-cap:] if cap > 0 else np.zeros(0, dtype=np.intp)
                final_ts = ts[keep_orig]
                final_order = np.argsort(final_ts, kind="stable")
                keep_orig = keep_orig[final_order]
                all_buckets = np.ones(len(keep_orig), dtype=np.int32)

                new_ts.append(ts[keep_orig])
                new_lengths[i] = len(keep_orig)
                for k in seq_keys:
                    new_arrays[k].append(batch[k][i][keep_orig])
                new_buckets.append(all_buckets)
                continue

            # Build group keys from sideinfo columns for older portion
            group_cols = np.stack(
                [batch[sideinfo_keys[si]][i][older_idx] for si in group_key_indices],
                axis=1,
            )
            # Encode each unique row as a group ID
            _, group_ids = np.unique(group_cols, axis=0, return_inverse=True)

            # Allocate slots across groups
            n_groups = group_ids.max() + 1 if len(group_ids) > 0 else 0
            group_counts = np.bincount(group_ids, minlength=n_groups)

            ts_older = ts[older_idx]
            group_max_ts = np.full(n_groups, np.iinfo(ts.dtype).min, dtype=ts.dtype)
            np.maximum.at(group_max_ts, group_ids, ts_older)

            allocation = self._allocate_budget(group_counts, group_max_ts, budget)

            # Score events within groups: diversity tier + recency
            diverse_vals = batch[sideinfo_keys[diverse_index]][i][older_idx]

            # Find (group, diverse_val) pairs — most recent gets diversity bonus
            pair_keys = np.stack([group_ids, diverse_vals], axis=1)
            _, pair_ids = np.unique(pair_keys, axis=0, return_inverse=True)
            n_pairs = pair_ids.max() + 1 if len(pair_ids) > 0 else 0

            pair_max_ts = np.full(n_pairs, np.iinfo(ts.dtype).min, dtype=ts.dtype)
            np.maximum.at(pair_max_ts, pair_ids, ts_older)
            is_diverse_rep = ts_older == pair_max_ts[pair_ids]

            # Score: diversity tier (2.0 for unique diverse rep) + normalized recency
            ts_min = ts_older.min()
            ts_range = ts_older.max() - ts_min
            if ts_range > 0:
                ts_norm = (ts_older - ts_min).astype(np.float64) / ts_range
            else:
                ts_norm = np.zeros(len(ts_older), dtype=np.float64)
            score = is_diverse_rep.astype(np.float64) * 2.0 + ts_norm

            # Per-group topk selection via sort + cumulative rank
            sort_order = np.lexsort((-score, group_ids))
            sorted_gids = group_ids[sort_order]

            # Compute within-group rank
            boundaries = np.flatnonzero(np.diff(sorted_gids)) + 1
            starts = np.zeros(len(sort_order), dtype=np.intp)
            if len(boundaries) > 0:
                starts[boundaries] = boundaries
                np.maximum.accumulate(starts, out=starts)
            within_group_rank = np.arange(len(sort_order)) - starts

            # Keep events where rank < allocation for their group
            per_event_alloc = allocation[sorted_gids]
            keep_in_sorted = within_group_rank < per_event_alloc
            kept_positions_in_older = sort_order[keep_in_sorted]

            # Compute count buckets: group_counts for each kept event's group
            kept_group_ids = group_ids[kept_positions_in_older]
            kept_counts = group_counts[kept_group_ids]
            kept_buckets = (
                np.searchsorted(self._boundaries, kept_counts, side="right").astype(np.int32) + 1
            )
            kept_buckets[kept_counts == 1] = 1

            # Protected tail gets bucket=1 (no compression)
            protected_buckets = np.ones(protect_k, dtype=np.int32)

            # Build final indices: selected older + protected tail
            keep_orig_older = older_idx[kept_positions_in_older]
            keep_orig = np.concatenate([keep_orig_older, protected_idx])
            all_buckets = np.concatenate([kept_buckets, protected_buckets])

            # Sort the output by timestamp ascending for consistency
            final_ts = ts[keep_orig]
            final_order = np.argsort(final_ts, kind="stable")
            keep_orig = keep_orig[final_order]
            all_buckets = all_buckets[final_order]

            new_ts.append(ts[keep_orig])
            new_lengths[i] = len(keep_orig)

            for k in seq_keys:
                new_arrays[k].append(batch[k][i][keep_orig])

            new_buckets.append(all_buckets)

        batch[ts_key] = new_ts
        batch[len_key] = new_lengths
        for k in seq_keys:
            batch[k] = new_arrays[k]
        batch[f"{domain}_run_bucket"] = new_buckets


class SeqTruncateBlock(BatchTransform):
    """Sort sequences by timestamp descending, then keep the head (most recent N events).

    For each sample, ensures the ``{domain}_ts`` column is in descending order
    (newest first), reorders all sibling feature arrays by the same permutation,
    then takes the first `ceiling` elements.

    Parameters
    ----------
    schema
        FeatureSchema — used to discover sequence domains.
    max_seq_lens
        Per-domain ceiling mapping, e.g. ``{"seq_a": 256, "seq_c": 512}``.
        Must include every domain present in the schema.
    """

    type_key = "seq_truncate"

    def __init__(
        self,
        schema: "FeatureSchema",
        max_seq_lens: dict[str, int],
    ) -> None:
        self._domains = schema.seq_domains
        self._ceilings = {domain: max_seq_lens[domain] for domain in self._domains}

    def output_specs(self) -> list[FeatureSpec]:
        """Emit {domain}_raw_len (pre-truncation length) for diagnostics."""
        return [
            FeatureSpec(
                name=f"{domain}_raw_len",
                dtype=Dtype.NUMERICAL,
                entity=Entity.USER,
                dim=1,
                source=Source.METADATA,
                batch_key=f"{domain}_raw_len",
            )
            for domain in self._domains
        ]

    def compute(self, batch: dict[str, Any]) -> None:
        """Sort sequences by timestamp descending, truncate to ceiling.

        Reads
        -----
        {domain}_len : np.ndarray, shape ``[B]``
        {domain}_ts : list[np.ndarray]
        {domain}_f* : list[np.ndarray]

        Writes
        ------
        {domain}_len : np.ndarray
            Truncated lengths (overwritten).
        {domain}_raw_len : np.ndarray
            Pre-truncation lengths (new key).
        {domain}_ts, {domain}_f* : list[np.ndarray]
            Sorted + truncated arrays (overwritten).
        """
        for domain in self._domains:
            ceiling = self._ceilings[domain]
            lengths = batch[f"{domain}_len"]
            batch[f"{domain}_raw_len"] = lengths.copy()
            ts_key = f"{domain}_ts"
            ts_arrays = batch.get(ts_key)

            B = len(lengths)
            if ts_arrays is None:
                raise ValueError(
                    f"SeqTruncateBlock requires '{ts_key}' in batch for domain '{domain}'"
                )

            sort_orders: list[np.ndarray | None] = [None] * B
            for i, ts_arr in enumerate(ts_arrays):
                if len(ts_arr) > 1:
                    if np.any(np.diff(ts_arr) > 0):
                        sort_orders[i] = np.argsort(-ts_arr, kind="stable")

            new_lengths = np.minimum(lengths, ceiling)
            batch[f"{domain}_len"] = new_lengths

            prefix = f"{domain}_"
            for key, val in list(batch.items()):
                if not key.startswith(prefix):
                    continue
                if key == f"{domain}_len":
                    continue
                if not isinstance(val, list):
                    continue
                out = []
                for i, arr in enumerate(val):
                    a = arr[sort_orders[i]] if sort_orders[i] is not None else arr
                    out.append(a[: int(new_lengths[i])])
                batch[key] = out


class ImpressionTimeCatsBlock(BatchTransform):
    """Low-cardinality time-of-day and day-of-week categoricals.

    Derives 6 features from the impression's unix timestamp. All IDs
    are 1-based (0 reserved for padding/missing). No fitting needed.
    """

    type_key = "time_cats"
    _CST_OFFSET: ClassVar[int] = 8 * 3600

    _SPECS: ClassVar[list[tuple[str, int]]] = [
        ("time_hour", 25),
        ("time_weekday", 8),
        ("time_weekend", 3),
        ("time_daypart", 7),
        ("time_weekpart_daypart", 13),
        ("time_hour_weekend", 49),
    ]

    def output_specs(self) -> list[FeatureSpec]:
        """Output features (all int32 scalars, shape `[B]`).

        time_hour : 1..24
            Hour of day (CST = UTC+8).
        time_weekday : 1..7
            Day of week (Thu=1 following unix epoch convention).
        time_weekend : 1..2
            1 = weekday, 2 = weekend (Sat/Sun under Monday=0 week).
        time_daypart : 1..6
            Coarse period: late_night/morning/noon/afternoon/evening/night.
        time_weekpart_daypart : 1..12
            Cross of weekend * daypart.
        time_hour_weekend : 1..48
            Cross of weekend * hour.
        """
        return [
            FeatureSpec(
                name=name,
                dtype=Dtype.CATEGORICAL,
                entity=Entity.USER,
                dim=1,
                # TODO (nsarang): was vocab+1, but _effective_vocab adds +1 for the padding
                # row. Storing raw count here matches base feature convention. Revert if
                # _effective_vocab changes to not add +1.
                vocab_size=vocab,
                source=Source.DERIVED,
                batch_key=name,
            )
            for name, vocab in self._SPECS
        ]

    def compute(self, batch: dict[str, Any]) -> None:
        """Derive time categoricals from unix timestamps.

        Reads
        -----
        timestamp : np.ndarray, shape ``[B]``

        Writes
        ------
        time_hour, time_weekday, time_weekend, time_daypart,
        time_weekpart_daypart, time_hour_weekend : np.ndarray, shape ``[B]``
            All int32, 1-based categorical IDs.
        """
        ts = batch["timestamp"].astype(np.int64, copy=False)
        ts_cst = ts + self._CST_OFFSET
        hour0 = ((ts_cst % 86400) // 3600).astype(np.int32)
        hour = hour0 + 1

        weekday0 = ((ts_cst // 86400) % 7).astype(np.int32)
        weekday = weekday0 + 1

        # Unix epoch 1970-01-01 was Thursday; under Monday=0, weekend is 5/6.
        dow_monday0 = ((ts_cst // 86400 + 3) % 7).astype(np.int32)
        is_weekend0 = (dow_monday0 >= 5).astype(np.int32)
        weekend = is_weekend0 + 1

        # Daypart: 0-5 late_night, 6-10 morning, 11-13 noon,
        # 14-17 afternoon, 18-21 evening, 22-23 night.
        daypart0 = np.zeros_like(hour0, dtype=np.int32)
        daypart0[(hour0 >= 6) & (hour0 <= 10)] = 1
        daypart0[(hour0 >= 11) & (hour0 <= 13)] = 2
        daypart0[(hour0 >= 14) & (hour0 <= 17)] = 3
        daypart0[(hour0 >= 18) & (hour0 <= 21)] = 4
        daypart0[(hour0 >= 22)] = 5
        daypart = daypart0 + 1

        weekpart_daypart = is_weekend0 * 6 + daypart
        hour_weekend = is_weekend0 * 24 + hour

        cats = [hour, weekday, weekend, daypart, weekpart_daypart, hour_weekend]
        for (name, _), arr in zip(self._SPECS, cats):
            batch[name] = arr.astype(np.int32)


class RecencySampleWeightBlock(BatchTransform):
    """Per-sample training weight from impression recency.

    Emits ``sample_weight`` (float32, shape ``[B]``): a recency multiplier
    the training loss applies per sample. Impressions closer to the
    train/val cutoff — hence closer to the future test period — get larger
    weight. Registered as ``METADATA`` so the model never ingests it; only
    the trainer reads it.

    The fitting phase scans the training timestamps (train-only by row
    group) to fix the reference time and a normalization constant so the
    mean train weight is 1. Mean-one normalization keeps the effective loss
    scale, grad clipping, and LR schedule unchanged versus uniform
    weighting, so a run is comparable to the unweighted baseline.

    Parameters
    ----------
    scheme
        ``"exp_halflife"`` halves the weight every `halflife_days` before
        the reference. ``"uniform"`` emits constant 1 (no-op A/B control).
    halflife_days
        Half-life of the exponential decay in days. Ignored for uniform.
    normalize
        ``"mean_one"`` rescales so the fitted train mean weight is 1;
        ``"none"`` leaves raw weights.
    max_weight
        Upper clamp on the final weight, guarding blow-ups for timestamps
        after the reference (e.g. val rows at inference).
    bin_seconds
        Histogram resolution used to compute the normalization constant
        during fit. One hour by default.
    """

    type_key = "sample_weight"

    _VALID_SCHEMES: ClassVar[tuple[str, ...]] = ("exp_halflife", "uniform")

    def __init__(
        self,
        scheme: str = "exp_halflife",
        halflife_days: float = 3.0,
        normalize: str = "mean_one",
        max_weight: float = 4.0,
        bin_seconds: int = 3600,
    ) -> None:
        if scheme not in self._VALID_SCHEMES:
            raise ValueError(f"scheme must be one of {self._VALID_SCHEMES}, got {scheme!r}")
        if normalize not in ("mean_one", "none"):
            raise ValueError(f"normalize must be 'mean_one' or 'none', got {normalize!r}")
        self._scheme = scheme
        self._halflife_sec = float(halflife_days) * 86400.0
        self._normalize = normalize
        self._max_weight = float(max_weight)
        self._bin_seconds = int(bin_seconds)

        # Fit state
        self._hist: dict[int, int] = {}
        self._ref_ts: float = None
        self._norm: float = None

    def output_specs(self) -> list[FeatureSpec]:
        """Register ``sample_weight`` as a static metadata scalar."""
        return [
            FeatureSpec(
                name="sample_weight",
                dtype=Dtype.NUMERICAL,
                entity=Entity.CONTEXT,
                dim=1,
                source=Source.METADATA,
                batch_key="sample_weight",
            )
        ]

    def fit_columns(self) -> list[str]:
        """Timestamp is needed only when the weight actually varies."""
        return [] if self._scheme == "uniform" else ["timestamp"]

    def partial_fit(self, batch: pa.RecordBatch) -> None:
        """Accumulate a coarse timestamp histogram over training rows."""
        ts = batch.column(batch.schema.get_field_index("timestamp")).to_numpy()
        bins, counts = np.unique(ts.astype(np.int64) // self._bin_seconds, return_counts=True)
        for b, c in zip(bins.tolist(), counts.tolist()):
            self._hist[b] = self._hist.get(b, 0) + c

    def finish_fit(self) -> None:
        """Set reference = most recent train bucket and the mean-one normalizer."""
        if not self._hist:
            raise RuntimeError("RecencySampleWeightBlock: no timestamps collected during fit")
        bins = np.array(sorted(self._hist), dtype=np.int64)
        counts = np.array([self._hist[int(b)] for b in bins], dtype=np.float64)
        centers = (bins + 0.5) * self._bin_seconds
        self._ref_ts = float(centers.max())
        w = self._raw_weight(centers)
        mean_w = float((w * counts).sum() / counts.sum())
        self._norm = (1.0 / mean_w) if (self._normalize == "mean_one" and mean_w > 0) else 1.0
        self._hist.clear()

    def _raw_weight(self, ts: np.ndarray) -> np.ndarray:
        """Exponential recency weight; 1.0 at the reference, halving per half-life."""
        age = np.maximum(self._ref_ts - ts, 0.0)
        return np.power(0.5, age / self._halflife_sec)

    def fit_state(self) -> dict[str, Any]:
        """Serialize the fitted reference time and normalizer."""
        if self._ref_ts is None:
            return None
        return {"ref_ts": self._ref_ts, "norm": self._norm}

    def load_fit_state(self, state: dict[str, Any]) -> None:
        """Restore reference time and normalizer, bypassing the I/O scan."""
        self._ref_ts = float(state["ref_ts"])
        self._norm = float(state["norm"])

    def compute(self, batch: dict[str, Any]) -> None:
        """Write ``sample_weight`` (float32 ``[B]``) from the impression timestamp.

        Reads
        -----
        timestamp : np.ndarray, shape ``[B]``

        Writes
        ------
        sample_weight : np.ndarray, shape ``[B]``
        """
        n_rows = len(batch["timestamp"])
        if self._scheme == "uniform" or self._ref_ts is None:
            batch["sample_weight"] = np.ones(n_rows, dtype=np.float32)
            return
        ts = batch["timestamp"].astype(np.float64, copy=False)
        w = self._raw_weight(ts) * self._norm
        np.clip(w, 0.0, self._max_weight, out=w)
        batch["sample_weight"] = w.astype(np.float32)


class SeqHourOfDayBlock(BatchTransform):
    """Per-event hour-of-day (1..24) from absolute event timestamps.

    Produces ``{domain}_time_hour`` as a list of int32 arrays — one per sample
    — following the same convention as ``TimeDeltaBucketBlock``. 0 = padding
    (missing timestamp). Must run before ``SeqTruncateBlock`` so sort+truncation
    applies automatically. Hours are in CST (UTC+8).

    Parameters
    ----------
    schema
        FeatureSchema — used to discover sequence domains.
    """

    type_key = "seq_hour"
    _CST_OFFSET: ClassVar[int] = 8 * 3600

    def __init__(self, schema: "FeatureSchema") -> None:
        self._domains = schema.seq_domains

    def output_specs(self) -> list[FeatureSpec]:
        """Output: ``{domain}_time_hour``, vocab 25 (0=pad, 1..24)."""
        return [
            FeatureSpec(
                name=f"{domain}_time_hour",
                dtype=Dtype.CATEGORICAL,
                entity=Entity.USER,
                dim=1,
                vocab_size=25,
                domain=domain,
                source=Source.DERIVED,
                batch_key=f"{domain}_time_hour",
            )
            for domain in self._domains
        ]

    def compute(self, batch: dict[str, Any]) -> None:
        """Compute per-event hour-of-day IDs (CST = UTC+8).

        Reads
        -----
        {domain}_ts : list[np.ndarray]
        {domain}_len : np.ndarray, shape ``[B]``

        Writes
        ------
        {domain}_time_hour : list[np.ndarray]
        """
        for domain in self._domains:
            ts_key = f"{domain}_ts"
            if ts_key not in batch:
                continue
            ts_arrays = batch[ts_key]
            lengths = batch[f"{domain}_len"]
            n_samples = len(ts_arrays)
            total_len = int(lengths.sum())
            if total_len == 0:
                batch[f"{domain}_time_hour"] = [np.zeros(0, dtype=np.int32)] * n_samples
                continue
            all_ts = np.concatenate(ts_arrays)
            ts_cst = all_ts + self._CST_OFFSET
            hours = ((ts_cst % 86400) // 3600 + 1).astype(np.int32)
            hours[all_ts == 0] = 0
            splits = np.cumsum(lengths[:-1])
            batch[f"{domain}_time_hour"] = np.split(hours, splits)


class SeqDayOfWeekBlock(BatchTransform):
    """Per-event day-of-week (1..7) from absolute event timestamps.

    Produces ``{domain}_time_weekday``. Day 1 = Thursday (Unix epoch convention).
    Days are in CST (UTC+8). 0 = padding. Must run before ``SeqTruncateBlock``.

    Parameters
    ----------
    schema
        FeatureSchema — used to discover sequence domains.
    """

    type_key = "seq_weekday"
    _CST_OFFSET: ClassVar[int] = 8 * 3600

    def __init__(self, schema: "FeatureSchema") -> None:
        self._domains = schema.seq_domains

    def output_specs(self) -> list[FeatureSpec]:
        """Output: ``{domain}_time_weekday``, vocab 8 (0=pad, 1..7)."""
        return [
            FeatureSpec(
                name=f"{domain}_time_weekday",
                dtype=Dtype.CATEGORICAL,
                entity=Entity.USER,
                dim=1,
                vocab_size=8,
                domain=domain,
                source=Source.DERIVED,
                batch_key=f"{domain}_time_weekday",
            )
            for domain in self._domains
        ]

    def compute(self, batch: dict[str, Any]) -> None:
        """Compute per-event day-of-week IDs (CST = UTC+8).

        Reads
        -----
        {domain}_ts : list[np.ndarray]
        {domain}_len : np.ndarray, shape ``[B]``

        Writes
        ------
        {domain}_time_weekday : list[np.ndarray]
        """
        for domain in self._domains:
            ts_key = f"{domain}_ts"
            if ts_key not in batch:
                continue
            ts_arrays = batch[ts_key]
            lengths = batch[f"{domain}_len"]
            n_samples = len(ts_arrays)
            total_len = int(lengths.sum())
            if total_len == 0:
                batch[f"{domain}_time_weekday"] = [np.zeros(0, dtype=np.int32)] * n_samples
                continue
            all_ts = np.concatenate(ts_arrays)
            ts_cst = all_ts + self._CST_OFFSET
            weekdays = ((ts_cst // 86400) % 7 + 1).astype(np.int32)
            weekdays[all_ts == 0] = 0
            splits = np.cumsum(lengths[:-1])
            batch[f"{domain}_time_weekday"] = np.split(weekdays, splits)


class CTargetHistoryBlock(BatchTransform):
    """Evidence signal from whether the target item appeared in Domain-C history.

    For each enabled granularity (exact item, same campaign, same advertiser),
    scans the user's Domain-C sequence and emits 8 dense statistics about
    matching events. The campaign/advertiser granularities require a fitting
    phase that builds `item_id → campaign/advertiser` lookup maps from
    training data.

    Parameters
    ----------
    add_exact
        Emit stats for exact item_id matches.
    add_same_campaign
        Emit stats for same-campaign matches (requires fit).
    add_same_advertiser
        Emit stats for same-advertiser matches (requires fit).
    recent_thresholds_sec
        Three time thresholds (seconds) for binary recency flags.
    max_seq_len_norm
        Denominator for log-count normalization.
    max_gap_sec
        Gap values are clamped to this before log transform.
    gap_missing_value
        Value written to the gap_missing column when timestamp is unavailable.
    max_action_id
        Action IDs above this are clipped before normalization.
    max_mapping_rows
        Stop scanning after this many rows during fit.
    max_mapping_entries
        Stop adding to maps after this many unique items.
    """

    type_key = "c_target"

    # 8 stats per kind
    _STATS_PER_KIND = 8

    def __init__(
        self,
        add_exact: bool = True,
        add_same_campaign: bool = True,
        add_same_advertiser: bool = True,
        emit_target_ids: bool = False,
        max_target_id_vocab: int = 500_000,
        recent_thresholds_sec: list[int] = None,
        max_seq_len_norm: int = 4000,
        max_gap_sec: int = 31536000,
        gap_missing_value: float = 1.0,
        max_action_id: int = 32,
        max_mapping_rows: int = 3_000_000,
        max_mapping_entries: int = 2_000_000,
    ) -> None:
        self._add_exact = add_exact
        self._add_same_campaign = add_same_campaign
        self._add_same_advertiser = add_same_advertiser
        self._emit_target_ids = emit_target_ids
        self._max_target_id_vocab = max_target_id_vocab
        self._thresholds = sorted(recent_thresholds_sec or [3600, 86400, 604800])[:3]
        while len(self._thresholds) < 3:
            self._thresholds.append(10**18)
        self._max_seq_len_norm = max(1, max_seq_len_norm)
        self._len_log_den = math.log1p(self._max_seq_len_norm)
        self._max_gap_sec = max(1, max_gap_sec)
        self._gap_log_den = math.log1p(self._max_gap_sec)
        self._gap_missing_value = gap_missing_value
        self._max_action_id = max(1, max_action_id)
        self._max_mapping_rows = max_mapping_rows
        self._max_mapping_entries = max_mapping_entries

        self._kinds: list[str] = []
        if add_exact:
            self._kinds.append("exact")
        if add_same_campaign:
            self._kinds.append("campaign")
        if add_same_advertiser:
            self._kinds.append("advertiser")

        self._item_to_campaign: dict[int, int] = {}
        self._item_to_advertiser: dict[int, int] = {}
        self._rows_seen = 0

    @property
    def dim(self) -> int:
        """Total number of dense output columns."""
        return len(self._kinds) * self._STATS_PER_KIND

    def output_specs(self) -> list[FeatureSpec]:
        """Output features.

        c_target_hist : float32, shape `[B, 8 * N_kinds]`
            Per-kind columns (repeated for exact/campaign/advertiser):

            - `flag` : 1.0 if any match found, else 0.0.
            - `log_count_norm` : log1p(match_count) / log1p(max_seq_len_norm).
            - `last_gap_log_norm` : log1p(seconds_since_last_match) / log1p(max_gap).
            - `gap_missing` : 1.0 if timestamp unavailable, else 0.0.
            - `recent_le_t1` : 1.0 if gap <= threshold_1.
            - `recent_le_t2` : 1.0 if gap <= threshold_2.
            - `recent_le_t3` : 1.0 if gap <= threshold_3.
            - `last_action_norm` : last matching event's action_id / max_action_id.
        """
        if self.dim == 0 and not self._emit_target_ids:
            return []
        specs = []
        if self.dim > 0:
            specs.append(
                FeatureSpec(
                    name="c_target_hist",
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.USER,
                    dim=self.dim,
                    source=Source.DERIVED,
                    batch_key="c_target_hist",
                )
            )
        if self._emit_target_ids:
            specs.append(
                FeatureSpec(
                    name="c_target_item_id",
                    dtype=Dtype.CATEGORICAL,
                    entity=Entity.USER,
                    dim=1,
                    vocab_size=self._max_target_id_vocab,
                    source=Source.DERIVED,
                    batch_key="c_target_item_id",
                )
            )
            if self._add_same_campaign:
                specs.append(
                    FeatureSpec(
                        name="c_target_campaign_id",
                        dtype=Dtype.CATEGORICAL,
                        entity=Entity.USER,
                        dim=1,
                        vocab_size=self._max_target_id_vocab,
                        source=Source.DERIVED,
                        batch_key="c_target_campaign_id",
                    )
                )
        return specs

    def fit_columns(self) -> list[str]:
        """Parquet columns needed to build item-to-campaign/advertiser maps."""
        if not self._add_same_campaign and not self._add_same_advertiser:
            return []
        cols = ["domain_c_seq_47", "domain_c_seq_29"]
        if self._add_same_advertiser:
            cols.append("domain_c_seq_37")
        return cols

    @property
    def fit_saturated(self) -> bool:
        """Whether enough rows have been scanned to populate the mapping."""
        return (
            self._rows_seen >= self._max_mapping_rows
            or len(self._item_to_campaign) >= self._max_mapping_entries
        )

    def partial_fit(self, batch: pa.RecordBatch) -> None:
        """Build item→campaign/advertiser maps from domain-C sequences."""
        if self._rows_seen >= self._max_mapping_rows:
            return
        if len(self._item_to_campaign) >= self._max_mapping_entries:
            return

        item_idx = batch.schema.get_field_index("domain_c_seq_47")
        camp_idx = batch.schema.get_field_index("domain_c_seq_29")
        if item_idx < 0 or camp_idx < 0:
            return

        item_col = batch.column(item_idx)
        camp_col = batch.column(camp_idx)
        item_vals = item_col.values.to_numpy(zero_copy_only=False)
        item_offs = item_col.offsets.to_numpy(zero_copy_only=False)
        camp_vals = camp_col.values.to_numpy(zero_copy_only=False)
        camp_offs = camp_col.offsets.to_numpy(zero_copy_only=False)

        adv_idx = batch.schema.get_field_index("domain_c_seq_37")
        if adv_idx >= 0 and self._add_same_advertiser:
            adv_col = batch.column(adv_idx)
            adv_vals = adv_col.values.to_numpy(zero_copy_only=False)
            adv_offs = adv_col.offsets.to_numpy(zero_copy_only=False)
        else:
            adv_vals = adv_offs = None

        B = batch.num_rows
        self._rows_seen += B
        for r in range(B):
            si, ei = int(item_offs[r]), int(item_offs[r + 1])
            sc, ec = int(camp_offs[r]), int(camp_offs[r + 1])
            n = min(ei - si, ec - sc)
            if n <= 0:
                continue
            for j in range(n):
                item = int(item_vals[si + j])
                if item <= 0:
                    continue
                camp = int(camp_vals[sc + j])
                if camp > 0 and item not in self._item_to_campaign:
                    self._item_to_campaign[item] = camp
                if adv_vals is not None:
                    sa, ea = int(adv_offs[r]), int(adv_offs[r + 1])
                    if j < ea - sa:
                        adv = int(adv_vals[sa + j])
                        if adv > 0 and item not in self._item_to_advertiser:
                            self._item_to_advertiser[item] = adv
                if len(self._item_to_campaign) >= self._max_mapping_entries:
                    return

    def fit_state(self) -> dict[str, Any] | None:
        """Serialize item→campaign/advertiser maps for checkpointing."""
        if not self._item_to_campaign:
            return None
        state: dict[str, Any] = {"item_to_campaign": self._item_to_campaign}
        if self._item_to_advertiser:
            state["item_to_advertiser"] = self._item_to_advertiser
        return state

    def load_fit_state(self, state: dict[str, Any]) -> None:
        """Restore item maps from checkpoint."""
        self._item_to_campaign = {int(k): v for k, v in state["item_to_campaign"].items()}
        if "item_to_advertiser" in state:
            self._item_to_advertiser = {int(k): v for k, v in state["item_to_advertiser"].items()}
        self._rows_seen = self._max_mapping_rows

    def compute(self, batch: dict[str, Any]) -> None:
        """Compute target-history evidence features from domain-C sequences.

        Reads
        -----
        item_id : np.ndarray, shape ``[B]``
        timestamp : np.ndarray, shape ``[B]``
        seq_c_f47, seq_c_f29, seq_c_f37 : list[np.ndarray]
        seq_c_ts, seq_c_f32 : list[np.ndarray]

        Writes
        ------
        c_target_hist : np.ndarray, shape ``[B, 8 * N_kinds]``
        c_target_item_id : np.ndarray, shape ``[B]``, optional
        c_target_campaign_id : np.ndarray, shape ``[B]``, optional
        """
        item_id = batch["item_id"]
        timestamp = batch["timestamp"]
        B = len(item_id)
        out = np.zeros((B, self.dim), dtype=np.float32)
        if self.dim == 0:
            batch["c_target_hist"] = out
            return

        # Domain-C seq keys (v2 naming: seq_c_f{fid})
        creative_key = "seq_c_f47"
        camp_key = "seq_c_f29"
        adv_key = "seq_c_f37"
        ts_key = "seq_c_ts"
        action_key = "seq_c_f32"

        has_creative = creative_key in batch
        has_camp = camp_key in batch
        has_adv = adv_key in batch
        has_ts = ts_key in batch
        has_action = action_key in batch

        if not has_creative:
            batch["c_target_hist"] = out
            return

        creatives_list = batch[creative_key]
        camps_list = batch[camp_key] if has_camp else None
        advs_list = batch[adv_key] if has_adv else None
        ts_list = batch[ts_key] if has_ts else None
        acts_list = batch[action_key] if has_action else None

        thresholds = self._thresholds

        for r in range(B):
            target_item = int(item_id[r])
            if target_item <= 0:
                continue
            target_campaign = self._item_to_campaign.get(target_item, 0)
            target_advertiser = self._item_to_advertiser.get(target_item, 0)

            creatives = creatives_list[r]
            n = len(creatives)
            if n == 0:
                continue

            camps = camps_list[r] if camps_list is not None else None
            advs = advs_list[r] if advs_list is not None else None
            tss = ts_list[r] if ts_list is not None else None
            acts = acts_list[r] if acts_list is not None else None

            pos = 0
            for kind in self._kinds:
                if kind == "exact":
                    mask = creatives == target_item
                elif kind == "campaign":
                    if camps is None or target_campaign <= 0:
                        mask = None
                    else:
                        m = min(len(camps), n)
                        mask = camps[:m] == target_campaign
                elif advs is None or target_advertiser <= 0:
                    mask = None
                else:
                    m = min(len(advs), n)
                    mask = advs[:m] == target_advertiser

                base = pos
                pos += self._STATS_PER_KIND
                if mask is None or len(mask) == 0:
                    out[r, base + 3] = self._gap_missing_value
                    continue
                idx = np.flatnonzero(mask)
                cnt = int(idx.size)
                if cnt <= 0:
                    out[r, base + 3] = self._gap_missing_value
                    continue

                out[r, base + 0] = 1.0
                out[r, base + 1] = math.log1p(min(cnt, self._max_seq_len_norm)) / self._len_log_den

                gap_missing = True
                if tss is not None and len(tss) > 0:
                    valid_idx = idx[idx < len(tss)]
                    if valid_idx.size:
                        ts_match = tss[valid_idx].astype(np.int64)
                        ts_match = ts_match[ts_match > 0]
                        if ts_match.size:
                            last_ts = int(ts_match.max())
                            gap = max(int(timestamp[r]) - last_ts, 0)
                            gap = min(gap, self._max_gap_sec)
                            out[r, base + 2] = math.log1p(gap) / self._gap_log_den
                            out[r, base + 4] = 1.0 if gap <= thresholds[0] else 0.0
                            out[r, base + 5] = 1.0 if gap <= thresholds[1] else 0.0
                            out[r, base + 6] = 1.0 if gap <= thresholds[2] else 0.0
                            gap_missing = False
                if gap_missing:
                    out[r, base + 3] = self._gap_missing_value

                if acts is not None and len(acts) > 0:
                    valid_idx = idx[idx < len(acts)]
                    if valid_idx.size:
                        last_action = int(acts[valid_idx][-1])
                        if last_action > 0:
                            out[r, base + 7] = (
                                min(last_action, self._max_action_id) / self._max_action_id
                            )

        batch["c_target_hist"] = out

        if self._emit_target_ids:
            item_ids_out = np.zeros(B, dtype=np.int64)
            campaign_ids_out = np.zeros(B, dtype=np.int64)
            for r in range(B):
                target = int(item_id[r])
                if target <= 0:
                    continue
                if has_creative:
                    seq = creatives_list[r]
                    if len(seq) > 0 and np.any(seq == target):
                        item_ids_out[r] = target
                campaign = self._item_to_campaign.get(target, 0)
                if campaign > 0:
                    campaign_ids_out[r] = campaign
            batch["c_target_item_id"] = item_ids_out
            if self._add_same_campaign:
                batch["c_target_campaign_id"] = campaign_ids_out


class ConversionStateBlock(BatchTransform):
    """Per-domain recent conversion counts/ratios and global activity-state features.

    Iterates over sequence domains, reads each domain's timestamp array, and
    computes recency-based statistics relative to the impression timestamp.

    Output groups:
    - Per-domain recent counts (log1p-normalized)
    - Per-domain recent ratios (recent_count / total_seq_len)
    - Global state: active_domains per threshold, any_domain_recent flags,
      length entropy, burst ratios, total recent/len counts

    All features are emitted under ``group="conversion_state"`` for exclusive
    group_head routing.

    Parameters
    ----------
    schema
        FeatureSchema — used to discover sequence domains.
    recent_thresholds_sec
        Time thresholds in seconds for recency counting.
    max_seq_len_norm
        Denominator for log-count normalization.
    """

    type_key = "conversion_state"

    def __init__(
        self,
        schema: "FeatureSchema",
        recent_thresholds_sec: list[int] = None,
        max_seq_len_norm: float = 4000,
    ) -> None:
        self._domains = schema.seq_domains
        self._thresholds = sorted(recent_thresholds_sec or [3600, 86400, 604800])
        self._max_seq_len_norm = max(1.0, max_seq_len_norm)
        self._len_log_den = math.log1p(self._max_seq_len_norm)
        self._names = self._build_names()

    def _suffix(self, thr: int) -> str:
        """Human-readable suffix for a threshold in seconds."""
        if thr % 86400 == 0:
            return f"{thr // 86400}d"
        if thr % 3600 == 0:
            return f"{thr // 3600}h"
        if thr % 60 == 0:
            return f"{thr // 60}m"
        return f"{thr}s"

    def _build_names(self) -> list[str]:
        """Build ordered list of output feature names."""
        names: list[str] = []
        # Per-domain recent counts
        for domain in self._domains:
            for thr in self._thresholds:
                names.append(f"conv_state_{domain}_recent_count_le_{self._suffix(thr)}")
        # Per-domain recent ratios
        for domain in self._domains:
            for thr in self._thresholds:
                names.append(f"conv_state_{domain}_recent_ratio_le_{self._suffix(thr)}")
        # Global state
        for thr in self._thresholds:
            names.append(f"conv_state_active_domains_le_{self._suffix(thr)}")
            names.append(f"conv_state_any_domain_recent_le_{self._suffix(thr)}")
        names.extend(
            [
                "conv_state_len_entropy_norm",
                "conv_state_burst_ratio_t0_over_t1",
                "conv_state_burst_ratio_t1_over_t2",
                "conv_state_total_recent_t0_log_norm",
                "conv_state_total_recent_t1_log_norm",
                "conv_state_total_len_log_norm",
            ]
        )
        return names

    @property
    def dim(self) -> int:
        """Total number of output features."""
        return len(self._names)

    def output_specs(self) -> list[FeatureSpec]:
        """Output features.

        conv_state : float32, shape `[B, dim]`
            Concatenation of per-domain counts, ratios, and global state
            features. All registered under group ``"conversion_state"``.
        """
        return [
            FeatureSpec(
                name=name,
                dtype=Dtype.NUMERICAL,
                entity=Entity.USER,
                dim=1,
                source=Source.DERIVED,
                group="conversion_state",
                batch_key="conv_state",
                col_range=(i, i + 1),
            )
            for i, name in enumerate(self._names)
        ]

    def compute(self, batch: dict[str, Any]) -> None:
        """Compute conversion-state features for a batch.

        Reads
        -----
        timestamp : np.ndarray, shape ``[B]``
        {domain}_len : np.ndarray, shape ``[B]``
        {domain}_ts : list[np.ndarray]

        Writes
        ------
        conv_state : np.ndarray, shape ``[B, dim]``
            Per-domain counts, ratios, and global activity-state features.
        """
        imp_ts = batch["timestamp"].astype(np.int64, copy=False)
        B = len(imp_ts)
        out = np.zeros((B, self.dim), dtype=np.float32)
        if self.dim == 0:
            batch["conv_state"] = out
            return

        D = len(self._domains)
        T = len(self._thresholds)
        raw_lens = np.zeros((B, D), dtype=np.float32)
        recent_counts = np.zeros((B, D, T), dtype=np.float32)

        for d_idx, domain in enumerate(self._domains):
            ts_key = f"{domain}_ts"
            if ts_key not in batch:
                continue
            lengths = batch[f"{domain}_len"].astype(np.int64)
            ts_arrays = batch[ts_key]

            total_len = int(lengths.sum())
            if total_len == 0:
                continue

            # Vectorized: concat all ts, build row labels, filter valid (>0)
            flat_ts_all = np.concatenate(ts_arrays).astype(np.int64)
            flat_rows_all = np.repeat(np.arange(B, dtype=np.int32), lengths)

            valid_mask = flat_ts_all > 0
            flat_ts = flat_ts_all[valid_mask]
            flat_rows = flat_rows_all[valid_mask]

            if flat_ts.size == 0:
                continue

            # Per-row valid counts for raw_lens
            raw_lens[:, d_idx] = np.bincount(flat_rows, minlength=B).astype(np.float32)

            flat_gaps = np.maximum(imp_ts[flat_rows] - flat_ts, 0)

            for t_idx, thr in enumerate(self._thresholds):
                mask = flat_gaps <= thr
                counts = np.bincount(flat_rows[mask], minlength=B).astype(np.float32)
                recent_counts[:, d_idx, t_idx] = counts

        pos = 0
        # Per-domain recent counts (log1p-normalized)
        for d_idx in range(D):
            for t_idx in range(T):
                out[:, pos] = np.log1p(recent_counts[:, d_idx, t_idx]) / self._len_log_den
                pos += 1

        # Per-domain recent ratios
        denom = np.maximum(raw_lens, 1.0)
        for d_idx in range(D):
            for t_idx in range(T):
                out[:, pos] = recent_counts[:, d_idx, t_idx] / denom[:, d_idx]
                pos += 1

        # Global state
        active_by_thr = recent_counts > 0
        for t_idx in range(T):
            n_active = active_by_thr[:, :, t_idx].sum(axis=1).astype(np.float32)
            out[:, pos] = n_active / max(D, 1)
            pos += 1
            out[:, pos] = (n_active > 0).astype(np.float32)
            pos += 1

        # Length entropy normalized by log(D)
        total_len = raw_lens.sum(axis=1)
        prob = raw_lens / np.maximum(total_len[:, None], 1.0)
        entropy = -(prob * np.log(np.maximum(prob, 1e-8))).sum(axis=1)
        out[:, pos] = entropy / math.log(max(D, 2))
        pos += 1

        # Burst ratios: narrow-window / wider-window activity
        total_recent = recent_counts.sum(axis=1) if T > 0 else np.zeros((B, 0), dtype=np.float32)
        if T >= 2:
            out[:, pos] = np.log1p(total_recent[:, 0]) / np.log1p(total_recent[:, 1] + 1.0)
        pos += 1
        if T >= 3:
            out[:, pos] = np.log1p(total_recent[:, 1]) / np.log1p(total_recent[:, 2] + 1.0)
        pos += 1

        # Total recent counts (log-normalized)
        if T >= 1:
            out[:, pos] = np.log1p(total_recent[:, 0]) / self._len_log_den
        pos += 1
        if T >= 2:
            out[:, pos] = np.log1p(total_recent[:, 1]) / self._len_log_den
        pos += 1
        out[:, pos] = np.log1p(total_len) / self._len_log_den
        pos += 1

        if pos != self.dim:
            raise RuntimeError(f"ConversionStateBlock mismatch: wrote {pos}, expected {self.dim}")

        batch["conv_state"] = out


class L2NormBlock(BatchTransform):
    """Per-sample L2 normalization for embedding-like dense features.

    Stateless — no fitting required. Each matched feature is independently
    normalized to unit L2 norm along the feature dimension.

    Parameters
    ----------
    schema
        FeatureSchema — used to discover and locate target features.
    pattern
        DSL filter expression selecting features to normalize.
    """

    type_key = "l2_norm"

    def __init__(
        self,
        schema: "FeatureSchema",
        pattern: str,
    ) -> None:
        self._schema = schema
        self._expr = pattern
        self._specs = schema.query(self._expr)

    def compute(self, batch: dict[str, Any]) -> None:
        """L2-normalize each matched feature per sample."""
        EPS = 1e-6
        for spec in self._specs:
            data = self._schema.extract(batch, names=spec.name)
            if data is None:
                continue
            norms = np.linalg.norm(data, axis=-1, keepdims=True)
            self._schema.update(batch, f"name = '{spec.name}'", data / np.maximum(norms, EPS))


class FreqFilterBlock(BatchTransform):
    """Zero rare categorical values based on training-set frequency.

    Values below `min_count` get zeroed (mapped to padding). Features with
    vocab >= `max_vocab` are handled according to `oov_mode`:
    - ``"skip"`` (default): left untouched, not counted.
    - ``"zero"``: zeroed unconditionally without counting.
    """

    type_key = "freq_filter"
    _EXPR = "dtype = 'categorical' and source = 'original'"

    def __init__(
        self,
        schema: FeatureSchema,
        min_count: int = 5,
        max_vocab: int = 2_000_000,
        max_rows: int = None,
        oov_mode: str = "skip",
    ) -> None:
        self._schema = schema
        self._min_count = min_count
        self._max_rows = max_rows
        if oov_mode not in ("skip", "zero"):
            raise ValueError(f"oov_mode must be 'skip' or 'zero', got {oov_mode!r}")
        self._oov_mode = oov_mode

        all_cat = schema.query(self._EXPR)
        # Split into countable vs out-of-vocab
        self._specs: list[FeatureSpec] = []
        self._zero_all: list[FeatureSpec] = []
        for s in all_cat:
            if s.vocab_size >= max_vocab:
                if oov_mode == "zero":
                    self._zero_all.append(s)
                # "skip": don't count, don't touch
            else:
                self._specs.append(s)

        self._counters = [np.zeros(s.vocab_size, dtype=np.int32) for s in self._specs]
        self._rows_seen: int = 0
        self._rare_masks: list[np.ndarray] | None = None
        self._executor: ThreadPoolExecutor = None

    def fit_columns(self) -> list[str]:
        """Categorical columns needed for frequency counting."""
        return [s.source_col for s in self._specs]

    @property
    def fit_saturated(self) -> bool:
        """Whether enough rows have been counted."""
        if self._max_rows is None:
            return False
        return self._rows_seen >= self._max_rows

    _FIT_CHUNK_ROWS = 10_000
    _FIT_MAX_WORKERS = 8

    def partial_fit(self, batch: pa.RecordBatch) -> None:
        """Count unique (row, value) pairs per categorical feature.

        List columns use chunked sort+mask with threading for GIL-free
        parallelism across row-aligned chunks.
        """
        self._rows_seen += batch.num_rows

        # Collect list-column work items (extract arrays outside the thread pool)
        list_tasks: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for i, spec in enumerate(self._specs):
            col_idx = batch.schema.get_field_index(spec.source_col)
            if col_idx < 0:
                continue
            col = batch.column(col_idx)
            if pa.types.is_list(col.type):
                flat = pc.list_flatten(col).to_numpy(zero_copy_only=False)
                parent_ids = pc.list_parent_indices(col).to_numpy(zero_copy_only=False)
                list_tasks.append((i, spec.vocab_size, flat, parent_ids))
            else:
                if col.null_count == len(col):
                    continue
                vc = pc.value_counts(col.drop_null())
                vals = vc.field("values").to_numpy(zero_copy_only=False).astype(np.intp)
                cnts = vc.field("counts").to_numpy(zero_copy_only=False).astype(np.int32)
                valid = (vals >= 0) & (vals < spec.vocab_size)
                np.add.at(self._counters[i], vals[valid], cnts[valid])

        if not list_tasks:
            return

        # Process each list column: chunked sort+mask, threaded across chunks
        chunk_rows = self._FIT_CHUNK_ROWS

        def _process_list_col(task):
            i, vocab_size, flat, parent_ids = task
            if len(flat) == 0:
                return i, np.zeros(vocab_size, dtype=np.int32)
            total_rows = int(parent_ids[-1]) + 1
            row_starts = np.searchsorted(parent_ids, np.arange(0, total_rows, chunk_rows))
            row_starts = np.append(row_starts, len(parent_ids))
            n_chunks = len(row_starts) - 1

            def _sort_chunk(ci):
                start, end = row_starts[ci], row_starts[ci + 1]
                if start >= end:
                    return None
                cf = flat[start:end]
                cp = parent_ids[start:end] - parent_ids[start]
                keys = cp.astype(np.int64) * vocab_size + cf.astype(np.int64)
                keys.sort()
                mask = np.empty(len(keys), dtype=bool)
                mask[0] = True
                mask[1:] = keys[1:] != keys[:-1]
                deduped = (keys[mask] % vocab_size).astype(np.intp)
                valid = (deduped >= 0) & (deduped < vocab_size)
                return np.bincount(deduped[valid], minlength=vocab_size)

            if n_chunks == 1:
                result = _sort_chunk(0)
                return i, result.astype(np.int32) if result is not None else np.zeros(
                    vocab_size, dtype=np.int32
                )

            chunk_counts = list(self._executor.map(_sort_chunk, range(n_chunks)))

            counter = np.zeros(vocab_size, dtype=np.int32)
            for c in chunk_counts:
                if c is not None:
                    counter += c.astype(np.int32)
            return i, counter

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._FIT_MAX_WORKERS)

        # Process columns sequentially (threading happens inside each column)
        for task in list_tasks:
            i, counter = _process_list_col(task)
            valid_len = min(len(counter), self._specs[i].vocab_size)
            self._counters[i][:valid_len] += counter[:valid_len]

    def finish_fit(self) -> None:
        """Build boolean masks from accumulated counts."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._rare_masks = []
        for counter in self._counters:
            mask = counter < self._min_count
            mask[0] = False  # padding index is never rare
            self._rare_masks.append(mask)

    def fit_state(self) -> dict[str, Any] | None:
        """Serialize rare masks for checkpointing."""
        if self._rare_masks is None:
            return None
        return {
            "rare_masks": [m.tolist() for m in self._rare_masks],
            "zero_all": [s.name for s in self._zero_all],
            "oov_mode": self._oov_mode,
        }

    def load_fit_state(self, state: dict[str, Any]) -> None:
        """Restore rare masks from checkpoint."""
        self._rare_masks = [np.array(m, dtype=bool) for m in state["rare_masks"]]
        # Rebuild _zero_all from names in case schema order changed
        zero_names = set(state.get("zero_all", []))
        self._zero_all = [s for s in self._schema.query(self._EXPR) if s.name in zero_names]

    def _zero_array(self, arr: np.ndarray, mask: np.ndarray) -> None:
        """Zero values flagged by mask, in-place."""
        in_range = (arr > 0) & (arr < len(mask))
        rare = in_range & mask[arr * in_range]
        arr[rare] = 0

    def compute(self, batch: dict[str, Any]) -> None:
        """Zero rare values and unconditionally zero high-vocab features."""
        if self._rare_masks is None:
            return

        for spec, mask in zip(self._specs, self._rare_masks):
            view = self._schema._slice(batch, spec.name)
            if isinstance(view, list):
                for arr in view:
                    if len(arr) > 0:
                        self._zero_array(arr, mask)
            else:
                self._zero_array(view, mask)

        for spec in self._zero_all:
            view = self._schema._slice(batch, spec.name)
            if isinstance(view, list):
                for arr in view:
                    arr[:] = 0
            else:
                view[:] = 0


class TemporalDynamicsBlock(BatchTransform):
    """Temporal velocity, fatigue, and context features from sequence histories.

    Computes 16 dense features in three groups:

    - **temporal_velocity** (7): per-domain category acceleration and
      cross-domain synchrony for the target item's category_l1.
    - **temporal_fatigue** (5): advertiser-level exposure curve signals
      from domain-C ad history.
    - **temporal_context** (4): ecosystem-level ad load and category share.

    Requires a fitting phase to build item→advertiser mapping from domain-C
    sequences (same source columns as CTargetHistoryBlock).

    Parameters
    ----------
    item_cat_col_idx
        Column index of category_l1 in the packed ``item_cat`` tensor.
    category_fids
        Mapping from domain name to the feature ID of category_l1 in that domain.
    advertiser_fid
        Feature ID of advertiser_id in domain-C.
    velocity_window_sec
        Recent window for velocity numerator (default 3 days).
    velocity_base_sec
        Base window for velocity denominator (default 30 days).
    fatigue_recent_sec
        Recent window for fatigue signals (default 1 day).
    fatigue_base_sec
        Base window for fatigue denominator (default 30 days).
    ad_load_window_sec
        Window for ad ecosystem load features (default 1 day).
    max_exposures_norm
        Denominator for total exposure count log normalization.
    max_recent_exposures_norm
        Denominator for recent exposure count log normalization.
    max_ad_load_norm
        Denominator for ad load count log normalization.
    max_advertisers_norm
        Denominator for distinct advertisers log normalization.
    max_gap_sec
        Maximum gap in seconds for days-since-first normalization (default 1 year).
    max_mapping_rows
        Stop scanning after this many rows during fit.
    max_mapping_entries
        Stop adding to advertiser map after this many entries.
    """

    type_key = "temporal_dynamics"

    _VELOCITY_NAMES: ClassVar[list[str]] = [
        "td_cat_velocity_seq_a",
        "td_cat_velocity_seq_b",
        "td_cat_velocity_seq_c",
        "td_cat_velocity_seq_d",
        "td_cross_domain_active",
        "td_cross_domain_organic",
        "td_cross_domain_ratio",
    ]
    _FATIGUE_NAMES: ClassVar[list[str]] = [
        "td_adv_n_exposures_log",
        "td_adv_recent_exposures",
        "td_adv_fatigue_ratio",
        "td_adv_exposure_accel",
        "td_adv_days_since_first",
    ]
    _CONTEXT_NAMES: ClassVar[list[str]] = [
        "td_first_exposure_flag",
        "td_ad_load_24h_log",
        "td_n_advertisers_24h_log",
        "td_category_share",
    ]

    def __init__(
        self,
        schema: "FeatureSchema" = None,
        item_cat_col_idx: int = 5,
        category_fids: dict[str, int] = None,
        advertiser_fid: int = 37,
        velocity_window_sec: int = 259200,
        velocity_base_sec: int = 2592000,
        fatigue_recent_sec: int = 86400,
        fatigue_base_sec: int = 2592000,
        ad_load_window_sec: int = 86400,
        max_exposures_norm: int = 500,
        max_recent_exposures_norm: int = 50,
        max_ad_load_norm: int = 200,
        max_advertisers_norm: int = 50,
        max_gap_sec: int = 31536000,
        max_mapping_rows: int = 3_000_000,
        max_mapping_entries: int = 2_000_000,
    ) -> None:
        self._item_cat_col_idx = item_cat_col_idx
        self._category_fids = category_fids or {
            "seq_a": 42,
            "seq_b": 70,
            "seq_c": 30,
            "seq_d": 18,
        }
        self._advertiser_fid = advertiser_fid
        self._velocity_window_sec = velocity_window_sec
        self._velocity_base_sec = velocity_base_sec
        self._fatigue_recent_sec = fatigue_recent_sec
        self._fatigue_base_sec = fatigue_base_sec
        self._ad_load_window_sec = ad_load_window_sec
        self._max_exposures_norm = max_exposures_norm
        self._max_recent_exposures_norm = max_recent_exposures_norm
        self._max_ad_load_norm = max_ad_load_norm
        self._max_advertisers_norm = max_advertisers_norm
        self._max_gap_sec = max_gap_sec
        self._max_mapping_rows = max_mapping_rows
        self._max_mapping_entries = max_mapping_entries

        # Precompute log denominators
        self._log_exposures_den = math.log1p(max_exposures_norm)
        self._log_recent_exp_den = math.log1p(max_recent_exposures_norm)
        self._log_ad_load_den = math.log1p(max_ad_load_norm)
        self._log_advertisers_den = math.log1p(max_advertisers_norm)
        self._log_gap_den = math.log1p(max_gap_sec)

        # Organic domains (non-ad) for cross-domain ratio
        self._organic_domains = ["seq_a", "seq_b", "seq_d"]
        self._ad_domain = "seq_c"

        self._all_names = self._VELOCITY_NAMES + self._FATIGUE_NAMES + self._CONTEXT_NAMES

        # Fit state
        self._item_to_advertiser: dict[int, int] = {}
        self._rows_seen = 0

    @property
    def dim(self) -> int:
        """Total output features."""
        return len(self._all_names)

    def output_specs(self) -> list[FeatureSpec]:
        """Register 16 features across 3 groups under one batch key."""
        specs = []
        for i, name in enumerate(self._VELOCITY_NAMES):
            specs.append(
                FeatureSpec(
                    name=name,
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.USER,
                    dim=1,
                    source=Source.DERIVED,
                    group="temporal_velocity",
                    batch_key="temporal_dynamics",
                    col_range=(i, i + 1),
                )
            )
        offset = len(self._VELOCITY_NAMES)
        for i, name in enumerate(self._FATIGUE_NAMES):
            specs.append(
                FeatureSpec(
                    name=name,
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.USER,
                    dim=1,
                    source=Source.DERIVED,
                    group="temporal_fatigue",
                    batch_key="temporal_dynamics",
                    col_range=(offset + i, offset + i + 1),
                )
            )
        offset += len(self._FATIGUE_NAMES)
        for i, name in enumerate(self._CONTEXT_NAMES):
            specs.append(
                FeatureSpec(
                    name=name,
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.USER,
                    dim=1,
                    source=Source.DERIVED,
                    group="temporal_context",
                    batch_key="temporal_dynamics",
                    col_range=(offset + i, offset + i + 1),
                )
            )
        return specs

    # ──── Fitting: build item → advertiser map ────

    def fit_columns(self) -> list[str]:
        """Same source columns as CTargetHistory for advertiser mapping."""
        return ["domain_c_seq_47", "domain_c_seq_37"]

    @property
    def fit_saturated(self) -> bool:
        """Whether the advertiser map has reached capacity."""
        return (
            self._rows_seen >= self._max_mapping_rows
            or len(self._item_to_advertiser) >= self._max_mapping_entries
        )

    def partial_fit(self, batch: pa.RecordBatch) -> None:
        """Build item_id → advertiser_id map from domain-C sequences."""
        if self.fit_saturated:
            return

        item_idx = batch.schema.get_field_index("domain_c_seq_47")
        adv_idx = batch.schema.get_field_index("domain_c_seq_37")
        if item_idx < 0 or adv_idx < 0:
            return

        item_col = batch.column(item_idx)
        adv_col = batch.column(adv_idx)
        item_vals = item_col.values.to_numpy(zero_copy_only=False)
        item_offs = item_col.offsets.to_numpy(zero_copy_only=False)
        adv_vals = adv_col.values.to_numpy(zero_copy_only=False)
        adv_offs = adv_col.offsets.to_numpy(zero_copy_only=False)

        B = batch.num_rows
        self._rows_seen += B
        for r in range(B):
            si, ei = int(item_offs[r]), int(item_offs[r + 1])
            sa, ea = int(adv_offs[r]), int(adv_offs[r + 1])
            n = min(ei - si, ea - sa)
            if n <= 0:
                continue
            for j in range(n):
                item = int(item_vals[si + j])
                if item <= 0:
                    continue
                adv = int(adv_vals[sa + j])
                if adv > 0 and item not in self._item_to_advertiser:
                    self._item_to_advertiser[item] = adv
                if len(self._item_to_advertiser) >= self._max_mapping_entries:
                    return

    def finish_fit(self) -> None:
        """No-op; map is ready after partial_fit accumulation."""

    def fit_state(self) -> dict[str, Any] | None:
        """Serialize the item-to-advertiser map for checkpointing."""
        if not self._item_to_advertiser:
            return None
        return {"item_to_advertiser": self._item_to_advertiser}

    def load_fit_state(self, state: dict[str, Any]) -> None:
        """Restore the item-to-advertiser map from a checkpoint."""
        self._item_to_advertiser = {int(k): v for k, v in state["item_to_advertiser"].items()}
        self._rows_seen = self._max_mapping_rows

    # ──── Compute ────

    def compute(self, batch: dict[str, Any]) -> None:
        """Compute temporal dynamics features for a batch.

        Reads
        -----
        timestamp : np.ndarray, shape ``[B]``
        item_cat : np.ndarray, shape ``[B, total_dim]``
        item_id : np.ndarray, shape ``[B]``
        {domain}_f{cat_fid} : list[np.ndarray]
        {domain}_ts : list[np.ndarray]
        {domain}_len : np.ndarray, shape ``[B]``
        seq_c_f37 : list[np.ndarray]

        Writes
        ------
        temporal_dynamics : np.ndarray, shape ``[B, 16]``
        """
        imp_ts = batch["timestamp"].astype(np.int64, copy=False)
        B = len(imp_ts)
        out = np.zeros((B, self.dim), dtype=np.float32)

        target_cat = batch["item_cat"][:, self._item_cat_col_idx].astype(np.int64)
        item_id = batch["item_id"].astype(np.int64)

        vel_window = self._velocity_window_sec
        vel_base = self._velocity_base_sec
        domain_order = ["seq_a", "seq_b", "seq_c", "seq_d"]

        # ── Velocity group (cols 0-6): vectorized per-domain ──
        per_domain_recent = np.zeros((B, 4), dtype=np.float32)

        for d_idx, domain in enumerate(domain_order):
            cat_fid = self._category_fids.get(domain)
            if cat_fid is None:
                continue
            cat_key = f"{domain}_f{cat_fid}"
            ts_key = f"{domain}_ts"
            if cat_key not in batch or ts_key not in batch:
                continue

            lengths = batch[f"{domain}_len"].astype(np.int64)
            total_len = int(lengths.sum())
            if total_len == 0:
                continue

            flat_cats = np.concatenate(batch[cat_key]).astype(np.int64)
            flat_ts = np.concatenate(batch[ts_key]).astype(np.int64)
            flat_rows = np.repeat(np.arange(B, dtype=np.int32), lengths)

            # Match: cat == target_cat[row] AND ts > 0
            per_row_target = target_cat[flat_rows]
            match = (flat_cats == per_row_target) & (per_row_target > 0) & (flat_ts > 0)
            if not match.any():
                continue

            m_rows = flat_rows[match]
            m_gaps = np.maximum(imp_ts[m_rows] - flat_ts[match], 0)

            recent_mask = m_gaps <= vel_window
            base_mask = m_gaps <= vel_base

            recent_counts = np.bincount(m_rows[recent_mask], minlength=B).astype(np.float32)
            base_counts = np.bincount(m_rows[base_mask], minlength=B).astype(np.float32)
            per_domain_recent[:, d_idx] = recent_counts

            # vel_window < vel_base guarantees recent ⊆ base, so base > 0
            # whenever recent > 0. Guard against misconfigured windows anyway.
            has_base = base_counts > 0
            out[has_base, d_idx] = np.log1p(recent_counts[has_base]) / np.log1p(
                base_counts[has_base]
            )

        # Cross-domain features
        active_mask = per_domain_recent > 0
        out[:, 4] = active_mask.sum(axis=1).astype(np.float32) / 4.0

        organic_idx = [domain_order.index(d) for d in self._organic_domains]
        organic_active = active_mask[:, organic_idx].sum(axis=1).astype(np.float32)
        out[:, 5] = np.log1p(organic_active) / math.log1p(3.0)

        ad_idx = domain_order.index(self._ad_domain)
        ad_recent = per_domain_recent[:, ad_idx]
        organic_total = per_domain_recent[:, organic_idx].sum(axis=1)
        out[:, 6] = np.log1p(organic_total) / np.log1p(ad_recent + 1.0)

        # ── Fatigue group (cols 7-11): vectorized ──
        fatigue_offset = len(self._VELOCITY_NAMES)
        context_offset = fatigue_offset + len(self._FATIGUE_NAMES)
        adv_key = f"seq_c_f{self._advertiser_fid}"
        ts_key_c = "seq_c_ts"

        # Build per-row target advertiser array
        target_adv = np.array(
            [self._item_to_advertiser.get(int(iid), 0) for iid in item_id], dtype=np.int64
        )

        has_exposures = np.zeros(B, dtype=bool)

        if adv_key in batch and ts_key_c in batch:
            lengths_c = batch["seq_c_len"].astype(np.int64)
            total_len_c = int(lengths_c.sum())

            if total_len_c > 0:
                flat_advs = np.concatenate(batch[adv_key]).astype(np.int64)
                flat_ts_c = np.concatenate(batch[ts_key_c]).astype(np.int64)
                flat_rows_c = np.repeat(np.arange(B, dtype=np.int32), lengths_c)

                # Match: adv == target_adv[row] AND target_adv > 0
                per_row_target_adv = target_adv[flat_rows_c]
                adv_match = (flat_advs == per_row_target_adv) & (per_row_target_adv > 0)

                # Count total advertiser exposures per row
                exposure_counts = np.bincount(flat_rows_c[adv_match], minlength=B).astype(
                    np.float32
                )

                has_exposures = exposure_counts > 0
                out[has_exposures, fatigue_offset + 0] = (
                    np.log1p(exposure_counts[has_exposures]) / self._log_exposures_den
                )

                # Time-based fatigue features (only for rows with valid ts matches)
                adv_ts_match = adv_match & (flat_ts_c > 0)
                if adv_ts_match.any():
                    m_rows_f = flat_rows_c[adv_ts_match]
                    m_gaps_f = np.maximum(imp_ts[m_rows_f] - flat_ts_c[adv_ts_match], 0)

                    # Recent exposures (24h)
                    recent_f = m_gaps_f <= self._fatigue_recent_sec
                    recent_exp_counts = np.bincount(m_rows_f[recent_f], minlength=B).astype(
                        np.float32
                    )
                    has_recent = recent_exp_counts > 0
                    out[has_recent, fatigue_offset + 1] = (
                        np.log1p(recent_exp_counts[has_recent]) / self._log_recent_exp_den
                    )

                    # Fatigue ratio: recent / base
                    base_f = m_gaps_f <= self._fatigue_base_sec
                    base_exp_counts = np.bincount(m_rows_f[base_f], minlength=B).astype(np.float32)
                    has_base_f = base_exp_counts > 0
                    ratio = np.zeros(B, dtype=np.float32)
                    ratio[has_base_f] = np.minimum(
                        recent_exp_counts[has_base_f] / base_exp_counts[has_base_f], 1.0
                    )
                    out[:, fatigue_offset + 2] = ratio

                    # Exposure acceleration: 3d density vs 3-30d density
                    three_day_mask = m_gaps_f <= self._velocity_window_sec
                    mid_mask = (m_gaps_f > self._velocity_window_sec) & (
                        m_gaps_f <= self._fatigue_base_sec
                    )
                    three_day_counts = np.bincount(m_rows_f[three_day_mask], minlength=B).astype(
                        np.float32
                    )
                    mid_counts = np.bincount(m_rows_f[mid_mask], minlength=B).astype(np.float32)
                    has_mid = mid_counts > 0
                    has_3d_only = (three_day_counts > 0) & ~has_mid
                    accel = np.zeros(B, dtype=np.float32)
                    # Normalize by window size ratio
                    recent_days = self._velocity_window_sec / 86400.0
                    mid_days = (self._fatigue_base_sec - self._velocity_window_sec) / 86400.0
                    accel[has_mid] = (three_day_counts[has_mid] / recent_days) / (
                        mid_counts[has_mid] / mid_days
                    )
                    out[has_mid, fatigue_offset + 3] = np.minimum(
                        np.log1p(accel[has_mid]) / math.log1p(10), 1.0
                    )
                    out[has_3d_only, fatigue_offset + 3] = 1.0

                    # Days since first: max gap per row
                    order = np.argsort(m_rows_f, kind="stable")
                    sorted_rows = m_rows_f[order]
                    sorted_gaps = m_gaps_f[order]
                    breaks = np.concatenate([[0], np.flatnonzero(np.diff(sorted_rows)) + 1])
                    max_gaps = np.maximum.reduceat(sorted_gaps, breaks)
                    unique_rows = sorted_rows[breaks]
                    out[unique_rows, fatigue_offset + 4] = (
                        np.log1p(max_gaps.astype(np.float32)) / self._log_gap_den
                    )

        # td_first_exposure_flag: 1.0 when user has never seen this advertiser
        # (no mapping, empty seq_c, or seq_c present but no advertiser matches)
        out[~has_exposures, context_offset + 0] = 1.0

        # ── Context group: ad load + distinct advertisers (cols 13-14) ──
        if ts_key_c in batch:
            lengths_c = batch["seq_c_len"].astype(np.int64)
            total_len_c = int(lengths_c.sum())
            if total_len_c > 0:
                flat_ts_c = np.concatenate(batch[ts_key_c]).astype(np.int64)
                flat_rows_c = np.repeat(np.arange(B, dtype=np.int32), lengths_c)
                valid = flat_ts_c > 0
                if valid.any():
                    flat_gaps_c = np.maximum(imp_ts[flat_rows_c[valid]] - flat_ts_c[valid], 0)
                    recent_mask = flat_gaps_c <= self._ad_load_window_sec
                    recent_rows = flat_rows_c[valid][recent_mask]

                    ad_counts = np.bincount(recent_rows, minlength=B).astype(np.float32)
                    out[:, context_offset + 1] = np.log1p(ad_counts) / self._log_ad_load_den

                    # Distinct advertisers in 24h
                    if adv_key in batch:
                        flat_advs_c = np.concatenate(batch[adv_key]).astype(np.int64)
                        flat_advs_recent = flat_advs_c[valid][recent_mask]
                        if len(flat_advs_recent) > 0:
                            # Pack (row, adv) into single int64, sort, count
                            # unique pairs per row. Multiplier must exceed max
                            # advertiser ID (schema vocab ~10K, use 1M for safety).
                            MUL = 1_000_000
                            pairs = recent_rows.astype(np.int64) * MUL + flat_advs_recent
                            pairs.sort()
                            unique_mask = np.empty(len(pairs), dtype=bool)
                            unique_mask[0] = True
                            unique_mask[1:] = pairs[1:] != pairs[:-1]
                            unique_rows = (pairs[unique_mask] // MUL).astype(np.int32)
                            n_adv_per_row = np.bincount(unique_rows, minlength=B).astype(np.float32)
                            out[:, context_offset + 2] = (
                                np.log1p(n_adv_per_row) / self._log_advertisers_den
                            )

        # ── td_category_share (col 15): vectorized across all domains ──
        total_cat_events = np.zeros(B, dtype=np.float32)
        target_cat_events = np.zeros(B, dtype=np.float32)
        for domain in domain_order:
            cat_fid = self._category_fids.get(domain)
            if cat_fid is None:
                continue
            cat_key = f"{domain}_f{cat_fid}"
            if cat_key not in batch:
                continue
            lengths = batch[f"{domain}_len"].astype(np.int64)
            total_len = int(lengths.sum())
            if total_len == 0:
                continue
            flat_cats = np.concatenate(batch[cat_key]).astype(np.int64)
            flat_rows = np.repeat(np.arange(B, dtype=np.int32), lengths)
            valid = flat_cats > 0
            if not valid.any():
                continue
            total_cat_events += np.bincount(flat_rows[valid], minlength=B).astype(np.float32)
            per_row_tgt = target_cat[flat_rows[valid]]
            cat_match = (flat_cats[valid] == per_row_tgt) & (per_row_tgt > 0)
            if cat_match.any():
                target_cat_events += np.bincount(flat_rows[valid][cat_match], minlength=B).astype(
                    np.float32
                )

        denom = np.maximum(total_cat_events, 1.0)
        out[:, context_offset + 3] = target_cat_events / denom

        batch["temporal_dynamics"] = out


class SeqSessionStateBlock(BatchTransform):
    """Temporal and categorical context features for sequence events.

    Derives per-event categorical IDs from existing batch data:

    - **gap**: absolute bucket-delta between adjacent events (always on)
    - **session**: membership, position, and length from time gaps (always on)
    - **daypart**: 6-bucket time-of-day grouping from hour IDs
    - **weekend**: weekday vs weekend flag from day-of-week IDs
    - **category transition**: same/different category vs adjacent events

    Must run after ``TimeDeltaBucketBlock`` (and ``SeqHourOfDayBlock``,
    ``SeqDayOfWeekBlock`` if daypart/weekend are enabled).

    Parameters
    ----------
    schema
        FeatureSchema — used to discover sequence domains.
    session_threshold_sec
        Per-domain gap (in seconds) above which a new session starts.
        Domains not listed default to 3600.
    n_gap_buckets
        Number of gap buckets (clamp ceiling for adjacent-bucket delta).
    add_daypart
        Derive 6-bucket daypart IDs from ``{domain}_time_hour``.
    add_weekend
        Derive weekend flag from ``{domain}_time_weekday``.
    add_category_transition
        Derive same-prev/same-next/change IDs from a category feature.
    category_slots
        Per-domain slot index into the original seq features list (same
        numbering as ``model.seq_local_writer.category_slots``). Resolved
        to batch keys via the schema. Required when
        `add_category_transition` is True.
    """

    type_key = "seq_session_state"

    _GAP_VOCAB_DEFAULT = 64
    _SS_BINARY_VOCAB = 3  # 0=pad, 1=false, 2=true
    _SS_POS_VOCAB = 7  # 0=pad, 1..6 = bucketed position
    _SS_LEN_VOCAB = 7  # 0=pad, 1..6 = bucketed length
    _DAYPART_VOCAB = 7  # 0=pad, 1=night..6=late night
    _WEEKEND_VOCAB = 3  # 0=pad, 1=weekday, 2=weekend

    def __init__(
        self,
        schema: "FeatureSchema",
        session_threshold_sec: dict[str, int] = None,
        n_gap_buckets: int = 64,
        add_daypart: bool = False,
        add_weekend: bool = False,
        add_category_transition: bool = False,
        category_slots: dict[str, int] = None,
    ) -> None:
        self._domains = schema.seq_domains
        self._session_threshold_sec = session_threshold_sec or {}
        self._n_gap_buckets = n_gap_buckets
        self._add_daypart = add_daypart
        self._add_weekend = add_weekend
        self._add_cat_trans = add_category_transition

        # Resolve category_slots to batch keys via schema
        self._category_features: dict[str, str] = {}
        if add_category_transition:
            raw_slots = category_slots or {}
            if not raw_slots:
                raise ValueError("add_category_transition=true requires category_slots mapping")
            _DOMAIN_ALIASES = (("a", "seq_a"), ("b", "seq_b"), ("c", "seq_c"), ("d", "seq_d"))
            expanded: dict[str, int] = {str(k): int(v) for k, v in raw_slots.items()}
            for short, full in _DOMAIN_ALIASES:
                if short in expanded and full not in expanded:
                    expanded[full] = expanded[short]
            for domain, slot_idx in expanded.items():
                specs = schema.query(
                    f"scope = 'seq' and domain = '{domain}' and source = 'original'"
                )
                if slot_idx < len(specs):
                    self._category_features[domain] = specs[slot_idx].batch_key
        # Reuse bucket boundaries from TimeDeltaBucketBlock for seconds approximation
        bounds = np.concatenate([[0], TimeDeltaBucketBlock.BUCKET_BOUNDARIES]).astype(np.float64)
        self._bounds = bounds

    def output_specs(self) -> list[FeatureSpec]:
        """Return feature specs for all enabled writer-state features."""
        specs = []
        for domain in self._domains:
            prefix = f"{domain}_slw"
            specs.append(
                FeatureSpec(
                    name=f"{prefix}_gap",
                    dtype=Dtype.CATEGORICAL,
                    entity=Entity.USER,
                    dim=1,
                    vocab_size=self._n_gap_buckets,
                    domain=domain,
                    source=Source.METADATA,
                    batch_key=f"{prefix}_gap",
                )
            )
            for suffix in ("ss_prev", "ss_next", "ss_bnd"):
                specs.append(
                    FeatureSpec(
                        name=f"{prefix}_{suffix}",
                        dtype=Dtype.CATEGORICAL,
                        entity=Entity.USER,
                        dim=1,
                        vocab_size=self._SS_BINARY_VOCAB,
                        domain=domain,
                        source=Source.METADATA,
                        batch_key=f"{prefix}_{suffix}",
                    )
                )
            specs.append(
                FeatureSpec(
                    name=f"{prefix}_ss_pos",
                    dtype=Dtype.CATEGORICAL,
                    entity=Entity.USER,
                    dim=1,
                    vocab_size=self._SS_POS_VOCAB,
                    domain=domain,
                    source=Source.METADATA,
                    batch_key=f"{prefix}_ss_pos",
                )
            )
            specs.append(
                FeatureSpec(
                    name=f"{prefix}_ss_len",
                    dtype=Dtype.CATEGORICAL,
                    entity=Entity.USER,
                    dim=1,
                    vocab_size=self._SS_LEN_VOCAB,
                    domain=domain,
                    source=Source.METADATA,
                    batch_key=f"{prefix}_ss_len",
                )
            )
            if self._add_daypart:
                specs.append(
                    FeatureSpec(
                        name=f"{prefix}_daypart",
                        dtype=Dtype.CATEGORICAL,
                        entity=Entity.USER,
                        dim=1,
                        vocab_size=self._DAYPART_VOCAB,
                        domain=domain,
                        source=Source.METADATA,
                        batch_key=f"{prefix}_daypart",
                    )
                )
            if self._add_weekend:
                specs.append(
                    FeatureSpec(
                        name=f"{prefix}_weekend",
                        dtype=Dtype.CATEGORICAL,
                        entity=Entity.USER,
                        dim=1,
                        vocab_size=self._WEEKEND_VOCAB,
                        domain=domain,
                        source=Source.METADATA,
                        batch_key=f"{prefix}_weekend",
                    )
                )
            # Only emit cat_trans specs for domains that have a resolved category key.
            # Domains without category_slots produce inconsistent empty lists in compute()
            # (L==0 samples append zeros; L>0 samples are skipped), crashing the collator.
            if self._add_cat_trans and domain in self._category_features:
                for suffix in ("cat_same_prev", "cat_same_next", "cat_change"):
                    specs.append(
                        FeatureSpec(
                            name=f"{prefix}_{suffix}",
                            dtype=Dtype.CATEGORICAL,
                            entity=Entity.USER,
                            dim=1,
                            vocab_size=self._SS_BINARY_VOCAB,
                            domain=domain,
                            source=Source.METADATA,
                            batch_key=f"{prefix}_{suffix}",
                        )
                    )
        return specs

    def compute(self, batch: dict[str, Any]) -> None:
        """Compute gap, session, daypart, weekend, and category transition IDs.

        Reads
        -----
        {domain}_time_bucket : list[np.ndarray]
        {domain}_len : np.ndarray, shape ``[B]``
        {domain}_time_hour : list[np.ndarray] (if add_daypart)
        {domain}_time_weekday : list[np.ndarray] (if add_weekend)
        category_features[domain] : list[np.ndarray] (if add_category_transition)

        Writes
        ------
        {domain}_slw_gap, {domain}_slw_ss_*, {domain}_slw_daypart,
        {domain}_slw_weekend, {domain}_slw_cat_same_prev,
        {domain}_slw_cat_same_next, {domain}_slw_cat_change
        """
        for domain in self._domains:
            tb_key = f"{domain}_time_bucket"
            if tb_key not in batch:
                continue
            tb_arrays = batch[tb_key]
            lengths = batch[f"{domain}_len"]
            n_samples = len(tb_arrays)
            total_len = int(lengths.sum())

            prefix = f"{domain}_slw"
            if total_len == 0:
                empty = [np.zeros(0, dtype=np.int32)] * n_samples
                batch[f"{prefix}_gap"] = empty
                for s in ("ss_prev", "ss_next", "ss_bnd", "ss_pos", "ss_len"):
                    batch[f"{prefix}_{s}"] = empty
                if self._add_daypart:
                    batch[f"{prefix}_daypart"] = empty
                if self._add_weekend:
                    batch[f"{prefix}_weekend"] = empty
                if self._add_cat_trans and domain in self._category_features:
                    for s in ("cat_same_prev", "cat_same_next", "cat_change"):
                        batch[f"{prefix}_{s}"] = empty
                continue

            thr = float(self._session_threshold_sec.get(domain, 3600))
            bounds = self._bounds

            gap_out = []
            ss_prev_out = []
            ss_next_out = []
            ss_bnd_out = []
            ss_pos_out = []
            ss_len_out = []
            daypart_out = [] if self._add_daypart else None
            weekend_out = [] if self._add_weekend else None
            _has_cat = self._add_cat_trans and domain in self._category_features
            cat_sp_out = [] if _has_cat else None
            cat_sn_out = [] if _has_cat else None
            cat_ch_out = [] if _has_cat else None

            hour_arrays = batch.get(f"{domain}_time_hour") if self._add_daypart else None
            wday_arrays = batch.get(f"{domain}_time_weekday") if self._add_weekend else None
            cat_key = self._category_features.get(domain)
            cat_arrays = batch.get(cat_key) if cat_key else None

            for i, tb in enumerate(tb_arrays):
                L = len(tb)
                if L == 0:
                    gap_out.append(np.zeros(0, dtype=np.int32))
                    ss_prev_out.append(np.zeros(0, dtype=np.int32))
                    ss_next_out.append(np.zeros(0, dtype=np.int32))
                    ss_bnd_out.append(np.zeros(0, dtype=np.int32))
                    ss_pos_out.append(np.zeros(0, dtype=np.int32))
                    ss_len_out.append(np.zeros(0, dtype=np.int32))
                    if daypart_out is not None:
                        daypart_out.append(np.zeros(0, dtype=np.int32))
                    if weekend_out is not None:
                        weekend_out.append(np.zeros(0, dtype=np.int32))
                    if cat_sp_out is not None:
                        cat_sp_out.append(np.zeros(0, dtype=np.int32))
                        cat_sn_out.append(np.zeros(0, dtype=np.int32))
                        cat_ch_out.append(np.zeros(0, dtype=np.int32))
                    continue

                valid = tb > 0

                # --- Gap IDs ---
                gap = np.zeros(L, dtype=np.int32)
                if L > 1:
                    delta = np.abs(np.diff(tb.astype(np.int32)))
                    gap[1:] = np.clip(delta, 0, self._n_gap_buckets - 1)
                gap_out.append(gap)

                # --- Session state ---
                # Approximate seconds from bucket IDs for session boundary detection
                approx_sec = bounds[np.clip(tb, 0, len(bounds) - 1)]
                # New session: first valid position OR gap > threshold
                new_session = np.zeros(L, dtype=bool)
                new_session[0] = valid[0]
                if L > 1:
                    gap_sec = np.abs(approx_sec[1:] - approx_sec[:-1])
                    new_session[1:] = valid[1:] & (gap_sec > thr)

                # ss_prev: 0=pad, 1=new session, 2=same session
                ss_prev = np.zeros(L, dtype=np.int32)
                ss_prev[valid & new_session] = 1
                ss_prev[valid & ~new_session] = 2

                # ss_next: 0=pad, 1=last in session, 2=continues
                next_valid = np.zeros(L, dtype=bool)
                if L > 1:
                    next_valid[:-1] = valid[1:]
                next_not_new = np.zeros(L, dtype=bool)
                if L > 1:
                    next_not_new[:-1] = ~new_session[1:]
                continues = valid & next_valid & next_not_new
                ss_next = np.zeros(L, dtype=np.int32)
                ss_next[valid & continues] = 2
                ss_next[valid & ~continues] = 1

                # ss_bnd: 0=pad, 1=not boundary, 2=boundary
                is_boundary = new_session | (valid & ~continues)
                ss_bnd = np.zeros(L, dtype=np.int32)
                ss_bnd[valid & is_boundary] = 2
                ss_bnd[valid & ~is_boundary] = 1

                # ss_pos: within-session position, bucketed
                # 0=pad, 1=pos1, 2=pos2, 3=pos3, 4=pos4-5, 5=pos6-10, 6=pos11+
                cum_valid = np.cumsum(valid)
                ns_pos = np.where(new_session & valid)[0]
                session_starts_cum = np.zeros(L, dtype=np.int64)
                if len(ns_pos) > 0:
                    starts = cum_valid[ns_pos] - 1
                    # Segment lengths: [ns_pos[0], ns_pos[1]-ns_pos[0], ..., L-ns_pos[-1]]
                    seg_lens = np.diff(np.concatenate([[0], ns_pos, [L]]))
                    # First segment (before first session start) gets 0, rest get starts values
                    vals = np.concatenate([[0], starts])
                    session_starts_cum = np.repeat(vals, seg_lens)
                pos = (cum_valid - session_starts_cum) * valid.astype(np.int64)
                ss_pos = np.zeros(L, dtype=np.int32)
                ss_pos[valid & (pos == 1)] = 1
                ss_pos[valid & (pos == 2)] = 2
                ss_pos[valid & (pos == 3)] = 3
                ss_pos[valid & (pos >= 4) & (pos <= 5)] = 4
                ss_pos[valid & (pos >= 6) & (pos <= 10)] = 5
                ss_pos[valid & (pos >= 11)] = 6

                # ss_len: session length, bucketed (count only valid positions)
                # 0=pad, 1=len1, 2=len2, 3=len3-4, 4=len5-8, 5=len9-16, 6=len17+
                session_id = np.cumsum(new_session & valid)
                sid = session_id * valid.astype(np.int64)
                max_sid = int(sid.max()) if sid.max() > 0 else 1
                counts = np.bincount(
                    sid, weights=valid.astype(np.float64), minlength=max_sid + 1
                ).astype(np.int64)
                sess_len_raw = counts[sid]
                ss_len = np.zeros(L, dtype=np.int32)
                ss_len[valid & (sess_len_raw == 1)] = 1
                ss_len[valid & (sess_len_raw == 2)] = 2
                ss_len[valid & (sess_len_raw >= 3) & (sess_len_raw <= 4)] = 3
                ss_len[valid & (sess_len_raw >= 5) & (sess_len_raw <= 8)] = 4
                ss_len[valid & (sess_len_raw >= 9) & (sess_len_raw <= 16)] = 5
                ss_len[valid & (sess_len_raw >= 17)] = 6

                ss_prev_out.append(ss_prev)
                ss_next_out.append(ss_next)
                ss_bnd_out.append(ss_bnd)
                ss_pos_out.append(ss_pos)
                ss_len_out.append(ss_len)

                # --- Daypart ---
                if daypart_out is not None and hour_arrays is not None:
                    h = hour_arrays[i]
                    valid_h = h > 0
                    # hour IDs are 1-based (1..24); convert to 0-based hour
                    h0 = np.clip(h - 1, 0, 23)
                    dp = np.zeros(L, dtype=np.int32)
                    dp[valid_h] = 1  # default: night (0-5)
                    dp[valid_h & (h0 >= 6) & (h0 <= 10)] = 2  # morning
                    dp[valid_h & (h0 >= 11) & (h0 <= 13)] = 3  # midday
                    dp[valid_h & (h0 >= 14) & (h0 <= 17)] = 4  # afternoon
                    dp[valid_h & (h0 >= 18) & (h0 <= 21)] = 5  # evening
                    dp[valid_h & (h0 >= 22)] = 6  # late night
                    daypart_out.append(dp)

                # --- Weekend ---
                if weekend_out is not None and wday_arrays is not None:
                    wd = wday_arrays[i]
                    valid_w = wd > 0
                    # weekday IDs: 1=Thursday (Unix epoch), convert to Monday=0
                    dow_mon0 = (np.clip(wd - 1, 0, 6) + 3) % 7
                    wknd = np.zeros(L, dtype=np.int32)
                    wknd[valid_w] = 1  # weekday
                    wknd[valid_w & (dow_mon0 >= 5)] = 2  # weekend (Sat/Sun)
                    weekend_out.append(wknd)

                # --- Category transition ---
                if cat_sp_out is not None and cat_arrays is not None:
                    cat = cat_arrays[i].astype(np.int32)
                    valid_cat = cat > 0
                    prev_cat = np.zeros(L, dtype=np.int32)
                    nxt_cat = np.zeros(L, dtype=np.int32)
                    if L > 1:
                        prev_cat[1:] = cat[:-1]
                        nxt_cat[:-1] = cat[1:]
                    valid_prev = valid_cat & (prev_cat > 0)
                    valid_next = valid_cat & (nxt_cat > 0)

                    same_prev = np.zeros(L, dtype=np.int32)
                    same_prev[valid_prev & (cat == prev_cat)] = 2
                    same_prev[valid_prev & (cat != prev_cat)] = 1

                    same_next = np.zeros(L, dtype=np.int32)
                    same_next[valid_next & (cat == nxt_cat)] = 2
                    same_next[valid_next & (cat != nxt_cat)] = 1

                    cat_change = np.zeros(L, dtype=np.int32)
                    cat_change[valid_prev & (cat != prev_cat)] = 2
                    cat_change[valid_prev & (cat == prev_cat)] = 1

                    cat_sp_out.append(same_prev)
                    cat_sn_out.append(same_next)
                    cat_ch_out.append(cat_change)

            batch[f"{prefix}_gap"] = gap_out
            batch[f"{prefix}_ss_prev"] = ss_prev_out
            batch[f"{prefix}_ss_next"] = ss_next_out
            batch[f"{prefix}_ss_bnd"] = ss_bnd_out
            batch[f"{prefix}_ss_pos"] = ss_pos_out
            batch[f"{prefix}_ss_len"] = ss_len_out
            if daypart_out is not None:
                batch[f"{prefix}_daypart"] = daypart_out
            if weekend_out is not None:
                batch[f"{prefix}_weekend"] = weekend_out
            if cat_sp_out is not None:
                batch[f"{prefix}_cat_same_prev"] = cat_sp_out
                batch[f"{prefix}_cat_same_next"] = cat_sn_out
                batch[f"{prefix}_cat_change"] = cat_ch_out


class SignalEngineeringBlock(BatchTransform):
    """Derived scalar features from configurable signal definitions.

    Produces four groups of features:

    **Item ratios** — log-scale magnitude comparison between item embedding
    pairs: ``log1p(|mean(num)|) - log1p(|mean(denom)|)``. Captures relative
    energy across embedding spaces.

    **Set sizes** — ``log1p(count_nonzero(slots))`` for multi-value
    categorical features. Measures breadth of user interest.

    **Category overlaps** — binary: does the target item's categorical ID
    appear anywhere in the user's behavioral sequence?

    **PCA compression** — projects high-dimensional dense features onto their
    top-k principal components learned during a fitting phase. Replaces the
    original feature with a compact representation.

    Only PCA requires fitting; ratios, set sizes, and overlaps are stateless.

    Parameters
    ----------
    schema
        FeatureSchema — used to locate features by name and resolve
        batch keys / column ranges.
    item_ratios
        Pairs of ``[numerator_suffix, denominator_suffix]``. Suffixes are
        resolved to ``item_cont_{suffix}`` keys. Emits one scalar per pair.
    set_size_features
        Names of multi-value categorical features. Emits
        ``log1p(n_nonzero)`` per feature.
    category_overlaps
        Each entry has keys ``item_feat`` (scalar categorical),
        ``seq_feat`` (sequence sideinfo key), ``name`` (output suffix).
        Emits 1.0 if item ID is found in the sequence, else 0.0.
    pca_features
        Each entry has keys ``feature`` (dense feature name) and
        ``n_components`` (output dimensionality). Fit collects up to
        `pca_max_rows` samples, computes SVD (dim <= 50) or
        eigendecomposition (dim > 50), then projects at inference.
    pca_max_rows
        Cap on rows collected during the pre-scan fitting phase.
    """

    type_key = "signal_engineering"

    def __init__(
        self,
        schema: FeatureSchema,
        item_ratios: list[list[str]] = None,
        set_size_features: list[str] = None,
        category_overlaps: list[dict[str, str]] = None,
        pca_features: list[dict[str, Any]] = None,
        pca_max_rows: int = 100_000,
    ) -> None:
        self._schema = schema
        self._item_ratios = item_ratios or []
        self._set_size_features = set_size_features or []
        self._category_overlaps = category_overlaps or []
        self._pca_features = pca_features or []
        self._pca_max_rows = pca_max_rows

        # PCA state (populated during fit)
        self._pca_components: dict[str, np.ndarray] = {}
        self._pca_means: dict[str, np.ndarray] = {}
        self._pca_buffers: dict[str, list[np.ndarray]] = {
            p["feature"]: [] for p in self._pca_features
        }
        self._pca_rows_collected = 0

        # Resolve col_range for set-size features at init time
        self._set_size_ranges: list[tuple[str, str, int, int]] = []
        for feat_name in self._set_size_features:
            spec = schema._specs.get(feat_name)
            if spec is not None and spec.col_range is not None:
                self._set_size_ranges.append(
                    (feat_name, spec.batch_key, spec.col_range[0], spec.col_range[1])
                )

        # Resolve item_cat col_range for category overlaps
        self._overlap_specs: list[dict[str, Any]] = []
        for ov in self._category_overlaps:
            item_spec = schema._specs.get(ov["item_feat"])
            if item_spec is None:
                continue
            self._overlap_specs.append(
                {
                    "item_batch_key": item_spec.batch_key,
                    "item_col_start": item_spec.col_range[0] if item_spec.col_range else 0,
                    "seq_feat": ov["seq_feat"],
                    "name": ov["name"],
                }
            )

    def output_specs(self) -> list[FeatureSpec]:
        """Declare all emitted features."""
        specs = []

        for num_feat, denom_feat in self._item_ratios:
            name = f"sig_ratio_{num_feat}_over_{denom_feat}"
            specs.append(
                FeatureSpec(
                    name=name,
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.ITEM,
                    dim=1,
                    source=Source.DERIVED,
                    batch_key=name,
                )
            )

        for feat_name in self._set_size_features:
            name = f"sig_setsize_{feat_name}"
            specs.append(
                FeatureSpec(
                    name=name,
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.USER,
                    dim=1,
                    source=Source.DERIVED,
                    batch_key=name,
                )
            )

        for ov in self._category_overlaps:
            name = f"sig_overlap_{ov['name']}"
            specs.append(
                FeatureSpec(
                    name=name,
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.USER,
                    dim=1,
                    source=Source.DERIVED,
                    batch_key=name,
                )
            )

        for pca_cfg in self._pca_features:
            name = f"sig_pca_{pca_cfg['feature']}"
            specs.append(
                FeatureSpec(
                    name=name,
                    dtype=Dtype.NUMERICAL,
                    entity=Entity.USER,
                    dim=pca_cfg["n_components"],
                    source=Source.DERIVED,
                    batch_key=name,
                )
            )

        return specs

    # ── Fitting (PCA only) ──

    def fit_columns(self) -> list[str]:
        """Parquet columns needed for PCA fitting."""
        if not self._pca_features:
            return []
        cols = []
        for pca_cfg in self._pca_features:
            spec = self._schema._specs.get(pca_cfg["feature"])
            if spec is not None and spec.source_col:
                cols.append(spec.source_col)
        return cols

    @property
    def fit_saturated(self) -> bool:
        """Whether enough rows have been collected for PCA fitting."""
        return self._pca_rows_collected >= self._pca_max_rows

    def partial_fit(self, batch) -> None:
        """Collect samples for PCA computation."""
        if self._pca_rows_collected >= self._pca_max_rows:
            return

        B = batch.num_rows
        row_idx = np.arange(B, dtype=np.intp)

        for pca_cfg in self._pca_features:
            feat = pca_cfg["feature"]
            spec = self._schema._specs.get(feat)
            if spec is None or spec.source_col is None:
                continue
            col_idx = batch.schema.get_field_index(spec.source_col)
            if col_idx < 0:
                continue
            col = batch.column(col_idx)
            # PyArrow list columns: use offsets+values (same as RSSCBlock)
            offs = col.offsets.to_numpy()
            vals = col.values.to_numpy()
            if len(vals) == 0:
                continue
            dim = spec.dim
            src_off = spec.source_offset
            starts = offs[row_idx] + src_off
            lengths = offs[row_idx + 1] - offs[row_idx] - src_off
            use = np.minimum(np.maximum(lengths, 0), dim)
            idx_2d = starts[:, None] + np.arange(dim)[None, :]
            mask = np.arange(dim)[None, :] < use[:, None]
            idx_2d = np.where(mask, idx_2d, 0)
            chunk = vals[idx_2d].astype(np.float32)
            chunk[~mask] = 0.0

            remaining = self._pca_max_rows - self._pca_rows_collected
            self._pca_buffers[feat].append(chunk[:remaining])

        self._pca_rows_collected += B

    def finish_fit(self) -> None:
        """Compute PCA components from collected samples."""
        for pca_cfg in self._pca_features:
            feat = pca_cfg["feature"]
            n_comp = pca_cfg["n_components"]
            buffers = self._pca_buffers[feat]
            if not buffers:
                continue
            data = np.concatenate(buffers, axis=0)
            del buffers[:]

            mean = data.mean(axis=0)
            centered = data - mean
            if centered.shape[1] <= 50:
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                components = Vt[:n_comp]
            else:
                cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = np.argsort(eigenvalues)[::-1][:n_comp]
                components = eigenvectors[:, idx].T

            self._pca_components[feat] = components.astype(np.float32)
            self._pca_means[feat] = mean.astype(np.float32)

        self._pca_buffers = {}

    def fit_state(self) -> dict[str, Any] | None:
        """Serialize fitted PCA components and means for checkpointing."""
        if not self._pca_components:
            return None
        return {
            "components": {k: v.tolist() for k, v in self._pca_components.items()},
            "means": {k: v.tolist() for k, v in self._pca_means.items()},
        }

    def load_fit_state(self, state: dict[str, Any]) -> None:
        """Restore PCA components and means from a saved state dict."""
        self._pca_components = {
            k: np.array(v, dtype=np.float32) for k, v in state["components"].items()
        }
        self._pca_means = {k: np.array(v, dtype=np.float32) for k, v in state["means"].items()}
        self._pca_rows_collected = self._pca_max_rows

    # ── Compute ──

    def compute(self, batch: dict[str, Any]) -> None:
        """Emit all derived features into the batch.

        Reads
        -----
        item_cat, user_cat : packed categorical tensors
        user_cont_f131, user_cont_f130 : dense feature arrays
        item_cont_f124, item_cont_f127, item_cont_f128 : item dense arrays
        seq_a_f40, seq_d_f17 : sequence sideinfo (list of arrays)

        Writes
        ------
        sig_ratio_*, sig_setsize_*, sig_overlap_*, sig_pca_* : derived scalars
        """
        B = len(batch["label"])

        # ── Item ratios ──
        for num_feat, denom_feat in self._item_ratios:
            num_key = f"item_cont_{num_feat}" if not num_feat.startswith("item_cont_") else num_feat
            den_key = (
                f"item_cont_{denom_feat}" if not denom_feat.startswith("item_cont_") else denom_feat
            )
            if num_key not in batch or den_key not in batch:
                name = f"sig_ratio_{num_feat}_over_{denom_feat}"
                batch[name] = np.zeros((B, 1), dtype=np.float32)
                continue
            num_arr = batch[num_key].astype(np.float64)
            den_arr = batch[den_key].astype(np.float64)
            num_mean = num_arr.mean(axis=-1, keepdims=True)
            den_mean = den_arr.mean(axis=-1, keepdims=True)
            ratio = np.log1p(np.abs(num_mean)) - np.log1p(np.abs(den_mean))
            name = f"sig_ratio_{num_feat}_over_{denom_feat}"
            batch[name] = ratio.astype(np.float32)

        # ── Set sizes ──
        for feat_name, batch_key, col_start, col_end in self._set_size_ranges:
            cat_data = batch[batch_key]  # [B, total_cat_dim]
            slot_data = cat_data[:, col_start:col_end]  # [B, dim]
            counts = (slot_data != 0).sum(axis=-1, keepdims=True).astype(np.float32)
            out_name = f"sig_setsize_{feat_name}"
            batch[out_name] = np.log1p(counts)

        # ── Category overlaps ──
        for ov_spec in self._overlap_specs:
            out_name = f"sig_overlap_{ov_spec['name']}"
            seq_key = ov_spec["seq_feat"]
            if seq_key not in batch:
                batch[out_name] = np.zeros((B, 1), dtype=np.float32)
                continue

            item_cat = batch[ov_spec["item_batch_key"]]  # [B, total_item_cat_dim]
            col_start = ov_spec["item_col_start"]
            item_ids = item_cat[:, col_start]  # [B] — single-dim categorical

            seq_list = batch[seq_key]  # list of B arrays, each [seq_len]
            result = np.zeros((B, 1), dtype=np.float32)
            for i in range(B):
                target_id = int(item_ids[i])
                if target_id <= 0:
                    continue
                seq_arr = seq_list[i]
                if len(seq_arr) > 0 and np.any(seq_arr == target_id):
                    result[i, 0] = 1.0
            batch[out_name] = result

        # ── PCA compression ──
        for pca_cfg in self._pca_features:
            feat = pca_cfg["feature"]
            out_name = f"sig_pca_{feat}"
            n_comp = pca_cfg["n_components"]

            if feat not in self._pca_components or feat not in batch:
                batch[out_name] = np.zeros((B, n_comp), dtype=np.float32)
                continue

            data = batch[feat].astype(np.float32)
            centered = data - self._pca_means[feat]
            projected = centered @ self._pca_components[feat].T  # [B, n_comp]
            batch[out_name] = projected


class AssemblyBlock(BatchTransform):
    """Terminal validation block. Runs last, before collator.

    Validates that all batch keys are either registered in the schema
    or in the known pass-through set. Dropped keys are removed from both
    the batch and the schema so downstream blocks never see them.
    """

    type_key = "assembly"

    PASSTHROUGH = frozenset({"label", "timestamp", "user_id", "item_id"})

    def __init__(self, schema: "FeatureSchema", drop: list[str] = None) -> None:
        self._schema = schema
        self._drop = set(drop or [])
        for spec in list(schema.query("name matches '*'")):
            if spec.batch_key in self._drop:
                schema.unregister(spec.name)
        self._registered_keys = frozenset(
            s.batch_key for s in schema.query("name matches '*'") if s.batch_key
        )

    def output_specs(self) -> list[FeatureSpec]:
        """No new features; this block only validates and prunes."""
        return []

    def compute(self, batch: dict[str, Any]) -> None:
        """Drop unregistered keys and validate the remainder.

        Reads
        -----
        All keys in `batch`.

        Writes
        ------
        Removes keys listed in ``drop`` config. Raises on any key not
        registered in FeatureSchema and not in PASSTHROUGH.
        """
        for key in list(batch):
            if key in self._drop:
                del batch[key]
            elif key not in self._registered_keys and key not in self.PASSTHROUGH:
                raise ValueError(
                    f"AssemblyBlock: batch key '{key}' is not registered in "
                    f"FeatureSchema and not in PASSTHROUGH. Register it or "
                    f"add it to 'drop' config."
                )
