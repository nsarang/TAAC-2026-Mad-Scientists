"""Map-style advertising dataset with addressable indices and composable blocks."""

from __future__ import annotations

import glob
import inspect
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from core.data.blocks import BatchTransform
from core.data.schema import (
    DatasetSchema,
    Dtype,
    Entity,
    FeatureSchema,
    FeatureSpec,
    Source,
    build_feature_schema,
)

LOG = logging.getLogger(__name__)


class RowIndex:
    """Flat addressable index over all rows in a Parquet dataset.

    Parameters
    ----------
    dataset_path
        Directory containing ``*.parquet`` files.
    """

    _LRU_MAX = 64

    def __init__(self, dataset_path: str | Path) -> None:
        dataset_path = Path(dataset_path)
        files = sorted(glob.glob(str(dataset_path / "*.parquet")))
        if not files:
            # Cache tap stores shards under rank_*/worker_*.
            files = sorted(glob.glob(str(dataset_path / "**" / "*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No .parquet files in {dataset_path}")

        # Read metadata without holding file descriptors open
        self._pf_cache: dict[str, pq.ParquetFile] = {}
        self._rg_meta: list[tuple[str, int, int]] = []
        for f in files:
            meta = pq.read_metadata(f)
            for i in range(meta.num_row_groups):
                nrows = meta.row_group(i).num_rows
                self._rg_meta.append((f, i, nrows))

        counts = np.array([nrows for _, _, nrows in self._rg_meta], dtype=np.int64)
        self._cum_rows = np.zeros(len(self._rg_meta) + 1, dtype=np.int64)
        self._cum_rows[1:] = np.cumsum(counts)

        self._total_rows = int(self._cum_rows[-1])
        # Stored as uint32 to halve persistent memory. Unsigned arithmetic
        # wraps silently, so cast to int64 before doing any subtraction.
        self._timestamps = self._read_timestamps(files)

    def _read_timestamps(self, files: list[str]) -> np.ndarray:
        chunks = []
        for f in files:
            table = pq.read_table(f, columns=["timestamp"])
            chunks.append(table.column("timestamp").to_numpy())
        return np.concatenate(chunks).astype(np.uint32)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pf_cache"] = {}
        return state

    def __len__(self) -> int:
        return self._total_rows

    @property
    def timestamps(self) -> np.ndarray:
        """Per-row timestamps, shape ``[total_rows]``."""
        return self._timestamps

    @property
    def rg_meta(self) -> list[tuple[str, int, int]]:
        """List of ``(file_path, rg_idx, num_rows)`` for each row group."""
        return self._rg_meta

    @property
    def cum_rows(self) -> np.ndarray:
        """Cumulative row counts, shape ``[num_rgs + 1]``. Entry *i* is the global start row of RG *i*."""
        return self._cum_rows

    @property
    def rg_boundaries(self) -> list[tuple[int, int]]:
        """List of ``(start_idx, end_idx)`` for each row group."""
        return [
            (int(self._cum_rows[i]), int(self._cum_rows[i + 1])) for i in range(len(self._rg_meta))
        ]

    def open_file(self, file_path: str) -> pq.ParquetFile:
        """Return a cached ParquetFile handle with LRU eviction."""
        pf = self._pf_cache.pop(file_path, None)
        if pf is not None:
            # Move to end (most recently used)
            self._pf_cache[file_path] = pf
            return pf
        pf = pq.ParquetFile(file_path)
        self._pf_cache[file_path] = pf
        while len(self._pf_cache) > self._LRU_MAX:
            self._pf_cache.pop(next(iter(self._pf_cache)))
        return pf

    def locate(self, idx: int) -> tuple[str, int, int]:
        """Map a flat index to ``(file_path, rg_idx, row_within_rg)``.

        Parameters
        ----------
        idx
            Flat row index in ``[0, len(self))``.
        """
        if idx < 0 or idx >= self._total_rows:
            raise IndexError(f"Index {idx} out of range [0, {self._total_rows})")
        rg_i = int(np.searchsorted(self._cum_rows[1:], idx, side="right"))
        file_path, rg_idx, _ = self._rg_meta[rg_i]
        row_offset = idx - int(self._cum_rows[rg_i])
        return file_path, rg_idx, row_offset

    def _partition_by_rg(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Core vectorized grouping: searchsorted + argsort + breaks.

        Returns
        -------
        tuple of (order, sorted_rg, breaks)
            ``order[breaks[i]:breaks[i+1]]`` gives the positions into
            `indices` that belong to one row group.
        """
        rg_ids = np.searchsorted(self._cum_rows[1:], indices, side="right")
        order = np.argsort(rg_ids, kind="stable")
        sorted_rg = rg_ids[order]
        breaks = np.concatenate([[0], np.flatnonzero(np.diff(sorted_rg)) + 1, [len(sorted_rg)]])
        return order, sorted_rg, breaks

    def group_by_rg(self, indices: np.ndarray) -> list[np.ndarray]:
        """Group flat indices by row group, fully vectorized.

        Parameters
        ----------
        indices
            1D array of flat row indices.

        Returns
        -------
        list of np.ndarray
            Each element contains the subset of `indices` that falls within
            one row group, in their original relative order.
        """
        order, _, breaks = self._partition_by_rg(indices)
        return [indices[order[breaks[i] : breaks[i + 1]]] for i in range(len(breaks) - 1)]

    def locate_batch(self, indices: np.ndarray) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
        """Group flat indices by row group with file metadata, fully vectorized.

        Parameters
        ----------
        indices
            1D array of flat row indices.

        Returns
        -------
        list of (file_path, rg_idx, row_offsets, batch_positions)
            Each entry has the within-RG offsets and the positions in the
            original `indices` array (for reassembly into batch order).
        """
        order, sorted_rg, breaks = self._partition_by_rg(indices)

        results = []
        for i in range(len(breaks) - 1):
            chunk = order[breaks[i] : breaks[i + 1]]
            rg_i = int(sorted_rg[breaks[i]])
            file_path, rg_idx, _ = self._rg_meta[rg_i]
            row_offsets = (indices[chunk] - self._cum_rows[rg_i]).astype(np.int64)
            results.append((file_path, rg_idx, row_offsets, chunk))
        return results


class AdDataset(Dataset):
    """Addressable map-style advertising dataset.

    Blocks are instantiated internally from registries, following the same
    pattern as the diagnostics system. The keys of `blocks` determine
    which blocks are active; each value is that block's config dict (empty
    if no config needed).

    Parameters
    ----------
    dataset_path
        Directory containing ``*.parquet`` files.
    schema_path
        Path to ``schema.json``.
    blocks
        Blocks to activate, keyed by type_key. Values are per-block config
        dicts. E.g. ``{"rssc": {"clip_value": 3.0}, "time_bucket": {}}``.
    clip_vocab
        If True, clip OOB int feature values to 0.
    is_training
        If True, derive label from label_type==2; else all-zeros.

    Notes
    -----
    Sequence feature layout is not set here — the collator assigns it when
    it receives the schema. This keeps the dataset format-agnostic.

    Blocks that require fitting must be fitted after construction via
    ``fit_blocks(indices)`` before the dataset is used. This separates
    the train/val split decision from the dataset itself.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        schema_path: str | Path,
        blocks: dict[str, dict] = None,
        blocks_order: list[str] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        split_f129: bool = True,
    ) -> None:
        self._dataset_path = Path(dataset_path)
        self._schema = DatasetSchema(schema_path)
        self._index = RowIndex(dataset_path)
        self._clip_vocab = clip_vocab
        self._is_training = is_training

        self._split_f129 = split_f129
        self._feature_schema = build_feature_schema(self._schema, split_f129=split_f129)

        fs = self._feature_schema
        seq_specs = fs.query("scope = 'seq'")
        LOG.info(
            "AdDataset: %d rows | %d static + %d seq features | domains=%s",
            len(self._index),
            len(fs.query("scope = 'static'")),
            len(seq_specs),
            sorted({s.domain for s in seq_specs if s.domain}),
        )

        self._unique_rgs: set[tuple[str, int]] = set()

        # Register pipeline timing metadata so downstream filters exclude it
        fs.register(
            FeatureSpec(
                name="_meta_timing",
                dtype=Dtype.NUMERICAL,
                entity=Entity.CONTEXT,
                dim=6,
                source=Source.METADATA,
                batch_key="_meta_timing",
                col_range=(0, 6),
            )
        )

        blocks = blocks or {}
        self._blocks = self._build_blocks(blocks, blocks_order)

    def _build_blocks(
        self, configs: dict[str, dict], blocks_order: list[str] = None
    ) -> list[BatchTransform]:
        """Look up and instantiate blocks (no fitting).

        Parameters
        ----------
        configs
            Block configs keyed by type_key.
        blocks_order
            Explicit execution order. When provided, blocks are sorted to
            match this sequence and every active block must appear in the
            list (raises ValueError otherwise).
        """
        registry = BatchTransform.registry
        active_keys = [k for k in configs if k in registry and configs[k] is not None]

        if blocks_order:
            order_set = set(blocks_order)
            missing = [k for k in active_keys if k not in order_set]
            if missing:
                raise ValueError(
                    f"blocks_order is specified but does not include active blocks: {missing}. "
                    f"Every block in 'blocks' must appear in 'blocks_order'."
                )
            rank = {k: i for i, k in enumerate(blocks_order)}
            active_keys.sort(key=lambda k: rank[k])
        else:
            # Default: dict order, assembly always last
            active_keys.sort(key=lambda k: (k == "assembly",))

        if not active_keys:
            return []

        blocks = []
        for key in active_keys:
            cls = registry[key]
            cfg = configs[key]
            params = set(inspect.signature(cls.__init__).parameters) - {"self"}
            init_kwargs = {k: v for k, v in cfg.items() if k in params}
            if "schema" in params:
                init_kwargs["schema"] = self._feature_schema
            block = cls(**init_kwargs)
            for spec in block.output_specs():
                self._feature_schema.register(spec)
            blocks.append(block)

        return blocks

    def fit_blocks(self, indices: np.ndarray) -> None:
        """Fit blocks that require a pre-scan over training data.

        Groups train RGs by file and uses ``iter_batches`` with threaded
        decompression for pipelined I/O. Each RG batch is filtered to only
        train rows before passing to blocks. Stops as soon as all blocks
        report ``fit_saturated``.

        Parameters
        ----------
        indices
            Flat row indices (e.g. from the train sampler) identifying which
            RGs belong to the training split.
        """
        blocks_to_fit = [b for b in self._blocks if b.fit_columns()]
        if not blocks_to_fit:
            return

        cols: set[str] = set()
        for b in blocks_to_fit:
            cols.update(b.fit_columns())
        columns = sorted(cols)

        # Identify which RGs contain train data, grouped by file for bulk reads
        rg_ids = np.searchsorted(self._index.cum_rows[1:], indices, side="right")
        train_rg_set = set(rg_ids.tolist())
        rg_meta = self._index.rg_meta

        # Precompute within-RG train offsets for filtering
        sorted_idx = np.argsort(indices)
        sorted_indices = indices[sorted_idx]
        sorted_rg_ids = rg_ids[sorted_idx]
        rg_offsets: dict[int, np.ndarray] = {}
        change_points = np.where(np.diff(sorted_rg_ids) != 0)[0] + 1
        for chunk in np.split(np.arange(len(sorted_indices)), change_points):
            if len(chunk) == 0:
                continue
            rg_i = int(sorted_rg_ids[chunk[0]])
            within_rg = (sorted_indices[chunk] - int(self._index.cum_rows[rg_i])).astype(np.int64)
            rg_offsets[rg_i] = within_rg

        # Group train RGs by file (for fragment subsetting) and record scan order
        # in a single pass so the two structures can't diverge.
        file_rg_groups: list[tuple[str, list[int]]] = []
        scan_rg_order: list[int] = []
        scan_rg_cum: list[int] = [0]
        for rg_i in range(len(rg_meta)):
            if rg_i not in train_rg_set:
                continue
            file_path, rg_idx, nrows = rg_meta[rg_i]
            if file_rg_groups and file_rg_groups[-1][0] == file_path:
                file_rg_groups[-1][1].append(rg_idx)
            else:
                file_rg_groups.append((file_path, [rg_idx]))
            scan_rg_order.append(rg_i)
            scan_rg_cum.append(scan_rg_cum[-1] + nrows)

        # Build a dataset scanner with fragment subsetting for parallel I/O
        dataset_obj = ds.dataset([fp for fp, _ in file_rg_groups], format="parquet")
        fragments = list(dataset_obj.get_fragments())
        subsetted_frags = []
        for frag, (_, rg_indices) in zip(fragments, file_rg_groups):
            subsetted_frags.append(frag.subset(row_group_ids=rg_indices))
        sub_dataset = ds.FileSystemDataset(
            subsetted_frags,
            schema=dataset_obj.schema,
            format=ds.ParquetFileFormat(),
        )

        _FLUSH_ROWS = 80_000
        t_io, n_batches = 0.0, 0
        t_block: dict[str, float] = {b.type_key: 0.0 for b in blocks_to_fit}
        rows_processed = 0
        pending: list[pa.RecordBatch] = []
        pending_rows = 0
        t_last_heartbeat = time.perf_counter()

        # Track absolute row position to map scanner batches back to RGs
        rows_seen_in_scan = 0
        rg_cursor = 0

        def _flush_pending() -> None:
            nonlocal pending, pending_rows
            if not pending:
                return
            merged = pa.concat_batches(pending)
            pending.clear()
            pending_rows = 0
            for b in blocks_to_fit:
                tb = time.perf_counter()
                b.partial_fit(merged)
                t_block[b.type_key] += time.perf_counter() - tb

        t0 = time.perf_counter()
        scanner = sub_dataset.scanner(columns=columns, batch_readahead=8, fragment_readahead=4)
        for batch in scanner.to_batches():
            t_io += time.perf_counter() - t0
            n_batches += 1
            batch_rows = batch.num_rows

            # A scanner batch may span part of one RG (when the RG is split
            # across multiple batches) or exactly one RG. Use cumulative row
            # counts to find which RG this batch starts in, then filter to
            # only the train offsets that fall within this batch's row range.
            batch_start = rows_seen_in_scan
            batch_end = batch_start + batch_rows
            rows_seen_in_scan = batch_end

            # Advance cursor to the RG containing batch_start
            while rg_cursor < len(scan_rg_order) and scan_rg_cum[rg_cursor + 1] <= batch_start:
                rg_cursor += 1

            # Collect train offsets from all RGs that overlap this batch
            take_indices: list[np.ndarray] = []
            cursor = rg_cursor
            while cursor < len(scan_rg_order) and scan_rg_cum[cursor] < batch_end:
                rg_i = scan_rg_order[cursor]
                rg_start = scan_rg_cum[cursor]
                offsets = rg_offsets[rg_i]
                # Translate within-RG offsets to within-batch offsets
                batch_offsets = offsets + (rg_start - batch_start)
                # Keep only offsets that fall within this batch
                valid = (batch_offsets >= 0) & (batch_offsets < batch_rows)
                if valid.any():
                    take_indices.append(batch_offsets[valid])
                cursor += 1

            if take_indices:
                all_indices = (
                    np.concatenate(take_indices) if len(take_indices) > 1 else take_indices[0]
                )
                batch = batch.take(all_indices)
                rows_processed += batch.num_rows
                pending.append(batch)
                pending_rows += batch.num_rows

            if pending_rows >= _FLUSH_ROWS:
                _flush_pending()

            now = time.perf_counter()
            if now - t_last_heartbeat >= 120:
                block_str = " | ".join(f"{k} {v:.1f}s" for k, v in t_block.items())
                LOG.info(
                    "fit_blocks: %d/%d train rows scanned (%d batches) | I/O %.1fs | %s",
                    rows_processed,
                    len(indices),
                    n_batches,
                    t_io,
                    block_str,
                )
                t_last_heartbeat = now

            if all(b.fit_saturated for b in blocks_to_fit):
                break
            t0 = time.perf_counter()

        _flush_pending()

        for b in blocks_to_fit:
            t0 = time.perf_counter()
            b.finish_fit()
            t_block[b.type_key] += time.perf_counter() - t0

        LOG.info(
            "fit_blocks: %.2fs (I/O %.2fs, %d batches, %d rows | %s)",
            t_io + sum(t_block.values()),
            t_io,
            n_batches,
            rows_processed,
            ", ".join(f"{k} {v:.2f}s" for k, v in t_block.items()),
        )

    def fit_state(self) -> dict[str, Any]:
        """Collect fitted state from all blocks for checkpoint serialization."""
        state = {}
        for b in self._blocks:
            s = b.fit_state()
            if s is not None:
                state[b.type_key] = s
        return state

    def load_fit_state(self, state: dict[str, Any]) -> None:
        """Restore block fit state from checkpoint, skipping the I/O scan."""
        for b in self._blocks:
            if b.type_key in state:
                b.load_fit_state(state[b.type_key])

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.get_batch(np.array([idx], dtype=np.int64))

    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        """Batch fetch for DataLoader — returns assembled batch dict."""
        return self.get_batch(np.array(indices, dtype=np.int64))

    @property
    def schema(self) -> DatasetSchema:
        """The parsed feature schema."""
        return self._schema

    @property
    def feature_schema(self) -> FeatureSchema:
        """Typed feature schema for model init and feature routing."""
        return self._feature_schema

    @property
    def index(self) -> RowIndex:
        """The row index."""
        return self._index

    def get_batch(self, indices: np.ndarray) -> dict[str, Any]:
        """Retrieve a batch of samples by flat indices.

        Parameters
        ----------
        indices
            1D array of flat row indices.

        Returns
        -------
        dict
            Named arrays/lists. Sequences are lists of per-sample dicts.
        """
        t0 = time.perf_counter()
        B = len(indices)
        groups = self._index.locate_batch(indices)

        row_results: list[dict[str, Any] | None] = [None] * B
        rows_decompressed = 0

        for file_path, rg_idx, row_offsets, positions in groups:
            pf = self._index.open_file(file_path)
            batch = pf.read_row_group(rg_idx).to_batches()[0]
            rows_decompressed += batch.num_rows
            self._unique_rgs.add((file_path, rg_idx))
            self.extract_rows_vectorized(batch, row_offsets, positions, row_results)

        t_io_ext = time.perf_counter()
        result = self.assemble_batch(row_results, B)
        t_end = time.perf_counter()

        result["_meta_timing"] = np.array(
            [
                (t_io_ext - t0) * 1000,  # io+extract ms
                (t_end - t_io_ext) * 1000,  # assemble ms
                0.0,  # reserved
                len(groups),  # n_rgs
                rows_decompressed,  # total rows decompressed
                len(self._unique_rgs),  # unique RGs seen by this worker
            ],
            dtype=np.float32,
        )
        return result

    def extract_rows_vectorized(
        self,
        batch,
        row_offsets: np.ndarray,
        positions: np.ndarray,
        row_results: list[dict[str, Any] | None],
    ) -> None:
        """Decode a PyArrow RecordBatch into per-row dicts at the given offsets.

        Reads scalar columns directly via numpy indexing. List-type columns
        (multi-valued categoricals, dense features, sequences) are decoded
        through Arrow's offsets/values layout. Sequence features are packed
        as ``{domain: {values, timestamps, length}}`` dicts per row.

        Results are written into `row_results` at `positions` for later
        reassembly by ``_assemble_batch``.
        """
        schema = self._schema
        n = len(positions)
        offsets_arr = row_offsets.astype(np.intp)

        # Timestamp
        ts_col = batch.column(batch.schema.get_field_index("timestamp"))
        ts_np = ts_col.to_numpy()
        timestamps = ts_np[offsets_arr]

        # user_id / item_id
        user_id_col = batch.column(batch.schema.get_field_index("user_id"))
        user_ids = user_id_col.fill_null(0).to_numpy(zero_copy_only=False)[offsets_arr]

        item_id_col = batch.column(batch.schema.get_field_index("item_id"))
        item_ids = item_id_col.fill_null(0).to_numpy(zero_copy_only=False)[offsets_arr]

        # Label
        if self._is_training:
            lt_col = batch.column(batch.schema.get_field_index("label_type"))
            lt_np = lt_col.fill_null(0).to_numpy(zero_copy_only=False)
            labels = (lt_np[offsets_arr] == 2).astype(np.int64)
        else:
            labels = np.zeros(n, dtype=np.int64)

        # user_cat
        total_user_cat_dim = sum(dim for _, _, dim in schema.user_cat)
        user_cat = np.zeros((n, total_user_cat_dim), dtype=np.int32)
        offset = 0
        for fid, vs, dim in schema.user_cat:
            col_name = f"user_int_feats_{fid}"
            col_idx = batch.schema.get_field_index(col_name)
            if col_idx < 0:
                offset += dim
                continue
            col = batch.column(col_idx)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int32)
                vals = arr[offsets_arr]
                vals[vals <= 0] = 0
                if self._clip_vocab and vs > 0:
                    vals[vals >= vs] = 0
                user_cat[:, offset] = vals
            else:
                col_offsets = col.offsets.to_numpy()
                col_values = col.values.to_numpy()
                for i, row_idx in enumerate(offsets_arr):
                    s = int(col_offsets[row_idx])
                    e = int(col_offsets[row_idx + 1])
                    use = min(e - s, dim)
                    if use > 0:
                        chunk = col_values[s : s + use].astype(np.int32)
                        chunk[chunk <= 0] = 0
                        if self._clip_vocab and vs > 0:
                            chunk[chunk >= vs] = 0
                        user_cat[i, offset : offset + use] = chunk
            offset += dim

        # item_cat
        total_item_cat_dim = sum(dim for _, _, dim in schema.item_cat)
        item_cat = np.zeros((n, total_item_cat_dim), dtype=np.int32)
        offset = 0
        for fid, vs, dim in schema.item_cat:
            col_name = f"item_int_feats_{fid}"
            col_idx = batch.schema.get_field_index(col_name)
            if col_idx < 0:
                offset += dim
                continue
            col = batch.column(col_idx)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int32)
                vals = arr[offsets_arr]
                vals[vals <= 0] = 0
                if self._clip_vocab and vs > 0:
                    vals[vals >= vs] = 0
                item_cat[:, offset] = vals
            else:
                col_offsets = col.offsets.to_numpy()
                col_values = col.values.to_numpy()
                for i, row_idx in enumerate(offsets_arr):
                    s = int(col_offsets[row_idx])
                    e = int(col_offsets[row_idx + 1])
                    use = min(e - s, dim)
                    if use > 0:
                        chunk = col_values[s : s + use].astype(np.int32)
                        chunk[chunk <= 0] = 0
                        if self._clip_vocab and vs > 0:
                            chunk[chunk >= vs] = 0
                        item_cat[i, offset : offset + use] = chunk
            offset += dim

        # user_cont — one array per feature
        user_cont_feats: dict[str, np.ndarray] = {}
        for fid, dim in schema.user_cont:
            col_name = f"user_dense_feats_{fid}"
            col_idx = batch.schema.get_field_index(col_name)
            arr = np.zeros((n, dim), dtype=np.float32)
            if col_idx >= 0:
                col = batch.column(col_idx)
                col_offsets = col.offsets.to_numpy()
                col_values = col.values.to_numpy()
                for i, row_idx in enumerate(offsets_arr):
                    s = int(col_offsets[row_idx])
                    e = int(col_offsets[row_idx + 1])
                    use = min(e - s, dim)
                    if use > 0:
                        arr[i, :use] = col_values[s : s + use]
            user_cont_feats[f"user_cont_f{fid}"] = arr

        # item_cont — one array per feature
        item_cont_feats: dict[str, np.ndarray] = {}
        for fid, dim in schema.item_cont:
            col_name = f"item_dense_feats_{fid}"
            col_idx = batch.schema.get_field_index(col_name)
            arr = np.zeros((n, dim), dtype=np.float32)
            if col_idx >= 0:
                col = batch.column(col_idx)
                col_offsets = col.offsets.to_numpy()
                col_values = col.values.to_numpy()
                for i, row_idx in enumerate(offsets_arr):
                    s = int(col_offsets[row_idx])
                    e = int(col_offsets[row_idx + 1])
                    use = min(e - s, dim)
                    if use > 0:
                        arr[i, :use] = col_values[s : s + use]
            if fid == 129:
                if self._split_f129:
                    item_cont_feats["item_cont_f129_emb"] = arr[:, :128]
                    item_cont_feats["item_cont_f129_count"] = arr[:, 128:129]
                else:
                    item_cont_feats["item_cont_f129"] = arr
            else:
                item_cont_feats[f"item_cont_f{fid}"] = arr

        # Sequences: variable-length per domain
        seq_results: dict[str, list[dict[str, Any]]] = {domain: [] for domain in schema.seq_domains}
        for domain in schema.seq_domains:
            cfg = schema.seq_config(domain)
            sideinfo_fids = cfg.sideinfo_fids
            n_feats = len(sideinfo_fids)

            feat_arrays: list[tuple[np.ndarray, np.ndarray]] = []
            for fid in sideinfo_fids:
                col = batch.column(batch.schema.get_field_index(f"{cfg.prefix}_{fid}"))
                feat_arrays.append((col.offsets.to_numpy(), col.values.to_numpy()))

            # Vectorized length computation from first feature
            first_offsets = feat_arrays[0][0]
            lengths = (first_offsets[offsets_arr + 1] - first_offsets[offsets_arr]).astype(np.intp)

            # Resolve timestamp column once
            ts_offsets_np = ts_values_np = None
            if cfg.ts_fid is not None:
                ts_col = batch.column(batch.schema.get_field_index(f"{cfg.prefix}_{cfg.ts_fid}"))
                ts_offsets_np = ts_col.offsets.to_numpy()
                ts_values_np = ts_col.values.to_numpy()

            for i in range(n):
                row_idx = offsets_arr[i]
                actual_len = int(lengths[i])

                values = np.zeros((n_feats, actual_len), dtype=np.int32)
                for feat_i, (c_offsets, c_values) in enumerate(feat_arrays):
                    s = int(c_offsets[row_idx])
                    use_len = min(int(c_offsets[row_idx + 1]) - s, actual_len)
                    if use_len > 0:
                        values[feat_i, :use_len] = c_values[s : s + use_len]
                values[values <= 0] = 0

                ts_arr = None
                if ts_offsets_np is not None:
                    ts_s = int(ts_offsets_np[row_idx])
                    ts_len = min(int(ts_offsets_np[row_idx + 1]) - ts_s, actual_len)
                    ts_arr = np.zeros(actual_len, dtype=np.int64)
                    if ts_len > 0:
                        ts_arr[:ts_len] = ts_values_np[ts_s : ts_s + ts_len]

                seq_results[domain].append(
                    {
                        "values": values,
                        "timestamps": ts_arr,
                        "length": actual_len,
                    }
                )

        # Write results into row_results at the correct positions
        for i, pos in enumerate(positions):
            row = {
                "timestamp": int(timestamps[i]),
                "user_id": int(user_ids[i]),
                "item_id": int(item_ids[i]),
                "label": int(labels[i]),
                "user_cat": user_cat[i],
                "item_cat": item_cat[i],
                "sequences": {domain: seq_results[domain][i] for domain in schema.seq_domains},
            }
            for key, arr in user_cont_feats.items():
                row[key] = arr[i]
            for key, arr in item_cont_feats.items():
                row[key] = arr[i]
            row_results[pos] = row

    def assemble_batch(self, row_results: list[dict[str, Any] | None], B: int) -> dict[str, Any]:
        """Collate per-row dicts into the flat batch format downstream expects.

        Stacks scalars and categoricals into arrays, then explodes the packed
        per-row sequence dicts into the uniform ``{domain}_f{fid}``,
        ``{domain}_len``, ``{domain}_ts`` keys that blocks and collators consume.
        """
        batch: dict[str, Any] = {}

        batch["user_cat"] = np.stack([r["user_cat"] for r in row_results])
        batch["item_cat"] = np.stack([r["item_cat"] for r in row_results])
        batch["label"] = np.array([r["label"] for r in row_results], dtype=np.int64)
        batch["timestamp"] = np.array([r["timestamp"] for r in row_results], dtype=np.int64)
        batch["user_id"] = np.array([r["user_id"] for r in row_results], dtype=np.int64)
        batch["item_id"] = np.array([r["item_id"] for r in row_results], dtype=np.int64)

        # Stack per-feature continuous arrays
        cont_keys = [k for k in row_results[0] if k.startswith(("user_cont_", "item_cont_"))]
        for key in cont_keys:
            batch[key] = np.stack([r[key] for r in row_results])

        # Explode packed seq dicts into the uniform per-feature format that
        # both blocks and collator consume: {domain}_f{fid} = [arr_per_sample],
        # {domain}_len = np.array, {domain}_ts = [arr_per_sample].
        for domain in self._schema.seq_domains:
            samples = [r["sequences"][domain] for r in row_results]
            batch[f"{domain}_len"] = np.array(
                [s["length"] for s in samples],
                dtype=np.int32,
            )
            cfg = self._schema.seq_config(domain)
            for row_idx, fid in enumerate(cfg.sideinfo_fids):
                batch[f"{domain}_f{fid}"] = [s["values"][row_idx, : s["length"]] for s in samples]
            if cfg.ts_fid is not None:
                batch[f"{domain}_ts"] = [
                    s["timestamps"][: s["length"]]
                    if s["timestamps"] is not None
                    else np.zeros(0, dtype=np.int64)
                    for s in samples
                ]

        # Blocks mutate batch in-place
        if self._blocks:
            for block in self._blocks:
                block.compute(batch)

        return batch
