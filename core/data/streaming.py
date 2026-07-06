"""StreamingLoader: IterableDataset that reads full row groups sequentially."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from core.data.collators import Collator
from core.data.dataset import AdDataset


def _passthrough(x):
    return x


class StreamingLoader(IterableDataset):
    """Streams batches by reading full row groups sequentially.

    Each worker takes every Nth RG from its assigned list, reads all rows
    in one extraction call per RG (maximizing vectorization efficiency),
    and yields collated batches.

    When ``shuffle_buffer_size > 0``, extracted rows accumulate in a buffer
    and batches are drawn randomly — suitable for training. RG visit order
    is also shuffled per epoch. When ``shuffle_buffer_size == 0``, rows are
    yielded in file order — suitable for validation.

    Parameters
    ----------
    dataset
        The shared AdDataset (provides extraction and assembly methods).
    indices
        Flat row indices belonging to this split.
    batch_size
        Number of samples per yielded batch.
    collator
        Converts numpy batch dict to torch tensors.
    shuffle_buffer_size
        When > 0, enables random buffered yielding for training.
    seed
        Base seed for RG-order shuffling (incremented per epoch).
    extra_rg_ids
        Val-only RGs to visit despite having no train rows, so the
        `on_rg_loaded` callback can capture their validation rows.
    on_rg_loaded
        Called with (file_path, global_rg_id, arrow_chunk) per RG read.
        Used by the validation cache tap to capture rows during training.
    """

    def __init__(
        self,
        dataset: AdDataset,
        indices: np.ndarray,
        batch_size: int,
        collator: Collator,
        shuffle_buffer_size: int = 0,
        seed: int = 42,
        num_workers: int = 1,
        extra_rg_ids: np.ndarray | None = None,
        on_rg_loaded: Callable[[str, int, Any], None] = None,
    ) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._collator = collator
        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed = seed
        self._num_workers = max(1, num_workers)
        self._on_rg_loaded = on_rg_loaded
        self._epoch = 0

        # Pre-compute which row offsets live in each RG
        index = dataset.index
        rg_ids = np.searchsorted(index.cum_rows[1:], indices, side="right")

        # Sort by RG then split
        order = np.argsort(rg_ids, kind="mergesort")
        sorted_rg_ids = rg_ids[order]
        sorted_indices = indices[order]
        change_points = np.where(np.diff(sorted_rg_ids) != 0)[0] + 1

        self._rg_offsets: dict[int, np.ndarray] = {}
        for chunk in np.split(np.arange(len(sorted_indices)), change_points):
            if len(chunk) == 0:
                continue
            rg_i = int(sorted_rg_ids[chunk[0]])
            within_rg = (sorted_indices[chunk] - int(index.cum_rows[rg_i])).astype(np.int64)
            self._rg_offsets[rg_i] = within_rg

        train_rg_ids = sorted(self._rg_offsets.keys())
        self._global_train_rg_ids = np.asarray(train_rg_ids, dtype=np.int64)
        extra = np.asarray(extra_rg_ids if extra_rg_ids is not None else [], dtype=np.int64)
        self._extra_rg_ids = sorted({int(rg) for rg in extra.tolist()})
        self._all_rgs = sorted(set(train_rg_ids) | set(self._extra_rg_ids))
        self._n_samples = len(indices)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for per-epoch shuffle variation.

        Unlike BatchSampler (which iterates in the main process),
        IterableDataset controls its own yield order inside workers.
        Epoch state must be set from the main process before iteration
        so forked workers pick up the new seed.
        """
        self._epoch = epoch

    @property
    def flat_indices(self) -> np.ndarray:
        """Flat dataset row indices in iteration order (RG-sorted, no shuffle)."""
        cum_rows = self._dataset.index.cum_rows
        if not self._rg_offsets:
            return np.array([], dtype=np.int64)
        return np.concatenate(
            [self._rg_offsets[rg] + int(cum_rows[rg]) for rg in sorted(self._rg_offsets.keys())]
        )

    @property
    def global_train_rg_ids(self) -> np.ndarray:
        """All train RG IDs across all ranks (set by the loader builder)."""
        return self._global_train_rg_ids

    def set_global_train_rg_ids(self, rg_ids: np.ndarray) -> None:
        """Set the global train RG ID array (used to identify val-only RGs)."""
        self._global_train_rg_ids = np.asarray(rg_ids, dtype=np.int64)

    def set_extra_rg_ids(self, rg_ids: np.ndarray | list[int] | None) -> None:
        """Set additional RG IDs to visit (val-only RGs for the tap callback)."""
        extra = np.asarray(rg_ids if rg_ids is not None else [], dtype=np.int64)
        self._extra_rg_ids = sorted({int(rg) for rg in extra.tolist()})
        self._all_rgs = sorted(set(self._rg_offsets.keys()) | set(self._extra_rg_ids))

    def set_on_rg_loaded(self, callback: Callable[[str, int, Any], None] | None) -> None:
        """Set or clear the per-RG callback invoked during iteration."""
        self._on_rg_loaded = callback

    def clone(self, **overrides) -> StreamingLoader:
        """Return a new StreamingLoader with the same config, overriding specified fields."""
        base = dict(
            dataset=self._dataset,
            indices=(
                np.concatenate(
                    [
                        offsets + int(self._dataset.index.cum_rows[rg])
                        for rg, offsets in self._rg_offsets.items()
                    ]
                )
                if self._rg_offsets
                else np.array([], dtype=np.int64)
            ),
            batch_size=self._batch_size,
            collator=self._collator,
            shuffle_buffer_size=self._shuffle_buffer_size,
            seed=self._seed,
            num_workers=self._num_workers,
            extra_rg_ids=np.asarray(self._extra_rg_ids, dtype=np.int64),
            on_rg_loaded=self._on_rg_loaded,
        )
        cloned = StreamingLoader(**(base | overrides))
        cloned.set_global_train_rg_ids(self._global_train_rg_ids)
        return cloned

    def __len__(self) -> int:
        """Exact number of batches yielded across all workers.

        Replicates the RG ordering that __iter__ will use for the current epoch
        so the per-worker sample counts (and thus batch counts) match exactly.
        """
        if self._shuffle_buffer_size > 0:
            rng = np.random.default_rng(self._seed + self._epoch)
            rg_list = rng.permutation(self._all_rgs).tolist()
        else:
            rg_list = self._all_rgs
        total = 0
        for w in range(self._num_workers):
            worker_rgs = rg_list[w :: self._num_workers]
            worker_samples = sum(len(self._rg_offsets.get(rg, ())) for rg in worker_rgs)
            if worker_samples > 0:
                total += (worker_samples + self._batch_size - 1) // self._batch_size
        return total

    def _stamp_timing(self, batch: dict[str, Any], asm_ms: float) -> dict[str, Any]:
        """Inject _meta_timing into a batch dict and reset I/O counters."""
        batch["_meta_timing"] = np.array(
            [self._io_ms, asm_ms, 0.0, self._rgs_read, self._rows_decomp, self._unique_rgs_seen],
            dtype=np.float32,
        )
        self._io_ms = 0.0
        self._rgs_read = 0
        self._rows_decomp = 0
        return batch

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        shuffling = self._shuffle_buffer_size > 0
        if shuffling:
            rng = np.random.default_rng(self._seed + self._epoch)
            rg_order = rng.permutation(self._all_rgs).tolist()
        else:
            rg_order = self._all_rgs

        my_rgs = rg_order[worker_id::num_workers]
        index = self._dataset.index

        # Per-worker I/O telemetry counters
        self._io_ms = 0.0
        self._rgs_read = 0
        self._rows_decomp = 0
        self._unique_rgs_seen = 0

        callback = self._on_rg_loaded  # holds writer threads; must be joined before worker exits
        auto_close_callback = callback is not None and bool(
            getattr(callback, "auto_close_on_iter_end", False)
        )

        # Group this worker's RGs by file for sequential I/O
        file_rg_groups: list[tuple[str, list[tuple[int, int, np.ndarray]]]] = []
        for rg_id in my_rgs:
            file_path, rg_idx, _ = index.rg_meta[rg_id]
            offsets = self._rg_offsets.get(
                rg_id
            )  # None for val-only RGs; still visited for the callback
            if offsets is None:
                offsets = np.array([], dtype=np.int64)
            if file_rg_groups and file_rg_groups[-1][0] == file_path:
                file_rg_groups[-1][1].append((rg_id, rg_idx, offsets))
            else:
                file_rg_groups.append((file_path, [(rg_id, rg_idx, offsets)]))

        for _, rg_entries in file_rg_groups:
            rg_entries.sort(key=lambda x: x[1])

        accumulator: list[dict[str, Any]] = []
        try:
            for file_path, rg_entries in file_rg_groups:
                pf = index.open_file(file_path)
                rg_indices = [rg_idx for _, rg_idx, _ in rg_entries]

                rg_cursor = 0
                rows_consumed = 0
                rg_row_start = 0

                for arrow_batch in pf.iter_batches(
                    batch_size=1_000_000, row_groups=rg_indices, use_threads=True
                ):
                    batch_rows = arrow_batch.num_rows
                    while rg_cursor < len(rg_entries):
                        rg_id, rg_idx, offsets = rg_entries[rg_cursor]
                        rg_nrows = pf.metadata.row_group(rg_idx).num_rows
                        rg_end = rg_row_start + rg_nrows

                        if rg_row_start >= rows_consumed + batch_rows:
                            break

                        local_start = rg_row_start - rows_consumed
                        t0 = time.perf_counter()

                        # Full RG slice so callback can select val rows at different offsets than train.
                        if callback is not None:
                            callback(file_path, rg_id, arrow_batch.slice(local_start, rg_nrows))

                        if len(offsets) > 0:
                            adjusted_offsets = offsets + local_start
                            n = len(adjusted_offsets)
                            positions = np.arange(n, dtype=np.intp)
                            row_results: list[dict[str, Any] | None] = [None] * n
                            self._dataset.extract_rows_vectorized(
                                arrow_batch, adjusted_offsets, positions, row_results
                            )
                            accumulator.extend(row_results)

                        self._io_ms += (time.perf_counter() - t0) * 1000
                        self._rgs_read += 1
                        self._rows_decomp += rg_nrows
                        self._unique_rgs_seen += 1
                        rg_row_start = rg_end
                        rg_cursor += 1

                        if shuffling:
                            while len(accumulator) >= self._shuffle_buffer_size:
                                yield from self._drain_buffer(accumulator, rng)
                        else:
                            while len(accumulator) >= self._batch_size:
                                chunk = accumulator[: self._batch_size]
                                accumulator = accumulator[self._batch_size :]
                                t_asm = time.perf_counter()
                                batch = self._dataset.assemble_batch(chunk, len(chunk))
                                asm_ms = (time.perf_counter() - t_asm) * 1000
                                yield self._collator(self._stamp_timing(batch, asm_ms))

                    rows_consumed += batch_rows

            # Flush remainder
            if shuffling:
                yield from self._drain_buffer(accumulator, rng, flush=True)
            elif accumulator:
                t_asm = time.perf_counter()
                batch = self._dataset.assemble_batch(accumulator, len(accumulator))
                asm_ms = (time.perf_counter() - t_asm) * 1000
                yield self._collator(self._stamp_timing(batch, asm_ms))
        finally:
            if auto_close_callback:
                close_fn = getattr(callback, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass

    def _drain_buffer(
        self,
        buffer: list[dict[str, Any]],
        rng: np.random.Generator,
        flush: bool = False,
    ):
        """Draw random batches from the buffer.

        When ``flush=False``, drains until buffer is below capacity.
        When ``flush=True``, drains everything.
        """
        threshold = 0 if flush else self._shuffle_buffer_size - self._batch_size
        while len(buffer) > threshold and len(buffer) >= self._batch_size:
            indices = rng.choice(len(buffer), size=self._batch_size, replace=False)
            indices.sort()
            chunk = [buffer[i] for i in indices]
            # Remove selected items in reverse order to preserve indices
            for i in reversed(indices):
                buffer[i] = buffer[-1]
                buffer.pop()
            t_asm = time.perf_counter()
            batch = self._dataset.assemble_batch(chunk, self._batch_size)
            asm_ms = (time.perf_counter() - t_asm) * 1000
            yield self._collator(self._stamp_timing(batch, asm_ms))
        # Flush partial last batch
        if flush and buffer:
            t_asm = time.perf_counter()
            batch = self._dataset.assemble_batch(buffer, len(buffer))
            asm_ms = (time.perf_counter() - t_asm) * 1000
            yield self._collator(self._stamp_timing(batch, asm_ms))
            buffer.clear()
