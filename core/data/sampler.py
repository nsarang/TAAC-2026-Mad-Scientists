"""Split computation and RG-aware sampling."""

from __future__ import annotations

import logging

import numpy as np
from torch.utils.data import BatchSampler, Sampler

from core.data.dataset import RowIndex

LOG = logging.getLogger(__name__)


def compute_split(
    index: RowIndex,
    split_mode: str,
    valid_ratio: float,
    train_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition row indices into (train, val). Guaranteed complementary.

    Parameters
    ----------
    index
        RowIndex providing timestamps and row-group structure.
    split_mode
        ``"positional"`` or ``"time"``.
    valid_ratio
        Fraction of data reserved for validation.
    train_ratio
        Fraction of training split to actually use (fixed subset).
    seed
        Random seed for reproducible subsampling.

    Returns
    -------
    tuple
        ``(train_indices, val_indices)`` as numpy arrays.
    """
    total = len(index.timestamps)

    if split_mode == "positional":
        n_valid = int(total * valid_ratio)
        n_train = total - n_valid
        train_pool = np.arange(n_train, dtype=np.int64)
        val_pool = np.arange(n_train, total, dtype=np.int64)

    elif split_mode == "time":
        cutoff = int(np.quantile(index.timestamps, 1 - valid_ratio))
        train_pool = np.where(index.timestamps < cutoff)[0].astype(np.int64)
        val_pool = np.where(index.timestamps >= cutoff)[0].astype(np.int64)

    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r}")

    # Fixed subsample for train
    if train_ratio < 1.0 and len(train_pool) > 0:
        if train_ratio <= 0.0:
            train_pool = np.array([], dtype=np.int64)
        else:
            rng = np.random.default_rng(seed)
            n_keep = max(1, int(len(train_pool) * train_ratio))
            chosen = rng.choice(len(train_pool), size=n_keep, replace=False)
            chosen.sort()
            train_pool = train_pool[chosen]

    return train_pool, val_pool


class RGAwareSampler(Sampler):
    """Iterates pre-computed indices with optional RG-clustered shuffle.

    Parameters
    ----------
    indices
        Flat row indices to iterate over.
    index
        RowIndex providing cumulative row counts for RG grouping.
    shuffle
        Whether to shuffle indices each epoch (RG-clustered).
    seed
        Random seed for reproducibility.
    """

    def __init__(
        self,
        indices: np.ndarray,
        index: RowIndex,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self._indices = indices
        self._index = index
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    @property
    def indices(self) -> np.ndarray:
        """Flat row indices belonging to this sampler."""
        return self._indices

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self):
        """Yield row indices with RG-clustered shuffle for IO locality.

        Permutes row-group order, then shuffles within each RG, so
        consecutive indices tend to hit the same Parquet page cache.
        """
        if self._shuffle:
            rng = np.random.default_rng(self._seed + self._epoch)
            # TODO (nsarang): _indices are fixed across epochs — train_ratio subsample
            # is drawn once at split time. Departure from v1 where resampling per
            # epoch was possible. Consider epoch-varying subsampling if needed.
            rg_groups = self._index.group_by_rg(self._indices)
            rg_order = rng.permutation(len(rg_groups))
            shuffled = []
            for rg_i in rg_order:
                group = rg_groups[rg_i]
                shuffled.extend(rng.permutation(group).tolist())
            self._epoch += 1
            return iter(shuffled)
        else:
            self._epoch += 1
            return iter(self._indices.tolist())

    def clone(self, **overrides) -> RGAwareSampler:
        """Return a new RGAwareSampler with the same config, overriding specified fields."""
        return RGAwareSampler(
            **dict(
                indices=self._indices,
                index=self._index,
                shuffle=self._shuffle,
                seed=self._seed,
            )
            | overrides
        )


class AffinityBatchSampler(BatchSampler):
    """Batch sampler that aligns workers with RG partitions for sequential I/O.

    Produces interleaved batches so PyTorch's round-robin dispatch gives each
    worker a contiguous stripe of row groups. Within each stripe, RG visit
    order and row order are shuffled (if enabled).

    Parameters
    ----------
    indices
        Flat row indices to iterate over.
    index
        RowIndex providing cumulative row counts for RG grouping.
    batch_size
        Number of samples per batch.
    num_workers
        Number of DataLoader workers. Must be >= 1.
    shuffle
        Whether to shuffle RG order and rows within RGs each epoch.
    drop_last
        Whether to drop the last incomplete batch per worker.
    seed
        Random seed for reproducibility.
    """

    def __init__(
        self,
        indices: np.ndarray,
        index: RowIndex,
        batch_size: int,
        num_workers: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        self._indices = indices
        self._index = index
        self._batch_size = batch_size
        self._num_workers = max(1, num_workers)
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._seed = seed
        self._epoch = 0

        # Pre-compute RG groups once (indices grouped by which RG they belong to)
        self._rg_groups = index.group_by_rg(indices)

    @property
    def indices(self) -> np.ndarray:
        """Flat row indices belonging to this sampler."""
        return self._indices

    def __len__(self) -> int:
        total = len(self._indices)
        if self._drop_last:
            return (total // self._batch_size) // self._num_workers * self._num_workers
        return (total + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        W = self._num_workers

        # Shuffle RG order, then partition RGs across workers round-robin
        n_rgs = len(self._rg_groups)
        rg_order = rng.permutation(n_rgs) if self._shuffle else np.arange(n_rgs)

        # Each worker gets every W-th RG in the shuffled order
        worker_rgs: list[list[int]] = [[] for _ in range(W)]
        for i, rg_i in enumerate(rg_order):
            worker_rgs[i % W].append(int(rg_i))

        # Build per-worker batch lists
        worker_batches: list[list[np.ndarray]] = [[] for _ in range(W)]
        for w in range(W):
            rows = []
            for rg_i in worker_rgs[w]:
                group = self._rg_groups[rg_i]
                if self._shuffle:
                    group = rng.permutation(group)
                rows.append(group)
            if not rows:
                continue
            all_rows = np.concatenate(rows)
            # Chunk into batches
            for start in range(0, len(all_rows), self._batch_size):
                end = start + self._batch_size
                chunk = all_rows[start:end]
                if self._drop_last and len(chunk) < self._batch_size:
                    continue
                worker_batches[w].append(chunk)

        # Interleave: [w0_b0, w1_b0, ..., wN_b0, w0_b1, ...] so round-robin
        # dispatch gives worker k batch k, k+W, k+2W, ...
        max_batches = max((len(wb) for wb in worker_batches), default=0)
        for b_idx in range(max_batches):
            for w in range(W):
                if b_idx < len(worker_batches[w]):
                    yield worker_batches[w][b_idx].tolist()

    def clone(self, **overrides) -> AffinityBatchSampler:
        """Return a new AffinityBatchSampler with the same config."""
        return AffinityBatchSampler(
            **dict(
                indices=self._indices,
                index=self._index,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=self._shuffle,
                drop_last=self._drop_last,
                seed=self._seed,
            )
            | overrides
        )
