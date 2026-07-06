"""DataLoader builder for dataset v2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.data.cache import (
    ValCacheTap,
    finalize_tap_cache,
)
from core.data.collators import Collator
from core.data.dataset import AdDataset
from core.data.sampler import AffinityBatchSampler, RGAwareSampler, compute_split
from core.data.streaming import StreamingLoader, _passthrough
from core.utils.device import select_device

LOG = logging.getLogger(__name__)


class ValCacheSwapper:
    """One-shot callable: detaches tap, finalizes cache, returns cached val loader.

    Implements the val_loader_swapper interface expected by the training loop.
    DDP-aware: uses two barriers so non-main ranks wait for the manifest.
    """

    def __init__(
        self,
        *,
        tap: "ValCacheTap",
        data_cfg: dict[str, Any],
        fit_state: dict[str, Any] = None,
        use_ddp: bool = False,
        local_rank: int = 0,
        global_rank: int = 0,
    ) -> None:
        self._tap = tap
        self._data_cfg = data_cfg
        self._fit_state = fit_state
        self._use_ddp = use_ddp
        self._local_rank = local_rank
        self._global_rank = global_rank
        self._settled = False

    def __call__(self, epoch: int):
        """Detach tap, finalize cache, return cached val loader (or None on failure)."""
        import torch.distributed as dist

        if self._settled:
            return None

        self._tap.detach()

        # First barrier: all ranks have flushed shards before main writes manifest.
        if self._use_ddp:
            dist.barrier(device_ids=[self._local_rank])

        is_main = self._global_rank == 0
        cache_dir_str = None
        ready_loader = None
        finalize_ok = True
        if is_main:
            try:
                ready_loader = finalize_val_cache(
                    cache_dir=self._tap.cache_dir,
                    data_cfg=self._data_cfg,
                    fit_state=self._fit_state,
                    manifest=self._tap.manifest,
                    keep_last=self._tap.keep_last,
                    write_manifest=True,
                )
                cache_dir_str = str(self._tap.cache_dir)
            except Exception:
                finalize_ok = False
                LOG.exception("Failed to finalize validation tap cache on rank 0")

        # Second barrier: manifest exists before non-main ranks try to read the cache.
        if self._use_ddp:
            payload = [cache_dir_str, finalize_ok]
            dist.broadcast_object_list(payload, src=0)
            cache_dir_str, finalize_ok = payload
            finalize_ok = bool(finalize_ok)
            dist.barrier(device_ids=[self._local_rank])

        self._settled = True
        if not finalize_ok or not cache_dir_str:
            LOG.warning("Val cache finalization failed; falling back to uncached validation")
            return None

        if is_main and ready_loader is not None:
            return ready_loader
        try:
            return finalize_val_cache(
                cache_dir=cache_dir_str,
                data_cfg=self._data_cfg,
                fit_state=self._fit_state,
            )
        except Exception:
            LOG.warning(
                "Val cache activation failed on rank %d; falling back to uncached validation",
                self._global_rank,
                exc_info=True,
            )
            return None


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move all tensors in a batch dict to `device`."""
    non_blocking = device.type == "cuda"
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device, non_blocking=non_blocking)
        else:
            result[k] = v
    return result


class PrefetchDataLoader(DataLoader):
    """DataLoader that overlaps H2D transfer of the next batch with computation.

    When `prefetch_device` is a CUDA device, uses a background stream to
    transfer batch N+1 while batch N is being processed on the default stream.
    """

    def __init__(self, *, prefetch_device: torch.device = None, **kwargs):
        super().__init__(**kwargs)
        self._prefetch_device = prefetch_device
        self._init_kwargs = kwargs
        self._stream = (
            torch.cuda.Stream(device=prefetch_device)
            if prefetch_device and prefetch_device.type == "cuda"
            else None
        )

    @property
    def seed(self) -> int:
        """Seed of the inner cloneable component."""
        if isinstance(self.dataset, StreamingLoader):
            return self.dataset._seed
        if isinstance(self.batch_sampler, AffinityBatchSampler):
            return self.batch_sampler._seed
        return self.sampler._seed

    def __iter__(self):
        if self._stream is None:
            yield from super().__iter__()
            return

        it = super().__iter__()
        try:
            batch = next(it)
        except StopIteration:
            return

        batch = self._transfer(batch)

        for next_batch in it:
            self._sync(batch)
            next_batch = self._transfer(next_batch)
            yield batch
            batch = next_batch

        self._sync(batch)
        yield batch

    def clone(self, **overrides) -> PrefetchDataLoader:
        """Rebuild this loader with kwargs overrides. No component routing."""
        device = overrides.pop("prefetch_device", self._prefetch_device)
        init_kwargs = self._init_kwargs | overrides
        # PyTorch forbids multiprocessing_context when num_workers == 0.
        if int(init_kwargs.get("num_workers", 0)) <= 0:
            init_kwargs.pop("multiprocessing_context", None)
        return PrefetchDataLoader(**init_kwargs, prefetch_device=device)

    def _transfer(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Kick off async H2D on the side stream."""
        with torch.cuda.stream(self._stream):
            result = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.to(self._prefetch_device, non_blocking=True)
                else:
                    result[k] = v
        return result

    def _sync(self, batch: dict[str, Any]) -> None:
        """Wait for the side-stream transfer to finish before yielding."""
        torch.cuda.current_stream(self._prefetch_device).wait_stream(self._stream)
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                v.record_stream(torch.cuda.current_stream(self._prefetch_device))


def clone_loader(loader: PrefetchDataLoader, **overrides) -> PrefetchDataLoader:
    """Clone a loader, routing overrides to its inner cloneable component.

    `num_workers` is forwarded to both layers (component RG partitioning
    and outer DataLoader fork count) since it spans both.
    """
    # num_workers lives in both layers
    loader_overrides = {}
    if "num_workers" in overrides:
        loader_overrides["num_workers"] = overrides["num_workers"]

    if isinstance(loader.dataset, StreamingLoader):
        cloned = loader.dataset.clone(**overrides)
        return loader.clone(dataset=cloned, **loader_overrides)
    if isinstance(loader.batch_sampler, AffinityBatchSampler):
        cloned = loader.batch_sampler.clone(**overrides)
        return loader.clone(batch_sampler=cloned, **loader_overrides)
    if isinstance(loader.sampler, RGAwareSampler):
        cloned = loader.sampler.clone(**overrides)
        return loader.clone(sampler=cloned, **loader_overrides)
    raise TypeError(f"No cloneable component in loader (dataset={type(loader.dataset).__name__})")


def build_dataloaders(
    dataset_path: str | Path,
    schema_path: str | Path,
    batch_size: int,
    val_batch_size: int,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    split_mode: str = "positional",
    collator_type: str = "padded",
    shuffle_train: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    val_num_workers: int = None,
    is_training: bool = True,
    clip_vocab: bool = True,
    split_f129: bool = True,
    blocks: dict[str, dict] = None,
    blocks_order: list[str] = None,
    pin_memory: bool = False,
    prefetch_device: str = None,
    fit_state: dict[str, Any] = None,
    streaming_val: bool = False,
    streaming_train: bool = False,
    shuffle_buffer_size: int = 50000,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader | None, DataLoader, AdDataset]:
    """Build train and validation DataLoaders.

    Parameters
    ----------
    dataset_path
        Directory with ``*.parquet`` files.
    schema_path
        Path to ``schema.json``.
    batch_size
        Samples per batch (training).
    val_batch_size
        Samples per batch (validation).
    valid_ratio
        Fraction of data for validation.
    train_ratio
        Fraction of training split to use (fixed subset).
    split_mode
        ``"positional"`` or ``"time"``.
    collator_type
        ``"padded"`` or ``"flat"``.
    shuffle_train
        Whether to shuffle training data.
    seed
        Random seed.
    num_workers
        DataLoader workers for training.
    val_num_workers
        DataLoader workers for validation. Defaults to ``num_workers``.
    is_training
        Whether labels should be derived or zeros.
    clip_vocab
        Clip out-of-vocabulary int features to 0.
    split_f129
        Split item_cont_f129 into ``item_cont_f129_emb`` (128-dim) and
        ``item_cont_f129_count`` (1-dim). Set False to keep as a single feature.
    blocks
        Blocks to activate, keyed by type_key. Values are per-block config
        dicts. E.g. ``{"rssc": {"clip_value": 3.0}, "time_bucket": {}}``.
    blocks_order
        Explicit execution order for blocks. When provided, all active blocks
        must appear in this list; a ValueError is raised otherwise.
    pin_memory
        Pin DataLoader memory for faster GPU transfer.
    prefetch_device
        Device for async prefetch (e.g. ``"cuda:0"``). None disables prefetch.
    fit_state
        Pre-fitted block state from a checkpoint. When provided, skips the
        parquet scan and restores block statistics directly.
    streaming_val
        Use streaming IterableDataset for validation (reads RGs sequentially,
        no index resolution overhead).
    streaming_train
        Use streaming IterableDataset for training (full-RG extraction with
        shuffle buffer for randomness).
    shuffle_buffer_size
        Number of rows to buffer before drawing random batches. Only used
        when ``streaming_train=True``.
    rank
        Local rank for DDP partitioning. 0 when not distributed.
    world_size
        Total number of DDP ranks. 1 when not distributed.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, dataset)``
    """
    if val_num_workers is None:
        val_num_workers = num_workers

    # num_workers is total for the machine; round to nearest multiple of
    # world_size then divide, so per-rank count stays close to the intent.
    if world_size > 1:
        num_workers = max(1, round(num_workers / world_size))
        val_num_workers = max(1, round(val_num_workers / world_size))

    dataset = AdDataset(
        dataset_path=dataset_path,
        schema_path=schema_path,
        blocks=blocks,
        blocks_order=blocks_order,
        clip_vocab=clip_vocab,
        is_training=is_training,
        split_f129=split_f129,
    )

    train_idx, val_idx = compute_split(dataset.index, split_mode, valid_ratio, train_ratio, seed)

    if fit_state:
        dataset.load_fit_state(fit_state)
        LOG.info("Restored pre-fitted block state from checkpoint")
    else:
        dataset.fit_blocks(train_idx)
        fitted_blocks = [type(b).__name__ for b in dataset._blocks if b.fit_columns()]
        if fitted_blocks:
            LOG.info("Fitted blocks: %s on %d train samples", fitted_blocks, len(train_idx))

    collator = Collator(dataset.feature_schema, format=collator_type)

    # Global RG set across all ranks; used to identify val-only RGs no rank naturally visits.
    train_rg_ids_global = (
        np.unique(np.searchsorted(dataset.index.cum_rows[1:], train_idx, side="right")).astype(
            np.int64
        )
        if len(train_idx) > 0
        else np.array([], dtype=np.int64)
    )

    # DDP: partition train RGs by rank after fitting (all ranks use same fit state)
    # Block partition gives each rank contiguous RGs for file locality
    if world_size > 1:
        rg_groups = dataset.index.group_by_rg(train_idx)
        n_rgs = len(rg_groups)
        per_rank = n_rgs // world_size
        remainder = n_rgs % world_size
        if rank < remainder:
            start = rank * (per_rank + 1)
            end = start + per_rank + 1
        else:
            start = remainder * (per_rank + 1) + (rank - remainder) * per_rank
            end = start + per_rank
        my_rgs = list(range(start, end))
        train_idx = (
            np.concatenate([rg_groups[i] for i in my_rgs])
            if my_rgs
            else np.array([], dtype=np.int64)
        )
        LOG.info(
            "Rank %d: %d/%d train RGs, %d train samples",
            rank,
            len(my_rgs),
            n_rgs,
            len(train_idx),
        )

    LOG.info(
        "Split (%s): %d train, %d val | batch_size=%d, collator=%s, prefetch=%s",
        split_mode,
        len(train_idx),
        len(val_idx),
        batch_size,
        collator_type,
        prefetch_device or "off",
    )

    device = select_device(prefetch_device)
    # CUDA is initialized before DataLoader iteration (model init, reinit, etc.).
    # Forked workers inherit dangling CUDA state and crash on exit (SIGABRT from
    # cudaFree in a process with no valid context). forkserver avoids this by
    # spawning workers from a pristine server forked before any CUDA usage.
    mp_context = "forkserver" if num_workers > 0 else None
    common_kwargs = dict(
        collate_fn=collator,
        drop_last=False,
        pin_memory=pin_memory,
        prefetch_device=device,
        multiprocessing_context=mp_context,
    )

    train_loader: DataLoader | None
    if len(train_idx) == 0:
        LOG.info("Train split is empty; skipping train DataLoader build")
        train_loader = None
    elif streaming_train:
        train_streaming_ds = StreamingLoader(
            dataset=dataset,
            indices=train_idx,
            batch_size=batch_size,
            collator=collator,
            shuffle_buffer_size=shuffle_buffer_size if shuffle_train else 0,
            seed=seed,
            num_workers=num_workers,
        )
        train_streaming_ds.set_global_train_rg_ids(train_rg_ids_global)
        train_loader = PrefetchDataLoader(
            dataset=train_streaming_ds,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=_passthrough,
            drop_last=False,
            pin_memory=pin_memory,
            prefetch_device=device,
            multiprocessing_context=mp_context,
        )
    else:
        train_batch_sampler = AffinityBatchSampler(
            train_idx,
            dataset.index,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle_train,
            seed=seed,
        )
        train_loader = PrefetchDataLoader(
            dataset=dataset,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            **common_kwargs,
        )

    if streaming_val:
        streaming_ds = StreamingLoader(
            dataset=dataset,
            indices=val_idx,
            batch_size=val_batch_size,
            collator=collator,
            num_workers=val_num_workers,
        )
        # batch_size=None disables automatic batching — each __iter__ item
        # is already a collated tensor dict, so pass through unchanged.
        val_loader = PrefetchDataLoader(
            dataset=streaming_ds,
            batch_size=None,
            num_workers=val_num_workers,
            collate_fn=_passthrough,
            drop_last=False,
            pin_memory=pin_memory,
            prefetch_device=device,
            multiprocessing_context=mp_context,
        )
    else:
        val_batch_sampler = AffinityBatchSampler(
            val_idx,
            dataset.index,
            batch_size=val_batch_size,
            num_workers=val_num_workers,
            shuffle=False,
            seed=seed,
        )
        val_loader = PrefetchDataLoader(
            dataset=dataset,
            batch_sampler=val_batch_sampler,
            num_workers=val_num_workers,
            **common_kwargs,
        )

    return train_loader, val_loader, dataset


def build_pretrain_loader(train_loader: PrefetchDataLoader) -> PrefetchDataLoader:
    """Clone the train loader with an independent RNG (seed + 1)."""
    return clone_loader(train_loader, seed=train_loader.seed + 1)


def finalize_val_cache(
    *,
    cache_dir: str | Path,
    data_cfg: dict[str, Any],
    fit_state: dict[str, Any] = None,
    manifest: dict[str, Any] = None,
    keep_last: int = 3,
    write_manifest: bool = False,
) -> DataLoader:
    """Build a streaming val loader over an existing tap cache directory.

    With `write_manifest=True`, also writes the manifest/_SUCCESS marker
    (called once by rank 0 after epoch 1). Without it, just builds the loader
    (called by non-main ranks or on cache hit).
    """
    resolved_cache_dir = Path(cache_dir)
    if write_manifest:
        if manifest is None:
            raise ValueError("manifest is required when write_manifest=True")
        finalize_tap_cache(
            cache_root=resolved_cache_dir.parent,
            cache_dir=resolved_cache_dir,
            manifest=manifest,
            keep_last=keep_last,
        )

    cache_data_cfg = dict(data_cfg)
    cache_data_cfg.pop("val_cache", None)
    cache_data_cfg.update(
        dataset_path=resolved_cache_dir,
        valid_ratio=1.0,
        train_ratio=0.0,
        split_mode="positional",
        shuffle_train=False,
        streaming_train=False,
        streaming_val=True,
        fit_state=fit_state,
    )
    _, cached_val_loader, _ = build_dataloaders(
        **cache_data_cfg,
        rank=0,
        world_size=1,
    )
    LOG.info("Activated compact validation cache: %s", resolved_cache_dir)
    return cached_val_loader
