"""Validation cache utilities for compacting val rows into Parquet shards."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, get_worker_info

from core.data.dataset import AdDataset, RowIndex
from core.data.streaming import StreamingLoader

LOG = logging.getLogger(__name__)

_NON_IDENTITY_MANIFEST_FIELDS = frozenset({"created_at_epoch_s", "parts"})
_TAP_MANIFEST_VERSION = 2
_STOP_SENTINEL = object()


def extract_validation_indices(val_loader: Any) -> np.ndarray:
    """Extract flat validation indices from either streaming or map-style loaders."""
    dataset = getattr(val_loader, "dataset", None)
    if isinstance(dataset, StreamingLoader):
        indices = dataset.flat_indices
    else:
        batch_sampler = getattr(val_loader, "batch_sampler", None)
        if batch_sampler is None:
            raise TypeError("Validation loader has no batch_sampler and is not a StreamingLoader")
        if hasattr(batch_sampler, "indices"):
            indices = batch_sampler.indices
        elif hasattr(batch_sampler, "_indices"):
            indices = batch_sampler._indices
        else:
            raise TypeError("Validation batch_sampler does not expose indices")

    out = np.asarray(indices, dtype=np.int64)
    if out.ndim != 1:
        out = out.reshape(-1)
    return np.ascontiguousarray(out)


@dataclass(frozen=True)
class ValCacheTapIdentity:
    """Stable identity used for tap-cache manifest hashing."""

    dataset_path: Path
    schema_path: Path
    val_indices: np.ndarray
    blocks: dict[str, Any] = None
    fit_state: dict[str, Any] = None
    world_size: int = 1


def build_tap_manifest(identity: ValCacheTapIdentity) -> dict[str, Any]:
    """Create manifest for tap-built validation cache."""
    return {
        "version": _TAP_MANIFEST_VERSION,
        "dataset_path": str(identity.dataset_path.resolve()),
        "schema_path": str(identity.schema_path.resolve()),
        "dataset_signature": _dataset_signature(identity.dataset_path),
        "val_indices": {
            "count": len(identity.val_indices),
            "sha256": _hash_indices(identity.val_indices),
        },
        "blocks_sha256": _hash_jsonable(identity.blocks or {}),
        "fit_state_sha256": _hash_fit_state(identity.fit_state),
        "world_size": int(identity.world_size),
    }


def locate_tap_cache(cache_root: Path, manifest: dict[str, Any]) -> Path | None:
    """Return matching cache dir if present and complete."""
    key = cache_key(manifest)
    cache_dir = cache_root / key
    if _cache_matches_manifest(cache_dir, manifest):
        return cache_dir
    return None


def prepare_tap_cache_dir(cache_root: Path, manifest: dict[str, Any], is_main: bool) -> Path:
    """Prepare destination directory for a new tap build."""
    key = cache_key(manifest)
    cache_dir = cache_root / key
    if is_main:
        cache_root.mkdir(parents=True, exist_ok=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def build_val_rg_offsets(val_indices: np.ndarray, row_index: RowIndex) -> dict[int, np.ndarray]:
    """Map global RG id -> offsets within RG for all validation rows."""
    if len(val_indices) == 0:
        return {}

    val_indices = np.asarray(val_indices, dtype=np.int64)
    rg_ids = np.searchsorted(row_index.cum_rows[1:], val_indices, side="right")
    order = np.argsort(rg_ids, kind="mergesort")
    sorted_rg_ids = rg_ids[order]
    sorted_indices = val_indices[order]
    split_points = np.where(np.diff(sorted_rg_ids) != 0)[0] + 1

    offsets_by_rg: dict[int, np.ndarray] = {}
    for chunk in np.split(np.arange(len(sorted_indices)), split_points):
        if len(chunk) == 0:
            continue
        rg_id = int(sorted_rg_ids[chunk[0]])
        within_rg = (sorted_indices[chunk] - int(row_index.cum_rows[rg_id])).astype(np.int64)
        offsets_by_rg[rg_id] = np.ascontiguousarray(within_rg)
    return offsets_by_rg


def assign_val_only_rgs(
    val_rg_offsets: dict[int, np.ndarray],
    global_train_rg_ids: np.ndarray,
    rank: int,
    world_size: int,
) -> np.ndarray:
    """Partition RGs that contain only validation rows across ranks."""
    if not val_rg_offsets:
        return np.array([], dtype=np.int64)

    train_set = {int(rg) for rg in np.asarray(global_train_rg_ids, dtype=np.int64).tolist()}
    val_only = [rg for rg in sorted(val_rg_offsets.keys()) if rg not in train_set]
    if not val_only:
        return np.array([], dtype=np.int64)

    n = len(val_only)
    per_rank = n // world_size
    remainder = n % world_size
    if rank < remainder:
        start = rank * (per_rank + 1)
        end = start + per_rank + 1
    else:
        start = remainder * (per_rank + 1) + (rank - remainder) * per_rank
        end = start + per_rank
    return np.asarray(val_only[start:end], dtype=np.int64)


class ValCacheTapWriter:
    """Asynchronous shard writer used by the train worker RG tap."""

    def __init__(
        self,
        *,
        out_dir: Path,
        shard_rows: int,
        row_group_size: int,
        compression: str,
        max_pending_flushes: int = 32,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._shard_rows = max(1, int(shard_rows))
        self._row_group_size = max(1, int(row_group_size))
        self._compression = str(compression)
        self._max_pending_flushes = max(1, int(max_pending_flushes))
        self._pending: list[pa.Table] = []
        self._pending_rows = 0
        self._part_idx = 0
        self._closed = False
        self._lock = Lock()
        self._queue: Queue = Queue(maxsize=self._max_pending_flushes)
        self._thread = Thread(
            target=self._writer_loop,
            name=f"val-cache-writer-{os.getpid()}",
            daemon=True,
        )
        self._thread.start()

    @property
    def part_count(self) -> int:
        """Number of parquet shard files written so far."""
        return self._part_idx

    def add_table(self, chunk: pa.Table | pa.RecordBatch) -> None:
        """Enqueue an arrow chunk for async writing; flushes when shard is full."""
        if chunk.num_rows == 0:
            return
        if isinstance(chunk, pa.RecordBatch):
            table = pa.Table.from_batches([chunk])
        elif isinstance(chunk, pa.Table):
            table = chunk
        else:
            raise TypeError(f"Unsupported chunk type for cache writer: {type(chunk)!r}")
        with self._lock:
            if self._closed:
                return
            self._pending.append(table)
            self._pending_rows += table.num_rows
            if self._pending_rows >= self._shard_rows:
                self._flush_pending_locked()

    def close(self) -> None:
        """Flush remaining rows, stop the writer thread, and wait for it to exit."""
        with self._lock:
            if self._closed:
                return
            self._flush_pending_locked()
            self._closed = True
        self._queue.put(_STOP_SENTINEL)
        self._thread.join()

    def _flush_pending_locked(self) -> None:
        if not self._pending:
            return
        chunk = self._pending
        self._pending = []
        self._pending_rows = 0
        # Bounded queue: apply backpressure before worker OOM on slow storage.
        self._queue.put(chunk)

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP_SENTINEL:
                return
            tables = item
            if not tables:
                continue
            table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
            out_path = self._out_dir / f"part-{self._part_idx:05d}.parquet"
            pq.write_table(
                table,
                out_path,
                compression=self._compression,
                row_group_size=self._row_group_size,
            )
            self._part_idx += 1


class ValCacheTapCallback:
    """Captures validation rows as a side effect of the epoch-1 training stream.

    Attached to StreamingLoader as an on_rg_loaded callback; receives the full
    arrow chunk for each row group as training iterates, selects the validation
    rows within it, and writes them to parquet shards asynchronously. This avoids
    a second independent read pass over the dataset for validation-cache building.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        rank: int,
        val_rg_offsets: dict[int, np.ndarray],
        shard_rows: int,
        row_group_size: int,
        compression: str,
        max_pending_shards: int = 32,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._rank = int(rank)
        self._val_rg_offsets = {
            int(rg): np.asarray(offsets, dtype=np.int64) for rg, offsets in val_rg_offsets.items()
        }
        self._shard_rows = max(1, int(shard_rows))
        self._row_group_size = max(1, int(row_group_size))
        self._compression = str(compression)
        self._max_pending_shards = max(1, int(max_pending_shards))
        self._writers: dict[int, ValCacheTapWriter] = {}
        self._writers_lock = Lock()
        self._registered_atexit = False
        self._disabled = False
        self.auto_close_on_iter_end = True

    def __call__(self, file_path: str, rg_id: int, rg_chunk: pa.Table | pa.RecordBatch) -> None:
        """Select val rows from `rg_chunk`. Uses global RG id (not file-local index)."""
        if self._disabled:
            return
        offsets = self._val_rg_offsets.get(int(rg_id))
        if offsets is None or len(offsets) == 0:
            return
        if len(offsets) == rg_chunk.num_rows:
            selected = rg_chunk
        else:
            selected = rg_chunk.take(pa.array(offsets, type=pa.int64()))
        if selected.num_rows == 0:
            return
        writer = self._get_or_create_writer()
        writer.add_table(selected)

    def close(self) -> None:
        """Flush and close all per-worker shard writers."""
        with self._writers_lock:
            writers = list(self._writers.values())
            self._writers.clear()
        for writer in writers:
            writer.close()

    def disable(self) -> None:
        """Permanently stop capturing rows and close all writers."""
        self._disabled = True
        self.close()

    def __getstate__(self):
        # Writers hold threads; drop them so the callback survives fork into workers.
        state = dict(self.__dict__)
        state["_writers"] = {}
        state["_writers_lock"] = None
        state["_registered_atexit"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._writers = {}
        self._writers_lock = Lock()

    def _get_or_create_writer(self) -> ValCacheTapWriter:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        with self._writers_lock:
            writer = self._writers.get(worker_id)
            if writer is not None:
                return writer
            out_dir = self._cache_dir / f"rank_{self._rank}" / f"worker_{worker_id}"
            writer = ValCacheTapWriter(
                out_dir=out_dir,
                shard_rows=self._shard_rows,
                row_group_size=self._row_group_size,
                compression=self._compression,
                max_pending_flushes=self._max_pending_shards,
            )
            self._writers[worker_id] = writer
            if not self._registered_atexit:
                atexit.register(self.close)
                self._registered_atexit = True
            return writer


def finalize_tap_cache(
    *,
    cache_root: Path,
    cache_dir: Path,
    manifest: dict[str, Any],
    keep_last: int = 3,
) -> Path:
    """Write manifest/_SUCCESS and prune stale completed caches."""
    part_paths = sorted(cache_dir.glob("rank_*/worker_*/part-*.parquet"))
    if not part_paths:
        raise RuntimeError(f"No parquet parts written under tap cache dir: {cache_dir}")

    payload = dict(manifest)
    payload["parts"] = {"count": len(part_paths)}
    _write_manifest(cache_dir, payload)
    (cache_dir / "_SUCCESS").write_text("ok\n")
    _cleanup_old_caches(cache_root, keep_last=max(1, int(keep_last)), keep={cache_dir})
    LOG.info("Validation tap cache finalized: %s (%d parts)", cache_dir, len(part_paths))
    return cache_dir


def cache_key(manifest: dict[str, Any]) -> str:
    """Compute a compact cache key from stable manifest fields."""
    payload = _identity_manifest_payload(manifest)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _identity_manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest.items() if k not in _NON_IDENTITY_MANIFEST_FIELDS}


def _dataset_signature(dataset_path: Path) -> dict[str, Any]:
    files = sorted(dataset_path.glob("*.parquet"))
    hasher = hashlib.sha256()
    for file_path in files:
        st = file_path.stat()
        rel = file_path.name
        hasher.update(rel.encode("utf-8"))
        hasher.update(str(st.st_size).encode("utf-8"))
        hasher.update(str(st.st_mtime_ns).encode("utf-8"))
    return {"num_files": len(files), "sha256": hasher.hexdigest()}


def _hash_indices(indices: np.ndarray) -> str:
    arr = np.ascontiguousarray(indices.astype(np.int64, copy=False))
    hasher = hashlib.sha256()
    hasher.update(str(len(arr)).encode("utf-8"))
    hasher.update(arr.view(np.uint8).tobytes())
    return hasher.hexdigest()


def _hash_jsonable(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_fit_state(fit_state: dict[str, Any] = None) -> str:
    if fit_state is None:
        return ""
    blob = pickle.dumps(fit_state, protocol=5)
    return hashlib.sha256(blob).hexdigest()


def _write_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    payload = dict(manifest)
    payload["created_at_epoch_s"] = int(time.time())
    (cache_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": ")),
        encoding="utf-8",
    )


def _cache_matches_manifest(cache_dir: Path, expected_manifest: dict[str, Any]) -> bool:
    success = cache_dir / "_SUCCESS"
    manifest_path = cache_dir / "manifest.json"
    if not success.exists() or not manifest_path.exists():
        return False
    try:
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return _identity_manifest_payload(cached_manifest) == _identity_manifest_payload(
        expected_manifest
    )


def _cleanup_old_caches(cache_root: Path, keep_last: int, keep: set[Path]) -> None:
    candidates = []
    for child in cache_root.iterdir():
        if not child.is_dir() or child in keep:
            continue
        if not (child / "_SUCCESS").exists():
            continue
        candidates.append(child)

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in candidates[keep_last:]:
        shutil.rmtree(stale, ignore_errors=True)


class ValCacheTap:
    """Captures validation rows by tapping epoch-1's training stream.

    Lifecycle: construct -> check `cache_hit` -> if miss, `attach()` -> train
    epoch 1 -> `detach()`. The caller (or a ValCacheSwapper) then finalizes
    the cache and builds a loader from the written shards.

    Parameters
    ----------
    data_cfg
        Data config dict (dataset_path, schema_path, blocks, etc.).
    train_loader
        The streaming train loader whose dataset will be tapped.
    dataset
        The shared AdDataset instance.
    val_loader
        Current validation loader (used to extract val indices for identity).
    fit_state
        Pre-fitted block state from checkpoint.
    is_main
        Whether this is rank 0.
    enabled
        Whether val cache is enabled.
    dir
        Root directory for cache storage.
    shard_rows
        Rows per parquet shard written by the tap.
    row_group_size
        Parquet row group size within shards.
    compression
        Parquet compression codec.
    max_pending_shards
        Backpressure limit for the async writer queue.
    keep_last
        Number of old completed caches to retain.
    rank
        DDP rank (0 when not distributed).
    world_size
        DDP world size (1 when not distributed).
    """

    def __init__(
        self,
        *,
        data_cfg: dict[str, Any],
        train_loader: DataLoader,
        dataset: AdDataset,
        val_loader: DataLoader,
        fit_state: dict[str, Any] = None,
        is_main: bool,
        enabled: bool,
        dir: str | Path = None,
        shard_rows: int = 4096,
        row_group_size: int = 4096,
        compression: str = "zstd",
        max_pending_shards: int = 32,
        keep_last: int = 3,
        rank: int = 0,
        world_size: int = 1,
        **_: Any,
    ) -> None:
        self._enabled = bool(enabled)
        self._cache_hit = False
        self._cache_dir: Path | None = None
        self._manifest: dict[str, Any] | None = None
        self._keep_last = max(1, int(keep_last))
        self._extra_rg_ids = np.array([], dtype=np.int64)
        self._tap_callback: ValCacheTapCallback | None = None
        self._train_ds: StreamingLoader | None = None

        if not self._enabled:
            return
        if train_loader is None or not isinstance(train_loader.dataset, StreamingLoader):
            raise RuntimeError("Validation cache tap requires data.streaming_train=true")

        val_indices = extract_validation_indices(val_loader)
        if len(val_indices) == 0:
            LOG.info("Validation cache requested but val split is empty; skipping cache build")
            self._enabled = False
            return

        if not dir:
            raise ValueError("data.val_cache.dir must be set when val_cache is enabled")
        cache_root = Path(dir)
        manifest = build_tap_manifest(
            ValCacheTapIdentity(
                dataset_path=Path(data_cfg["dataset_path"]),
                schema_path=Path(data_cfg["schema_path"]),
                val_indices=val_indices,
                blocks=data_cfg["blocks"] if "blocks" in data_cfg else None,
                fit_state=fit_state,
                world_size=world_size,
            )
        )
        self._manifest = manifest
        self._cache_dir = cache_root / cache_key(manifest)

        if cache_hit_dir := locate_tap_cache(cache_root, manifest):
            LOG.info("Validation tap cache hit: %s", cache_hit_dir)
            self._cache_dir = cache_hit_dir
            self._cache_hit = True
            return

        prepare_tap_cache_dir(cache_root, manifest, is_main=is_main)

        # RGs overlapping training are tapped naturally; val-only RGs must be assigned as extras.
        train_ds = train_loader.dataset
        global_train_rg_ids = train_ds.global_train_rg_ids
        val_rg_offsets = build_val_rg_offsets(val_indices, dataset.index)
        extra_rg_ids = assign_val_only_rgs(val_rg_offsets, global_train_rg_ids, rank, world_size)
        self._tap_callback = ValCacheTapCallback(
            cache_dir=self._cache_dir,
            rank=rank,
            val_rg_offsets=val_rg_offsets,
            shard_rows=shard_rows,
            row_group_size=row_group_size,
            compression=compression,
            max_pending_shards=max_pending_shards,
        )
        self._extra_rg_ids = np.asarray(extra_rg_ids, dtype=np.int64)

    @property
    def enabled(self) -> bool:
        """Whether the tap is active (False if disabled or val split was empty)."""
        return self._enabled

    @property
    def cache_hit(self) -> bool:
        """Whether a valid completed cache was found at construction time."""
        return self._cache_hit

    @property
    def cache_dir(self) -> Path | None:
        """Directory where cache shards are written (or were found on hit)."""
        return self._cache_dir

    @property
    def manifest(self) -> dict[str, Any] | None:
        """Identity manifest for this cache build."""
        return self._manifest

    @property
    def keep_last(self) -> int:
        """Number of old completed caches to retain during cleanup."""
        return self._keep_last

    def attach(self, train_ds: StreamingLoader) -> None:
        """Wire extra_rg_ids and tap callback onto the training dataset."""
        self._train_ds = train_ds
        train_ds.set_extra_rg_ids(self._extra_rg_ids)
        train_ds.set_on_rg_loaded(self._tap_callback)

    def detach(self) -> None:
        """Remove tap callback and flush writers so shards are complete on disk."""
        if self._train_ds is not None:
            self._train_ds.set_on_rg_loaded(None)
            self._train_ds.set_extra_rg_ids(None)
        if self._tap_callback is not None:
            self._tap_callback.disable()
