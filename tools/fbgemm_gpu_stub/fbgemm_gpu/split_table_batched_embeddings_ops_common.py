"""Stub enums from fbgemm_gpu.split_table_batched_embeddings_ops_common."""

import enum
from typing import NamedTuple


class EmbeddingLocation(enum.IntEnum):
    """Physical placement of embedding tables."""

    DEVICE = 0
    MANAGED = 1
    MANAGED_CACHING = 2
    HOST = 3
    MTIA = 4


class PoolingMode(enum.IntEnum):
    """Pooling reduction mode for embedding lookups."""

    SUM = 0
    MEAN = 1
    NONE = 2

    def do_pooling(self) -> bool:
        """Return True if this mode applies a reduction."""
        return self is not PoolingMode.NONE


class ComputeDevice(enum.IntEnum):
    """Compute device for embedding operations."""

    CPU = 0
    CUDA = 1
    MTIA = 2


class BoundsCheckMode(enum.IntEnum):
    """Behavior when embedding indices are out of bounds."""

    FATAL = 0
    WARNING = 1
    IGNORE = 2
    NONE = 3


class CacheAlgorithm(enum.Enum):
    """Cache eviction algorithm for managed-caching embeddings."""

    LRU = "lru"
    LFU = "lfu"


class BackendType(enum.Enum):
    """Storage backend for embedding tables."""

    SSD = "ssd"
    DRAM = "dram"


class RecordCacheMetrics(NamedTuple):
    """Config for which cache metrics to record."""

    record_cache_miss_counter: bool = False
    record_tablewise_cache_miss: bool = False


class MultiPassPrefetchConfig(NamedTuple):
    """Config for multi-pass prefetch in TBE cache."""

    num_passes: int = 1
    min_splitable_pass_size: int = 6 * 1024 * 1024


class CacheState:
    """Runtime state of the TBE embedding cache."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class SplitState:
    """Internal split-table state container."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class KVZCHTBEConfig:
    """Config for key-value ZCH TBE operations."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


MAX_PREFETCH_DEPTH = 100


def construct_cache_state(*args, **kwargs):
    """Build a CacheState from the given arguments."""
    return CacheState()


def get_bounds_check_version_for_platform():
    """Return the default bounds-check mode for the current platform."""
    return BoundsCheckMode.NONE
