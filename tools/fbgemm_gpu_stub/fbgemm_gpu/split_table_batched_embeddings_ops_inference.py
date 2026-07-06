"""Stub for fbgemm_gpu.split_table_batched_embeddings_ops_inference."""

from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen as IntNBitTableBatchedEmbeddingBagsCodegen,
)

__all__ = ["EmbeddingLocation", "IntNBitTableBatchedEmbeddingBagsCodegen", "PoolingMode"]
