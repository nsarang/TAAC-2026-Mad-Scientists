"""Stub for fbgemm_gpu.split_table_batched_embeddings_ops_training."""

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType  # noqa: F401
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (  # noqa: F401
    BoundsCheckMode,
    ComputeDevice,
    EmbeddingLocation,
    PoolingMode,
)
from torch import nn

# Re-export so callers can do `from ...ops_training import OptimType`
OptimType = EmbOptimType


class SplitTableBatchedEmbeddingBagsCodegen(nn.Module):
    """
    CPU/MPS fallback: wraps multiple nn.EmbeddingBag (pooled) or nn.Embedding
    (NONE mode) into one module.

    Not fused — just provides the same interface so torchrec modules
    can instantiate it without crashing. Actual performance comes from
    the real CUDA TBE kernel.
    """

    def __init__(
        self,
        embedding_specs: list[tuple[int, int, EmbeddingLocation, ComputeDevice]],
        pooling_mode: PoolingMode = PoolingMode.SUM,
        optimizer: EmbOptimType = EmbOptimType.NONE,
        learning_rate: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__()
        self.pooling_mode = pooling_mode
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._dims: list[int] = []
        self._num_rows: list[int] = []

        if pooling_mode == PoolingMode.NONE:
            # Real TBE stores weights as flat buffers, not nn.Parameters.
            # Register as buffers so .to(device) moves them correctly.
            self._weight_buffers: list[torch.Tensor] = []
            for i, (num_embeddings, embedding_dim, _location, _device) in enumerate(
                embedding_specs
            ):
                buf = torch.zeros(num_embeddings, embedding_dim)
                self.register_buffer(f"_weight_{i}", buf)
                self._weight_buffers.append(buf)
                self._dims.append(embedding_dim)
                self._num_rows.append(num_embeddings)
        else:
            mode = {PoolingMode.SUM: "sum", PoolingMode.MEAN: "mean"}.get(pooling_mode, "sum")
            self.embedding_bags = nn.ModuleList()
            for num_embeddings, embedding_dim, _location, _device in embedding_specs:
                self.embedding_bags.append(
                    nn.EmbeddingBag(
                        num_embeddings, embedding_dim, mode=mode, include_last_offset=True
                    )
                )
                self._dims.append(embedding_dim)
                self._num_rows.append(num_embeddings)

        self._total_dim = sum(self._dims)

        # Adagrad accumulator state (per-row sum of squared gradients)
        if optimizer == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
            self._momentum: list[torch.Tensor] = []
            for num_embeddings, *_ in embedding_specs:
                self._momentum.append(torch.zeros(num_embeddings))

    def forward(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Look up embeddings, optionally applying per_sample_weights."""
        if self.pooling_mode == PoolingMode.NONE:
            return self._forward_none(indices, offsets, per_sample_weights)
        return self._forward_pooled(indices, offsets, per_sample_weights)

    def _forward_none(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """NONE mode: sequence embedding, output (T * total_tokens, D)."""
        num_tables = len(self._weight_buffers)
        batch_size = (offsets.numel() - 1) // num_tables
        results = []
        for i, weight in enumerate(self._weight_buffers):
            seg_start = i * batch_size
            seg_end = seg_start + batch_size
            idx_start = offsets[seg_start].item()
            idx_end = offsets[seg_end].item()
            table_indices = indices[idx_start:idx_end]
            emb = torch.nn.functional.embedding(table_indices, weight)
            if per_sample_weights is not None:
                psw = per_sample_weights[idx_start:idx_end]
                emb = emb * psw.unsqueeze(-1)
            results.append(emb)
        # Feature-major layout: all tokens for table 0, then table 1, etc.
        return torch.cat(results, dim=0)

    def _forward_pooled(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Pooled mode (SUM/MEAN): output (B, sum_of_dims)."""
        num_tables = len(self.embedding_bags)
        batch_size = (offsets.numel() - 1) // num_tables
        results = []
        for i, bag in enumerate(self.embedding_bags):
            seg_start = i * batch_size
            seg_end = seg_start + batch_size
            table_offsets = offsets[seg_start : seg_end + 1] - offsets[seg_start]
            idx_start = offsets[seg_start].item()
            idx_end = offsets[seg_end].item()
            table_indices = indices[idx_start:idx_end]
            psw = None
            if per_sample_weights is not None:
                psw = per_sample_weights[idx_start:idx_end]
            results.append(bag(table_indices, table_offsets, psw))
        return torch.cat(results, dim=1)

    def split_embedding_weights(self) -> list[torch.Tensor]:
        """Return per-table weight tensors (views into the underlying storage)."""
        if self.pooling_mode == PoolingMode.NONE:
            # Return from registered buffers so .to(device/dtype) is reflected
            return [getattr(self, f"_weight_{i}") for i in range(len(self._dims))]
        return [b.weight.data for b in self.embedding_bags]

    def get_optimizer_state(self) -> list[dict[str, torch.Tensor]]:
        """Return per-table optimizer state dicts with a 'sum' key (rowwise accumulator)."""
        if not hasattr(self, "_momentum"):
            return [{} for _ in self._num_rows]
        return [{"sum": m} for m in self._momentum]

    def set_learning_rate(self, lr: float) -> None:
        """Set the internal learning rate (takes effect on next forward+backward)."""
        self._learning_rate = lr

    def get_learning_rate(self) -> float:
        """Return the current learning rate."""
        return self._learning_rate
