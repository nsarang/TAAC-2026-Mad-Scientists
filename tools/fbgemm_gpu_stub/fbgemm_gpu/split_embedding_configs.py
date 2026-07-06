"""Stub enums from fbgemm_gpu.split_embedding_configs."""

import enum


@enum.unique
class EmbOptimType(enum.Enum):
    """Embedding optimizer types supported by TBE."""

    SGD = "sgd"
    EXACT_SGD = "exact_sgd"
    LAMB = "lamb"
    ADAM = "adam"
    EXACT_ADAGRAD = "exact_adagrad"
    EXACT_ROWWISE_ADAGRAD = "exact_row_wise_adagrad"
    LARS_SGD = "lars_sgd"
    PARTIAL_ROWWISE_ADAM = "partial_row_wise_adam"
    PARTIAL_ROWWISE_LAMB = "partial_row_wise_lamb"
    ROWWISE_ADAGRAD = "row_wise_adagrad"
    NONE = "none"


@enum.unique
class SparseType(enum.Enum):
    """Quantization types for sparse embedding weights."""

    FP32 = "fp32"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    BF16 = "bf16"

    def as_dtype(self) -> "torch.dtype":  # noqa: F821
        """Convert to the corresponding torch dtype."""
        import torch

        _MAP = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e4m3fn,
            "int8": torch.int8,
        }
        return _MAP.get(self.value, torch.float32)
