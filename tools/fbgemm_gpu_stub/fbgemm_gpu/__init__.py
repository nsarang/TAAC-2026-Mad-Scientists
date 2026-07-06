"""
Stub fbgemm_gpu package for CPU/MPS environments.

Registers pure-PyTorch fallbacks for torch.ops.fbgemm.* operators so that
torchrec can be imported and run without the real fbgemm_gpu (CUDA-only).
"""

from fbgemm_gpu import ops  # noqa: F401 — triggers op registration
from fbgemm_gpu import sparse_ops  # noqa: F401
