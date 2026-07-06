"""Torch device selection helpers."""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def select_device(device_str: str = None) -> torch.device:
    """Return the best available torch device, or the one requested by `device_str`."""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _TransparentParallelMixin:
    """Shared transparency logic for DP and DDP wrappers.

    Makes the wrapper invisible to the rest of the codebase:
    - Attribute access falls through to the inner module.
    - ``named_parameters`` / ``named_modules`` strip the ``module.`` prefix.
    - ``state_dict`` / ``load_state_dict`` delegate directly to the inner module.

    Only ``forward()`` (via ``__call__``) goes through the parallel machinery.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Delegate to inner module, stripping the ``module.`` prefix."""
        yield from self.module.named_parameters(prefix=prefix, recurse=recurse)

    def named_modules(
        self, memo: set[nn.Module] = None, prefix: str = "", remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Module]]:
        """Delegate to inner module, stripping the ``module.`` prefix."""
        yield from self.module.named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Delegate to inner module so checkpoint keys have no ``module.`` prefix."""
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to inner module so checkpoint keys have no ``module.`` prefix."""
        return self.module.load_state_dict(*args, **kwargs)


class TransparentDataParallel(_TransparentParallelMixin, nn.DataParallel):
    """``nn.DataParallel`` that delegates attribute access to the inner module."""


class TransparentDDP(_TransparentParallelMixin, DistributedDataParallel):
    """``DistributedDataParallel`` that delegates attribute access to the inner module.

    Only ``forward()`` routes through DDP (triggering gradient all-reduce).
    All other method calls (get_sparse_params, reinit, etc.) reach the inner
    module directly — these are single-rank operations.
    """
