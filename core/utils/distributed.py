"""Distributed training utilities (DDP)."""

from __future__ import annotations

import functools
import os

import torch
import torch.distributed as dist


def get_rank_info() -> tuple[int, int, int]:
    """Read rank and world_size from environment variables set by torchrun/TorchDistributor.

    Returns
    -------
    tuple
        ``(local_rank, global_rank, world_size)``
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, global_rank, world_size


def is_distributed() -> bool:
    """True if torch.distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def rank_zero_only(fn):
    """Decorator that skips execution on non-zero ranks."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_distributed() or dist.get_rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapper


def broadcast_bool(value: bool, src: int = 0) -> bool:
    """Broadcast a boolean from src rank to all ranks."""
    if not is_distributed():
        return value
    tensor = torch.tensor([int(value)], dtype=torch.int32, device="cuda")
    dist.broadcast(tensor, src=src)
    return bool(tensor.item())


def log_platform_topology() -> None:
    """Print GPU count, distributed env vars, device topology, and nproc."""
    import subprocess

    def _run(cmd):
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10, check=False
            )
            return r.stdout.strip() or "(empty)"
        except Exception as e:
            return f"(error: {e})"

    print(f"device_count={torch.cuda.device_count()}", flush=True)
    for key in [
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "NVIDIA_VISIBLE_DEVICES",
    ]:
        val = os.environ.get(key)
        if val is not None:
            print(f"  {key}={val}", flush=True)
    print(f"nvidia-smi -L: {_run('nvidia-smi -L 2>/dev/null')}", flush=True)
    print(f"nproc: {_run('nproc 2>/dev/null')}", flush=True)
    print(f"/dev/nvidia*: {_run('ls /dev/nvidia* 2>/dev/null')}", flush=True)
