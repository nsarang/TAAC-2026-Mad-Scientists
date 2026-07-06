"""Deterministic hashing utilities."""

from __future__ import annotations

try:
    from core.utils._hashing import batch_token_ids, stable_hash64, token_id
except ImportError:
    import hashlib

    def stable_hash64(value: str) -> int:
        """8-byte blake2b hash mapped to a positive 63-bit integer."""
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
        hashed = int.from_bytes(digest, byteorder="big", signed=False)
        return (hashed % ((1 << 63) - 1)) + 1

    def token_id(signature: str, vocab_size: int) -> int:
        """Hash a signature string into a token ID in [1, vocab_size)."""
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        return (stable_hash64(signature) % (vocab_size - 1)) + 1

    def batch_token_ids(signatures: list[str], vocab_size: int) -> list[int]:
        """Hash a list of strings into token IDs (pure-Python fallback)."""
        import numpy as np

        return np.array([token_id(s, vocab_size) for s in signatures], dtype=np.int64)
