"""Compact array codec for DIAG payloads: quantize/cast -> zlib -> base64.

Keeps diagnostic arrays (cross-interaction matrices, 2-D projection coordinates)
inside the text log — the only egress on a TensorBoard-only platform — without
bloating it. ``uint16`` is a lossy max-abs linear quantization for non-negative
data (e.g. Frobenius norms); ``float16`` casts signed data (e.g. projection
coordinates) with ~3-4 significant figures. Payload format is
``tag|scale|shape|base64`` — base64's alphabet never contains the ``|`` / ``;;`` /
``:`` protocol separators.
"""

from __future__ import annotations

import base64
import zlib

import numpy as np


def encode_array(arr: np.ndarray, dtype: str = "uint16") -> str:
    """Encode a float array as ``tag|scale|shape|base64(zlib(bytes))``."""
    arr = np.ascontiguousarray(np.nan_to_num(np.asarray(arr, dtype=np.float64)))
    shape = ",".join(str(s) for s in arr.shape)
    if dtype == "uint16":
        scale = float(np.abs(arr).max()) or 1.0
        q = np.round(arr / scale * 65535.0).clip(0, 65535).astype("<u2")
        blob = base64.b64encode(zlib.compress(q.tobytes(), 9)).decode("ascii")
        return f"u16|{scale:.8g}|{shape}|{blob}"
    if dtype == "float16":
        blob = base64.b64encode(zlib.compress(arr.astype("<f2").tobytes(), 9)).decode("ascii")
        return f"f16|0|{shape}|{blob}"
    raise ValueError(f"unknown codec dtype {dtype!r}")


def decode_array(payload: str) -> np.ndarray:
    """Inverse of :func:`encode_array`."""
    tag, scale, shape, blob = payload.split("|", 3)
    raw = zlib.decompress(base64.b64decode(blob))
    shp = tuple(int(s) for s in shape.split(",")) if shape else ()
    if tag == "u16":
        q = np.frombuffer(raw, dtype="<u2").reshape(shp).astype(np.float64)
        return q * (float(scale) / 65535.0)
    if tag == "f16":
        return np.frombuffer(raw, dtype="<f2").reshape(shp).astype(np.float64)
    raise ValueError(f"unknown codec tag {tag!r}")
