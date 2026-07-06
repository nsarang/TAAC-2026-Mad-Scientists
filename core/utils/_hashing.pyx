# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""C-accelerated blake2b hashing for token ID computation."""

from cpython.bytes cimport PyBytes_AS_STRING
from cpython.list cimport PyList_GET_ITEM, PyList_GET_SIZE
from cpython.object cimport PyObject
from libc.stdint cimport int64_t, uint8_t, uint64_t

import numpy as np

cimport numpy as np

np.import_array()

cdef extern from "blake2.h" nogil:
    int blake2b(
        uint8_t *out,
        const void *inp,
        const void *key,
        size_t outlen,
        size_t inlen,
        size_t keylen,
    )

cdef extern from "Python.h":
    const char *PyUnicode_AsUTF8AndSize(object o, Py_ssize_t *size)

cdef uint64_t _MAX_INT63 = (<uint64_t>1 << 63) - 1


cdef inline uint64_t _c_hash64(const char *data, Py_ssize_t length) nogil:
    cdef uint8_t digest[8]
    blake2b(digest, data, NULL, 8, <size_t>length, 0)
    return (
        (<uint64_t>digest[0] << 56) | (<uint64_t>digest[1] << 48)
        | (<uint64_t>digest[2] << 40) | (<uint64_t>digest[3] << 32)
        | (<uint64_t>digest[4] << 24) | (<uint64_t>digest[5] << 16)
        | (<uint64_t>digest[6] << 8) | <uint64_t>digest[7]
    )


def stable_hash64(str value) -> int:
    """8-byte blake2b hash mapped to a positive 63-bit integer.

    Drop-in replacement for the pure-Python version in hashing.py.
    """
    cdef Py_ssize_t length
    cdef const char *data = PyUnicode_AsUTF8AndSize(value, &length)
    cdef uint64_t h = _c_hash64(data, length)
    return <int64_t>((h % _MAX_INT63) + 1)


def token_id(str signature, int vocab_size) -> int:
    """Hash a signature string into a token ID in [1, vocab_size)."""
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    cdef Py_ssize_t length
    cdef const char *data = PyUnicode_AsUTF8AndSize(signature, &length)
    cdef uint64_t h = _c_hash64(data, length)
    return <int64_t>(((h % _MAX_INT63 + 1) % (<uint64_t>(vocab_size - 1))) + 1)


def batch_token_ids(list signatures, int vocab_size) -> np.ndarray:
    """Hash a list of strings into an int64 array of token IDs.

    Each element is mapped to [1, vocab_size) using blake2b.
    """
    cdef Py_ssize_t n = PyList_GET_SIZE(signatures)
    cdef np.ndarray[np.int64_t, ndim=1] out = np.empty(n, dtype=np.int64)
    cdef int64_t *out_ptr = <int64_t *>np.PyArray_DATA(out)
    cdef uint64_t vs = <uint64_t>(vocab_size - 1)
    cdef const char *data
    cdef Py_ssize_t length
    cdef Py_ssize_t i
    cdef uint64_t h

    for i in range(n):
        data = PyUnicode_AsUTF8AndSize(<object>PyList_GET_ITEM(signatures, i), &length)
        h = _c_hash64(data, length)
        out_ptr[i] = <int64_t>(((h % _MAX_INT63 + 1) % vs) + 1)

    return out
