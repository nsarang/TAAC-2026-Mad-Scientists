"""Build Cython extensions in-place: python setup.py build_ext --inplace."""

from __future__ import annotations

import numpy as np
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

extensions = []
if USE_CYTHON:
    extensions.append(
        Extension(
            "core.utils._hashing",
            sources=["core/utils/_hashing.pyx"],
            include_dirs=[np.get_include(), "/opt/homebrew/include"],
            library_dirs=["/opt/homebrew/lib"],
            libraries=["b2"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    )
    extensions = cythonize(extensions, language_level=3)

setup(ext_modules=extensions)
