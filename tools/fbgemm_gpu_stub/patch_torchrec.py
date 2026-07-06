"""
Post-install patch for torchrec to work without CUDA/distributed deps.

Replaces torchrec/distributed/__init__.py with a minimal stub that only
exports the few utility classes referenced by sparse/ and modules/.
Also strips torchrec/__init__.py of eager imports that pull in distributed/quant.
"""

import importlib
import pathlib

torchrec_spec = importlib.util.find_spec("torchrec")
if torchrec_spec is None or torchrec_spec.origin is None:
    raise RuntimeError("torchrec is not installed")

torchrec_root = pathlib.Path(torchrec_spec.origin).parent

# 1. Replace distributed/__init__.py with minimal stub
dist_init = torchrec_root / "distributed" / "__init__.py"
dist_init.write_text('"""Stubbed — distributed features disabled for CPU/MPS."""\n')

# 2. Stub distributed/types.py (LazyAwaitable, NoWait used by fx/tracer)
dist_types = torchrec_root / "distributed" / "types.py"
dist_types.write_text('''\
"""Minimal stubs for types referenced by torchrec.fx and torchrec.modules."""

import abc
from typing import Generic, TypeVar

W = TypeVar("W")


class Awaitable(abc.ABC, Generic[W]):
    @abc.abstractmethod
    def wait(self) -> W: ...


class NoWait(Awaitable[W]):
    def __init__(self, result: W) -> None:
        self._result = result

    def wait(self) -> W:
        return self._result


class _LazyAwaitableMeta(abc.ABCMeta):
    pass


class LazyAwaitable(Awaitable[W], metaclass=_LazyAwaitableMeta):
    pass


class ShardingEnv2D:
    pass
''')

# 3. Stub distributed/utils.py (none_throws used by quant/)
dist_utils = torchrec_root / "distributed" / "utils.py"
dist_utils.write_text('''\
"""Minimal stub for torchrec.distributed.utils."""

from typing import TypeVar

T = TypeVar("T")


def none_throws(value: T | None, msg: str = "") -> T:
    if value is None:
        raise ValueError(msg or "Unexpected None")
    return value
''')

# 4. Stub fx/__init__.py and fx/tracer.py (uses torch internals that may not match)
fx_dir = torchrec_root / "fx"
fx_dir.mkdir(exist_ok=True)
(fx_dir / "__init__.py").write_text('''\
"""Stubbed torchrec.fx — tracing features disabled for CPU/MPS."""
''')
(fx_dir / "tracer.py").write_text('''\
"""Stubbed torchrec.fx.tracer."""


def is_fx_tracing() -> bool:
    return False


def symbolic_trace(*args, **kwargs):
    raise NotImplementedError("torchrec.fx.tracer is not available in CPU/MPS stub")


class Tracer:
    pass
''')

# 5. Patch torchrec/__init__.py — comment out quant and streamable
init_file = torchrec_root / "__init__.py"
text = init_file.read_text()
replacements = [
    ("import torchrec.distributed  # noqa", "# import torchrec.distributed  # noqa"),
    ("import torchrec.quant  # noqa", "# import torchrec.quant  # noqa"),
    ("from torchrec.streamable import", "# from torchrec.streamable import"),
]
for old, new in replacements:
    text = text.replace(old, new)
init_file.write_text(text)

print(f"Patched torchrec at {torchrec_root}")
