"""
Custom OmegaConf resolvers.

``eval``
    Evaluate a Python expression: ``${eval:" 10 if x else 5 "}``

``now``
    Current UTC timestamp: ``${now:}`` -> ``20260427-143012``
    Custom format: ``${now:%Y%m%d}`` -> ``20260427``

``first``
    First non-null argument: ``${first:${experiment_name},${model.name}}``
"""

from datetime import datetime, timezone

from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass

try:
    OmegaConf.register_new_resolver(
        "now",
        lambda fmt="": datetime.now(timezone.utc).strftime(fmt or "%Y%m%d-%H%M%S"),
    )
except ValueError:
    pass

try:
    OmegaConf.register_new_resolver(
        "first",
        lambda *args: next((a for a in args if a is not None), None),
    )
except ValueError:
    pass


# ---------------------------------------------------------------------------
# Monkey-patch: ``__replace__: true`` support for OmegaConf.merge
# ---------------------------------------------------------------------------
# OmegaConf 2.3.0 always deep-merges dicts.  Hydra adds ``_replace_`` but
# that concept doesn't exist in standalone OmegaConf.  This patch intercepts
# ``_map_merge`` so that a source dict containing ``__replace__: true`` fully
# replaces the destination dict instead of merging into it.
from omegaconf._utils import _get_value
from omegaconf.basecontainer import BaseContainer

_orig_map_merge = BaseContainer._map_merge


def _patched_map_merge(dest, src):
    if not src._is_missing() and "__replace__" in list(src):
        val = _get_value(src._get_node("__replace__", validate_access=False))
        if val is True:
            src_keys = {k for k in src if k != "__replace__"}
            for key in list(dest):
                if key not in src_keys:
                    del dest[key]
    _orig_map_merge(dest, src)
    if "__replace__" in dest:
        del dest["__replace__"]


BaseContainer._map_merge = staticmethod(_patched_map_merge)
