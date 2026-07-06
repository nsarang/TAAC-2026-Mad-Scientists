"""OmegaConf config loading with YAML inheritance and CLI overrides."""

from __future__ import annotations

import glob
import os
import pydoc
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

import core.config.omegaconf as _  # noqa: F401  # register resolvers


def load_yaml(
    filepath: str | Path,
    visited: set[str] = None,
) -> DictConfig:
    """Load a YAML file with ``__inherits__`` support.

    Supports glob patterns for loading multiple files. When a YAML contains an
    ``__inherits__`` key (string or list of strings), each parent is loaded and
    merged before the current file, with the current file winning on conflicts.
    Inheritance paths resolve relative to the directory of the current YAML.

    Parameters
    ----------
    filepath
        Path to a single YAML file or a glob pattern.
    visited
        Tracks already-visited files to detect circular dependencies.

    Returns
    -------
    DictConfig
        Merged configuration with inheritance resolved.
    """
    if visited is None:
        visited = set()

    filepath = os.path.abspath(str(filepath))
    if filepath in visited:
        raise ValueError(f"Circular dependency detected: {filepath}")

    visited = visited | {filepath}

    paths = [os.path.abspath(p) for p in glob.glob(filepath)]
    if not paths:
        raise FileNotFoundError(f"No files found at {filepath}")

    cfg = OmegaConf.merge(*[OmegaConf.load(p) for p in paths])

    if cfg.get("__inherits__") is not None:
        base_dir = os.path.dirname(filepath)
        imports = cfg.__inherits__
        if isinstance(imports, str):
            imports = [imports]

        parent_cfgs = [load_yaml(os.path.join(base_dir, imp), visited=visited) for imp in imports]
        cfg = OmegaConf.merge(*parent_cfgs, cfg)
        del cfg["__inherits__"]

    return cfg


def load_config(
    config_file: str | Path,
    overrides: list[str] = None,
) -> DictConfig:
    """Load a config YAML (with ``__inherits__`` resolved) and apply CLI overrides.

    Parameters
    ----------
    config_file
        Path to the YAML config. Use ``__inherits__`` inside the YAML to pull
        in base/parent configs.
    overrides
        Dot-notation CLI overrides, e.g.
        ``["train.epochs=10", "model.hidden_dim=128"]``.
    """
    cfg = load_yaml(config_file)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def to_dict(cfg: DictConfig, resolve: bool = True) -> dict[str, Any]:
    """Convert an OmegaConf config to a plain dict."""
    return OmegaConf.to_container(cfg, resolve=resolve)


class DotDict(dict):
    """Dict subclass that supports attribute access for keys.

    Enables ``cfg.train.epochs`` instead of ``cfg["train"]["epochs"]``.
    Raises ``AttributeError`` (not ``KeyError``) on missing keys to play
    nicely with ``hasattr`` and ``getattr`` patterns.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


def instantiate_config(config: Any, **kwargs: Any) -> Any:
    """Recursively instantiate ``_cls`` / ``_init`` nodes in a resolved config.

    A dict with ``{"_cls": "module.Class", "_init": {...}}`` is replaced by
    ``Class(**init)``.  ``_init`` may also be a list (passed as ``*args``) or
    a scalar (single positional arg).  A dict with only ``_cls`` returns the
    class/function itself without calling it.

    Nodes with ``_defer: true`` have the flag stripped, their children recursed,
    and are returned as a `DotDict` without instantiation. All other dicts are
    wrapped in `DotDict` recursively.

    Parameters
    ----------
    config
        A resolved config (plain dict/list/scalar), typically from
        ``OmegaConf.to_container(cfg, resolve=True)``.
    **kwargs
        Extra keyword arguments merged into ``_init`` when instantiating a
        ``_cls`` + ``_init`` node. Useful for passing runtime values (e.g.
        ``params=model.parameters()``) to deferred nodes on second pass.
    """
    if isinstance(config, dict):
        if config.get("_defer"):
            stripped = {k: instantiate_config(v) for k, v in config.items() if k != "_defer"}
            return DotDict(stripped)

        if len(config) == 1 and "_cls" in config:
            cls = pydoc.locate(config["_cls"])
            if cls is None:
                raise ValueError(f"Object `{config['_cls']}` not found.")
            return cls

        if len(config) == 2 and "_cls" in config and "_init" in config:
            cls = pydoc.locate(config["_cls"])
            if cls is None:
                raise ValueError(f"Object `{config['_cls']}` not found.")
            args = instantiate_config(config["_init"])
            if isinstance(args, dict):
                merged = {**args, **kwargs}
                return cls(**merged)
            elif isinstance(args, list):
                return cls(*args, **kwargs)
            if kwargs:
                return cls(args, **kwargs)
            return cls(args)

        return DotDict({k: instantiate_config(v) for k, v in config.items()})

    if isinstance(config, list):
        return [instantiate_config(item) for item in config]

    return config
