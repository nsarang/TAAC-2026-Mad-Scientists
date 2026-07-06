#!/usr/bin/env python
"""Run any engine action (train, infer, …). Engine is resolved from config.

Usage: python -m scripts.execute ACTION [OVERRIDES...]
"""

from __future__ import annotations

import logging
import pydoc
import warnings

import click

from core.config.loader import load_config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s.%(funcName)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger().setLevel(logging.DEBUG)
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._inductor\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\._dynamo\..*")


def run_action(action: str, config: str, overrides: tuple[str, ...]) -> None:
    """Resolve an engine function from config and run it."""
    cfg = load_config(config_file=config, overrides=list(overrides))
    engine_path = cfg.get(action, {}).get("engine")
    if engine_path is None:
        raise RuntimeError(f"Config has no '{action}.engine' key")
    engine_fn = pydoc.locate(engine_path)
    if engine_fn is None:
        raise ImportError(f"Cannot resolve engine: {engine_path}")

    import inspect

    sig = inspect.signature(engine_fn)
    params = list(sig.parameters.keys())
    if len(params) >= 2 and params[0] in ("config_file", "config"):
        engine_fn(config, overrides)
    else:
        engine_fn(cfg)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("action")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Config YAML path.",
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def main(action: str, config: str, overrides: tuple[str, ...]) -> None:
    """Run an action from a YAML config with optional dot-notation overrides."""
    run_action(action, config, overrides)


if __name__ == "__main__":
    main()
