"""Bundle a Python entry point and its local dependencies into a single file."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
from omegaconf import OmegaConf

from core.config.loader import load_yaml
from tools.bundler.analysis import analyze_modules, validate_no_dunder_file
from tools.bundler.discovery import (
    discover_dependencies,
    infer_root,
    resolve_dotted_path,
)
from tools.bundler.transform import assemble_output

_DOTTED_PATH_RE = re.compile(r"^[a-zA-Z_]\w+(?:\.[a-zA-Z_]\w+){2,}$")


def _extract_dotted_paths(obj: object) -> list[str]:
    """Walk a parsed YAML structure and collect string values that look like
    qualified Python paths (at least 3 dot-separated identifiers).
    """
    results: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            results.extend(_extract_dotted_paths(v))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_extract_dotted_paths(item))
    elif isinstance(obj, str) and _DOTTED_PATH_RE.match(obj):
        results.append(obj)
    return results


def _resolve_config_entry_points(config_paths: tuple[str, ...], root: Path) -> list[Path]:
    """Parse YAML configs (with __inherits__ resolved) and resolve dotted Python
    paths to local .py files."""
    seen: set[Path] = set()
    for cfg_path in config_paths:
        cfg = load_yaml(cfg_path)
        data = OmegaConf.to_container(cfg, resolve=False)
        if data is None:
            continue
        for dotted in _extract_dotted_paths(data):
            resolved = resolve_dotted_path(dotted, root)
            if resolved is not None and resolved not in seen:
                seen.add(resolved)
    return sorted(seen)


def _copy_assets(assets: tuple[str, ...], output: Path) -> None:
    out_dir = output.parent
    for spec in assets:
        src_str, dest_rel = spec.rsplit(":", 1) if ":" in spec else (spec, ".")
        src = Path(src_str).resolve()
        dest_dir = (out_dir / dest_rel).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest_dir / src.name, dirs_exist_ok=True)
            click.echo(f"Copied directory {src} -> {dest_dir / src.name}")
        elif src.is_file():
            shutil.copy2(src, dest_dir / src.name)
            click.echo(f"Copied {src} -> {dest_dir / src.name}")
        else:
            raise click.BadParameter(f"asset source does not exist: {src}")


@click.command()
@click.argument("entry_point", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", required=True, type=click.Path(), help="Output file path.")
@click.option("--include-all", is_flag=True, help="Bundle every .py under the project root.")
@click.option(
    "--config",
    "-c",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML config whose dotted-path references (engine, model.target, _cls) "
    "are resolved to .py files and used as additional entry points for "
    "dependency discovery. Repeatable.",
)
@click.option(
    "--asset",
    "-a",
    multiple=True,
    help="Copy an asset file or directory alongside the bundle. "
    "Format: SRC[:DEST_DIR]. DEST_DIR is relative to the bundle output "
    "directory (defaults to '.').",
)
def main(
    entry_point: str,
    output: str,
    include_all: bool,
    config: tuple[str, ...],
    asset: tuple[str, ...],
) -> None:
    """Bundle ENTRY_POINT and all its local dependencies into a single .py file."""
    entry = Path(entry_point).resolve()
    root = infer_root(entry)

    extra_entry_points: list[Path] = []
    if config:
        extra_entry_points = _resolve_config_entry_points(config, root)
        if extra_entry_points:
            click.echo(f"Config-derived entry points ({len(extra_entry_points)}):")
            for ep in extra_entry_points:
                click.echo(f"  {ep}")

    modules = discover_dependencies(
        entry, include_all=include_all, extra_entry_points=extra_entry_points or None
    )
    click.echo(f"Discovered {len(modules)} modules:")
    for p, _ in modules:
        click.echo(f"  {p}")

    validate_no_dunder_file(modules)

    analysis = analyze_modules(modules)
    if analysis.renames:
        click.echo(f"Resolved {len(analysis.renames)} name collision(s):")
        for r in analysis.renames:
            click.echo(f"  {r.file_path.name}: {r.original_name} -> {r.new_name}")

    result = assemble_output(modules, analysis, entry, root)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result)
    click.echo(f"Bundled {len(modules)} modules -> {out_path}")

    if asset:
        _copy_assets(asset, out_path)


if __name__ == "__main__":
    main()
