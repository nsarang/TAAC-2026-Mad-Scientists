"""Produce a self-contained train.py and infer.py for legacy submission."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click

_FUTURE_LINE = "from __future__ import annotations"
_MAIN_SENTINEL = 'if __name__ == "__main__":'

_TRAIN_PREAMBLE = """\
# --- inlined by tools/inline_deps.py ---
import os as _os
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile

_EXTRA_DEPS = {deps_literal}
_EMBEDDED_CONFIGS = {configs_literal}

if _EXTRA_DEPS:
    _subprocess.check_call(
        [_sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", *_EXTRA_DEPS]
    )

_LEGACY_CONFIG_DIR = _tempfile.mkdtemp(prefix="legacy_cfg_")
_LEGACY_CONFIG_PATHS = []
for _name, _content in _EMBEDDED_CONFIGS.items():
    _p = _os.path.join(_LEGACY_CONFIG_DIR, _name)
    with open(_p, "w") as _f:
        _f.write(_content)
    _LEGACY_CONFIG_PATHS.append(_p)
# --- end inlined preamble ---
"""

_TRAIN_MAIN = """\
if __name__ == "__main__":
    import os as _os  # noqa: F811

    _data_path = _os.environ.get("TRAIN_DATA_PATH", "")
    _ckpt_path = _os.environ.get("TRAIN_CKPT_PATH", "")
    _log_path = _os.environ.get("TRAIN_LOG_PATH", "")
    _tf_path = _os.environ.get("TRAIN_TF_EVENTS_PATH", "")

    for _cfg_path in _LEGACY_CONFIG_PATHS:
        _overrides = [
            f"data.dataset_path={_data_path}",
            f"data.schema_path={_data_path}/schema.json",
            f"train.checkpoint.dir={_ckpt_path}",
            f"train.output_dir={_log_path}",
            f"diagnostics.log_dir={_tf_path}",
        ]
        run_action("train", _cfg_path, tuple(_overrides))
"""


def _read_extra_deps(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _deps_literal(deps: list[str]) -> str:
    return "[" + ", ".join(repr(d) for d in deps) + "]"


def _configs_literal(paths: list[Path]) -> str:
    lines = ["{"]
    for p in paths:
        lines.append(f"    {p.name!r}: {p.read_text()!r},")
    lines.append("}")
    return "\n".join(lines)


def _strip_click_main(lines: list[str]) -> list[str]:
    main_idx = next(
        (i for i in range(len(lines) - 1, -1, -1) if lines[i].rstrip() == _MAIN_SENTINEL),
        None,
    )
    if main_idx is None:
        raise ValueError("Could not find __main__ block in execute.py")
    click_idx = next(
        (i for i in range(main_idx - 1, -1, -1) if lines[i].lstrip().startswith("@click.command")),
        main_idx,
    )
    return lines[:click_idx]


def build_train_py(execute_py: Path, extra_txt: Path, config_paths: list[Path]) -> str:
    """Assemble a self-contained train.py from the bundled execute.py."""
    preamble = _TRAIN_PREAMBLE.format(
        deps_literal=_deps_literal(_read_extra_deps(extra_txt)),
        configs_literal=_configs_literal(config_paths),
    )

    source_lines = execute_py.read_text().splitlines()

    future_idx = next(
        (i for i, line in enumerate(source_lines) if line.strip() == _FUTURE_LINE), None
    )
    if future_idx is not None:
        header = source_lines[future_idx] + "\n\n"
        body_lines = source_lines[future_idx + 1 :]
    else:
        header = ""
        body_lines = source_lines

    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    body_lines = _strip_click_main(body_lines)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    return header + preamble + "\n" + "\n".join(body_lines) + "\n\n\n" + _TRAIN_MAIN


def build_infer_py(infer_py: Path, extra_txt: Path) -> str:
    """Assemble a self-contained infer.py with inlined dependency install."""
    deps_lit = _deps_literal(_read_extra_deps(extra_txt))
    source = infer_py.read_text()

    source = re.sub(
        r"subprocess\.check_call\(\s*\[.*?\"extra\.txt\"\).*?\]\s*,?\s*\)\n",
        f"_EXTRA_DEPS = {deps_lit}\n"
        "subprocess.check_call(\n"
        '    [sys.executable, "-m", "pip", "install", "-q", *_EXTRA_DEPS]\n'
        ")\n",
        source,
        flags=re.DOTALL,
    )
    return source.replace("from execute import", "from train import")


@click.command()
@click.option("--execute-py", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--infer-py", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--extra-txt", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--config", "config_paths", multiple=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("--out-dir", required=True, type=click.Path(file_okay=False))
def main(
    execute_py: str, infer_py: str, extra_txt: str, config_paths: tuple[str, ...], out_dir: str
) -> None:
    """Write self-contained train.py and infer.py into `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, builder, args in [
        (
            "train.py",
            build_train_py,
            (Path(execute_py), Path(extra_txt), [Path(c) for c in config_paths]),
        ),
        ("infer.py", build_infer_py, (Path(infer_py), Path(extra_txt))),
    ]:
        src = builder(*args)
        (out / name).write_text(src)
        click.echo(f"Wrote {out / name} ({len(src.splitlines())} lines)")


if __name__ == "__main__":
    main()
