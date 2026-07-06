"""Render 2-D RepresentationProbe coordinates from DIAG logs.

Reads a training log with `parse_log`, loads the already-computed 2-D
coordinates emitted by ``RepresentationProbeCode``, and renders a scatter
coloured by emitted metadata. No full representation vectors are stored in or
read from the log.

Example
-------
    python scripts/repr_projection.py --log runs/<run>/train.log \
        --rep repr_probes_seq_repr --color-by label
"""

from __future__ import annotations

from pathlib import Path

import click
import plotly.express as px

from core.training.callbacks.diagnostics import parse_log


def _load(log_path: str) -> tuple[dict, dict, dict]:
    """Load projection coordinates and metadata from the newest REPR_PROBE run."""
    for exp in reversed(parse_log(path=log_path)):
        acc = exp.get("REPR_PROBE")
        projection = acc.get("projection") if acc else None
        if projection and projection.get("coords"):
            return (
                projection["coords"],
                projection.get("meta", {}),
                projection.get("manifest", {}),
            )
    raise click.ClickException("no REPR_PROBE projection data in log")


def _color(meta: dict, manifest: dict, color_by: str):
    if color_by not in meta:
        raise click.ClickException(
            f"{color_by!r} not in projection metadata. Available: {sorted(meta)}"
        )
    values = meta[color_by]
    categories = manifest.get("categories", {}).get(color_by)
    if categories:
        lookup = {float(k): v for k, v in categories.items()}
        return [lookup.get(float(v), str(v)) for v in values], True
    if color_by == "label":
        return values.astype(int).astype(str), True
    return values, False


@click.command()
@click.option("--log", "log_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--rep", default=None, help="Projected representation key; default: first available.")
@click.option(
    "--color-by",
    type=str,
    default="label",
    help="Projection metadata field to colour by.",
)
@click.option("--out", default=None, type=click.Path(dir_okay=False))
def main(log_path: str, rep: str, color_by: str, out: str) -> None:
    """Render a precomputed 2-D representation projection from a training log."""
    coords_by_rep, meta, manifest = _load(log_path)
    rep_key = rep or sorted(coords_by_rep)[0]
    if rep_key not in coords_by_rep:
        raise click.ClickException(f"{rep_key!r} not in log. Available: {sorted(coords_by_rep)}")

    coords = coords_by_rep[rep_key]
    color, categorical = _color(meta, manifest, color_by)
    method = manifest.get("method", "projection").upper()

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=color,
        labels={"x": f"{method}-1", "y": f"{method}-2", "color": color_by},
        title=f"{rep_key} — {method} coloured by {color_by} (n={coords.shape[0]})",
        color_continuous_scale=None if categorical else "Viridis",
        opacity=0.7,
    )
    fig.update_traces(marker={"size": 4})
    out_path = (
        Path(out)
        if out
        else Path(log_path).with_name(
            f"projection_{rep_key}_{manifest.get('method', 'projection')}_{color_by}.html"
        )
    )
    fig.write_html(str(out_path))
    click.echo(f"wrote {out_path}")


if __name__ == "__main__":
    main()
