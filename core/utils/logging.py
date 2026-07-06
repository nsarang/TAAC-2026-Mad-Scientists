"""SQLite-backed experiment store. Works air-gapped, no network dependencies."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scalars (
    step      INTEGER NOT NULL,
    key       TEXT    NOT NULL,
    value     REAL,
    timestamp REAL    NOT NULL,
    PRIMARY KEY (key, step)
);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT NOT NULL PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    step      INTEGER NOT NULL,
    key       TEXT    NOT NULL,
    path      TEXT    NOT NULL,
    dtype     TEXT    NOT NULL,
    timestamp REAL    NOT NULL,
    PRIMARY KEY (key, step)
);
"""


class SQLiteStore:
    """Per-run structured store backed by SQLite.

    Parameters
    ----------
    run_dir
        Directory for this run. A ``run.db`` database and ``artifacts/``
        subdirectory are created inside it.
    """

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_dir = self.run_dir / "artifacts"
        self._artifact_dir.mkdir(exist_ok=True)
        self._conn = sqlite3.connect(self.run_dir / "run.db")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def log(self, key: str, value: int | float | bool, step: int) -> None:
        """Log a scalar value."""
        self._conn.execute(
            "INSERT OR REPLACE INTO scalars (step, key, value, timestamp) VALUES (?, ?, ?, ?)",
            (step, key, float(value), time.time()),
        )
        self._conn.commit()

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log a dict of scalars, flattening nested dicts with ``/``."""
        rows = []
        ts = time.time()
        self._flatten(metrics, prefix, rows, ts, step)
        self._conn.executemany(
            "INSERT OR REPLACE INTO scalars (step, key, value, timestamp) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def _flatten(
        self,
        d: dict[str, Any],
        prefix: str,
        rows: list[tuple[int, str, float, float]],
        ts: float,
        step: int,
    ) -> None:
        for k, v in d.items():
            full_key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                self._flatten(v, full_key, rows, ts, step)
            elif v is not None and isinstance(v, (int, float, bool)):
                rows.append((step, full_key, float(v), ts))

    def log_artifact(
        self,
        key: str,
        value: np.ndarray | Any,
        step: int,
    ) -> Path:
        """Save an artifact to disk and record it in the database.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        safe_key = key.replace("/", "_")
        if isinstance(value, np.ndarray):
            filename = f"{safe_key}_{step}.npy"
            path = self._artifact_dir / filename
            np.save(path, value)
            dtype = "npy"
        else:
            filename = f"{safe_key}_{step}.json"
            path = self._artifact_dir / filename
            path.write_text(json.dumps(value, default=str))
            dtype = "json"

        self._conn.execute(
            "INSERT OR REPLACE INTO artifacts (step, key, path, dtype, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (step, key, str(path.relative_to(self.run_dir)), dtype, time.time()),
        )
        self._conn.commit()
        return path

    def log_metadata(self, key: str, value: Any) -> None:
        """Log a run-level metadata entry."""
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, json.dumps(value, default=str)),
        )
        self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Run an arbitrary SQL query against this run's database."""
        return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        """Commit and close the database connection."""
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> SQLiteStore:  # noqa: PYI034
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
