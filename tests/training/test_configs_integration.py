"""CPU smoke tests for the published DragonChariot configurations."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_r2"
SAMPLE_SCHEMA = SAMPLE_DIR / "schema.json"

_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": str(PROJECT_ROOT),
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

# Adapted from origin/main:tests/training/test_pp_regression.py. These overrides
# preserve the model paths while keeping the sample-data run small and deterministic.
_FAST_OVERRIDES = [
    "data.batch_size=32",
    "data.val_batch_size=32",
    "train.seed=42",
    "train.mid_epoch_evals=0",
    "train.progress_bar=false",
    "train.amp_dtype=null",
    "train.device=cpu",
    "train.ddp=false",
    "train.data_parallel=false",
    "train.enable_torch_compile=false",
    "train.pretrain_phase.steps=5",
    "data.num_workers=0",
    "data.val_num_workers=0",
    "data.valid_ratio=0.5",
    "data.train_ratio=0.5",
    "data.shuffle_train=false",
    "data.streaming_train=false",
    "data.streaming_val=false",
    "data.val_cache.enabled=false",
    "model.hash_cardinality_threshold=50000",
    "diagnostics.active_codes=[metrics,done]",
    "diagnostics.warmup_steps=0",
    "diagnostics.n_logs_per_epoch=2",
]

_CONFIGS = [
    pytest.param("small", "configs/final/small_model.yaml", 2, True, id="small"),
    pytest.param("medium", "configs/final/medium_yaml.yaml", 1, False, id="medium"),
    pytest.param("full", "configs/final/full_model.yaml", 1, False, id="full"),
]


def _run(action: str, config: str | Path, overrides: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.execute",
            action,
            "--config",
            str(config),
            *overrides,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=PROJECT_ROOT,
        check=False,
        env=_SUBPROCESS_ENV,
    )


def _assert_success(result: subprocess.CompletedProcess, action: str) -> None:
    assert result.returncode == 0, (
        f"{action} failed (exit {result.returncode}).\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )


def _assert_cpu_dependencies() -> None:
    missing = [
        name for name in ("torchrec", "fbgemm_gpu") if importlib.util.find_spec(name) is None
    ]
    assert not missing, (
        f"Missing CPU test dependencies: {', '.join(missing)}. "
        "Run `make install-deps && make install-torchrec`."
    )


def _check_inference(checkpoint_dir: Path, tmp_path: Path) -> None:
    infer_dir = tmp_path / "inference"
    result = _run(
        "infer",
        checkpoint_dir / "config.yaml",
        [
            f"data.dataset_path={SAMPLE_DIR}",
            f"data.schema_path={checkpoint_dir / 'schema.json'}",
            f"train.checkpoint.dir={checkpoint_dir}",
            f"train.output_dir={infer_dir}",
            "train.device=cpu",
            "data.batch_size=32",
            "data.val_batch_size=32",
            "data.num_workers=0",
            "data.val_num_workers=0",
            "data.streaming_train=false",
            "data.streaming_val=false",
        ],
    )
    _assert_success(result, "inference")
    assert "Restored pre-fitted block state" in result.stdout + result.stderr

    prediction_path = infer_dir / "predictions.json"
    assert prediction_path.exists()
    predictions = json.loads(prediction_path.read_text())["predictions"]
    expected_rows = pq.read_metadata(next(SAMPLE_DIR.glob("*.parquet"))).num_rows
    assert len(predictions) == expected_rows
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in predictions.values())


@pytest.mark.slow
@pytest.mark.parametrize("name,config,epochs,check_inference", _CONFIGS)
def test_config_cpu_smoke(
    name: str,
    config: str,
    epochs: int,
    check_inference: bool,
    tmp_path: Path,
) -> None:
    """Train each config; also verify checkpoint inference for the small model."""
    _assert_cpu_dependencies()
    run_dir = tmp_path / name
    result = _run(
        "train",
        config,
        [
            f"train.max_epochs={epochs}",
            f"train.output_dir={run_dir}",
            f"data.dataset_path={SAMPLE_DIR}",
            f"data.schema_path={SAMPLE_SCHEMA}",
            *_FAST_OVERRIDES,
        ],
    )
    _assert_success(result, f"{name} training")

    log_path = run_dir / "train.log"
    assert log_path.exists()
    auc_matches = re.findall(r"DONE:best_auc=([0-9.]+)", log_path.read_text())
    assert auc_matches, "Training completed without a DONE.best_auc diagnostic"
    assert 0.0 <= float(auc_matches[-1]) <= 1.0

    best_dirs = list((run_dir / "checkpoints").glob("*.best_model"))
    assert len(best_dirs) == 1
    checkpoint_dir = best_dirs[0]
    for filename in ("model.pt", "config.yaml", "schema.json"):
        assert (checkpoint_dir / filename).exists()

    if check_inference:
        _check_inference(checkpoint_dir, tmp_path)
