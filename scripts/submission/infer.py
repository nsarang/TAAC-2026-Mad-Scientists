"""Inference entry point for the AngelML evaluation container.

The platform calls ``main()`` with no arguments. All paths come from
environment variables. Delegates to the bundled ``execute.py`` sitting
in the same directory.
"""

from __future__ import annotations

import os
import subprocess
import sys

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "extra.txt"),
    ]
)

_SCRIPT_DIR = os.environ.get("EVAL_INFER_PATH") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)


import logging

from execute import run_action

LOG = logging.getLogger(__name__)


def main() -> None:
    """Load a trained checkpoint and write predictions for the eval split."""
    model_dir = os.environ["MODEL_OUTPUT_PATH"]
    data_dir = os.environ["EVAL_DATA_PATH"]
    result_dir = os.environ["EVAL_RESULT_PATH"]

    config_path = os.path.join(model_dir, "config.yaml")

    LOG.info(f"Model dir: {model_dir}")
    LOG.info(f"Data dir: {data_dir}")
    LOG.info(f"Result dir: {result_dir}")
    LOG.info(f"Config path: {config_path}")

    overrides = [
        f"data.dataset_path={data_dir}",
        f"data.schema_path={model_dir}/schema.json",
        f"train.checkpoint.dir={model_dir}",
        f"train.output_dir={result_dir}",
    ]

    LOG.info(f"Running inference with overrides: {overrides}")
    run_action("infer", config_path, tuple(overrides))


if __name__ == "__main__":
    main()
