#!/usr/bin/env python3
"""Extract prior trial results from a tuning log for warm-starting a new study.

Usage
-----
    python tools/extract_prior.py experiment_logs/run1.txt -o prior.json

The output JSON is then passed to bundle-submission via PRIOR_LOG and gets
bundled alongside the config. On the next run, tune.py finds it automatically
and replays the prior trials before sampling new ones.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def extract_prior(log_text: str) -> dict:
    """Parse TUNE|TRIAL_END lines from a tuning log into a prior dict."""
    start_m = re.search(r"TUNE\|START\|study=([^,\s]+)", log_text)
    study_name = start_m.group(1) if start_m else "tune"

    trials = []
    pattern = re.compile(
        r"TUNE\|TRIAL_END\|trial=(\d+),auc=([\d.]+),best_so_far=[\d.]+"
        r";;PARAMS:(\{.*\})"
    )
    for line in log_text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        trial_n, auc, params_json = m.groups()
        trials.append(
            {
                "number": int(trial_n),
                "value": float(auc),
                "params": json.loads(params_json),
            }
        )

    return {"study_name": study_name, "trials": trials}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", help="Path to the tuning log file")
    parser.add_argument(
        "-o", "--output", default="prior.json", help="Output path (default: prior.json)"
    )
    args = parser.parse_args()

    log_text = Path(args.log).read_text()
    prior = extract_prior(log_text)

    if not prior["trials"]:
        print("ERROR: no TUNE|TRIAL_END lines found in log", file=sys.stderr)
        sys.exit(1)

    Path(args.output).write_text(json.dumps(prior, indent=2))
    print(f"Extracted {len(prior['trials'])} trial(s) from '{prior['study_name']}' → {args.output}")
    for t in prior["trials"]:
        print(f"  trial {t['number']}: auc={t['value']:.6f}  params={t['params']}")


if __name__ == "__main__":
    main()
