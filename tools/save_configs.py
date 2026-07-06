"""Save resolved configs into the bundle output directory."""

import argparse
import os

from omegaconf import OmegaConf

from core.config.loader import load_yaml


def main() -> None:
    """Resolve and save configs into the bundle output directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--commit", default="")
    args = parser.parse_args()
    for i, path in enumerate(args.configs, 1):
        path = path.strip("\"'")
        cfg = load_yaml(path)
        if args.commit:
            cfg.source_commit = args.commit
        name = cfg.get("experiment_name", os.path.splitext(os.path.basename(path))[0])
        out_path = os.path.join(args.out, f"config_{str(i).zfill(2)}_{name}.yaml")
        OmegaConf.save(cfg, out_path)
        print(f"Saved config {i}: {out_path}")


if __name__ == "__main__":
    main()
