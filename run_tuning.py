"""
run_tuning.py

Reads tuning_config.yml and launches tune_retriever.py for one dataset (or all,
sequentially) with the merged `defaults` + per-dataset settings.

    python run_tuning.py --config tuning_config.yml --dataset litwd19k
    python run_tuning.py --config tuning_config.yml --all --dry-run
    python run_tuning.py --config tuning_config.yml --dataset litwd1k --fresh-study

"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def build_command(settings: dict, tuner: Path, fresh_study: bool) -> list[str]:
    cmd = [sys.executable, str(tuner)]
    for key, value in settings.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, list):
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([flag, str(value)])
    if fresh_study:
        cmd.append("--fresh-study")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("tuning_config.yml"))
    parser.add_argument("--tuner", type=Path, default=Path(__file__).with_name("tune_retriever.py"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Key under `datasets:` in the config, e.g. litwd19k")
    group.add_argument("--all", action="store_true", help="Run every dataset in the config, in file order")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh-study", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    defaults = cfg.get("defaults", {})
    datasets = cfg["datasets"]

    names = list(datasets) if args.all else [args.dataset]
    for name in names:
        if name not in datasets:
            raise SystemExit(f"Unknown dataset '{name}'. Available: {', '.join(datasets)}")

    for name in names:
        settings = {**defaults, **datasets[name]}
        cmd = build_command(settings, args.tuner, args.fresh_study)
        print(f"\n# {name}\n{shlex.join(cmd)}")
        if not args.dry_run:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise SystemExit(f"tune_retriever.py failed for '{name}' (exit {result.returncode}); stopping.")


if __name__ == "__main__":
    main()