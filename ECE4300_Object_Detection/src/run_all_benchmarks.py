"""
ECE 4300 - Step 5: Run benchmark configurations A-D one after another.

Each configuration runs in its own Python process, strictly sequentially
(never two at once). This keeps memory use low on an 8 GB Mac and gives
every configuration the same clean starting state. All runs share one
session ID, and the summary/speedup table for that session is printed and
saved at the end.

A configuration that fails (for example, MPS not available) is recorded in
results/raw/failed_runs.csv and reported; the remaining configurations
still run.

Run from the project root:
    python -m src.run_all_benchmarks               # A, B, C, D
    python -m src.run_all_benchmarks --configs A C # a subset, in the given order
"""

import argparse
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from src import compare_results
from src.benchmark_config import CONFIG_PATH, PROJECT_ROOT, ConfigError, find_config, load_all_configs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run benchmark configurations sequentially.")
    p.add_argument("--configs", nargs="+", default=None,
                   help="Configuration keys or names to run (default: all, A-D)")
    p.add_argument("--config-file", type=Path, default=CONFIG_PATH)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        configs = load_all_configs(args.config_file)
        selected = ([find_config(c, configs) for c in args.configs]
                    if args.configs else list(configs.values()))
    except ConfigError as e:
        print(f"ERROR: {e}")
        return 1

    session_id = f"S{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    print(f"Session {session_id}: running {', '.join(c.name for c in selected)} sequentially.\n",
          flush=True)

    exit_codes = {}
    for cfg in selected:
        cmd = [sys.executable, "-m", "src.benchmark", "--config", cfg.key,
               "--session-id", session_id, "--config-file", str(args.config_file)]
        exit_codes[cfg.name] = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
        print(flush=True)

    print("Run status:")
    for name, code in exit_codes.items():
        print(f"  {name:<8} {'OK' if code == 0 else f'FAILED (exit code {code})'}")
    print()

    compare_results.main(["--session", session_id, "--config-file", str(args.config_file)])
    return 0 if all(code == 0 for code in exit_codes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
