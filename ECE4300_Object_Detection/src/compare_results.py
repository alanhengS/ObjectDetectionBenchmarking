"""
ECE 4300 - Step 5: Summary table and speedup comparisons.

Reads the raw per-trial CSVs in results/raw/, picks one run per
configuration (A-D), and reports for each configuration:
    mean execution time, average time per image, throughput

and the speedups defined in config/benchmark_config.json, using the course
formula:
    Speedup = Baseline execution time / Comparison execution time
(mean execution time over the five measured trials).

No benchmark is run here; this only reads measured data. Configurations
without a successful run are reported as missing/failed, never estimated.

Run from the project root:
    python -m src.compare_results                     # latest run of each configuration
    python -m src.compare_results --session <id>      # runs from one run_all session
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.benchmark_config import (
    CONFIG_PATH, PROJECT_ROOT, ConfigError, load_all_configs, load_comparisons,
)

FAILURES_FILE = "failed_runs.csv"   # same name as in src/benchmark.py
SAME_PERFORMANCE_TOLERANCE = 0.01   # |speedup - 1| below this = "about the same"

SUMMARY_COLUMNS = [
    "config_id", "configuration", "device", "resolution", "status", "run_id",
    "session_id", "timestamp", "trials", "num_images", "warmup_images",
    "mean_total_time_s", "stdev_total_time_s", "min_total_time_s", "max_total_time_s",
    "mean_avg_time_ms_per_image", "mean_throughput_img_per_s", "error",
]
SPEEDUP_COLUMNS = [
    "comparison", "kind", "baseline", "comparison_config",
    "baseline_mean_time_s", "comparison_mean_time_s", "speedup", "interpretation",
]


# ----------------------------------------------------------------------------
# Reading raw results
# ----------------------------------------------------------------------------
def read_raw_rows(raw_dir, configs):
    """
    Return every trial row from results/raw/*.csv.

    Step 4 CSVs lack the Step 5 columns (config_id, device, session_id, ...);
    those are filled in from the configuration name so old baseline runs
    can still be selected.
    """
    by_name = {c.name: c for c in configs.values()}
    rows = []
    for path in sorted(Path(raw_dir).glob("*.csv")):
        if path.name == FAILURES_FILE:
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cfg = by_name.get(row.get("configuration", ""))
                if cfg is None:
                    continue
                row.setdefault("config_id", cfg.key)
                row.setdefault("device", cfg.device)
                row["session_id"] = row.get("session_id") or row["run_id"]
                row.setdefault("warmup_images", "")      # unknown for Step 4 runs
                row.setdefault("model_sha256", "")
                row.setdefault("image_list_sha256", "")
                row.setdefault("measurement", "elapsed_workload_time")
                row["_file"] = path.name
                rows.append(row)
    return rows


def read_failures(raw_dir):
    path = Path(raw_dir) / FAILURES_FILE
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_runs(rows):
    """{run_id: [trial rows sorted by trial number]}"""
    runs = defaultdict(list)
    for row in rows:
        runs[row["run_id"]].append(row)
    for trials in runs.values():
        trials.sort(key=lambda r: int(r["trial"]))
    return runs


def select_runs(runs, configs, session_id=None):
    """
    Choose one run per configuration key.

    With session_id: the run of each configuration from that session.
    Otherwise: the most recent run of each configuration.
    """
    chosen = {}
    for run_id, trials in runs.items():
        first = trials[0]
        key = first["config_id"]
        if key not in configs:
            continue
        if session_id is not None and first["session_id"] != session_id:
            continue
        current = chosen.get(key)
        if current is None or first["timestamp"] > current[0]["timestamp"]:
            chosen[key] = trials
    return chosen


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def summarize_run(trials):
    times = [float(r["total_time_s"]) for r in trials]
    first = trials[0]
    return {
        "run_id": first["run_id"],
        "session_id": first["session_id"],
        "timestamp": first["timestamp"],
        "trials": len(trials),
        "num_images": int(first["num_images"]),
        "warmup_images": first["warmup_images"],
        "model_sha256": first["model_sha256"],
        "image_list_sha256": first["image_list_sha256"],
        "dataset": first.get("dataset", ""),
        "model": first.get("model", ""),
        "device": first["device"],
        "mean_total_time_s": statistics.mean(times),
        "stdev_total_time_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_total_time_s": min(times),
        "max_total_time_s": max(times),
        "mean_avg_time_ms_per_image": statistics.mean(
            float(r["avg_time_ms_per_image"]) for r in trials),
        "mean_throughput_img_per_s": statistics.mean(
            float(r["throughput_img_per_s"]) for r in trials),
    }


def speedup(baseline_time_s, comparison_time_s):
    """Speedup = Baseline execution time / Comparison execution time."""
    if baseline_time_s <= 0 or comparison_time_s <= 0:
        raise ValueError("Execution times must be positive.")
    return baseline_time_s / comparison_time_s


def interpret(s, base_name, comp_name, kind):
    """Plain-language meaning of a speedup value."""
    if abs(s - 1.0) < SAME_PERFORMANCE_TOLERANCE:
        result = f"{comp_name} and {base_name} performed about the same (speedup ~1.0)."
    elif s > 1.0:
        result = (f"{comp_name} was {s:.2f}x faster than {base_name} "
                  f"(it needed {100 / s:.0f}% of the time).")
    else:
        result = (f"Speedup below 1.0: {comp_name} was SLOWER than {base_name}; "
                  f"it took {1 / s:.2f}x as long.")
    if kind == "resolution":
        note = ("Resolution effect: the input changes (about 4x fewer pixels at 320), "
                "so this is NOT a same-workload hardware speedup.")
    else:
        note = "Same workload on different hardware: a same-workload CPU/GPU speedup."
    return f"{result} {note}"


def check_consistency(summaries):
    """Return warnings if selected runs did not do identical work."""
    warnings = []
    sessions = {k: s["session_id"] for k, s in summaries.items()}
    if len(set(sessions.values())) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(sessions.items()))
        warnings.append(f"Runs come from different benchmark sessions ({detail}); conditions "
                        "may differ. Use --session <id> to compare runs from one session.")
    for field, label in (("num_images", "number of images"), ("dataset", "dataset"),
                         ("model", "model file"), ("model_sha256", "model weights"),
                         ("image_list_sha256", "image list/order"),
                         ("warmup_images", "warm-up")):
        values = {k: s[field] for k, s in summaries.items()}
        known = {v for v in values.values() if v not in ("", None)}
        if len(known) > 1:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(values.items()))
            warnings.append(f"Runs differ in {label}: {detail}")
        missing = sorted(k for k, v in values.items() if v in ("", None))
        if missing and field in ("model_sha256", "image_list_sha256", "warmup_images"):
            warnings.append(f"{label} not recorded for {', '.join(missing)} "
                            "(Step 4 run made before Step 5; methodology may differ).")
    return warnings


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", newline="") as f:     # never overwrite an earlier summary
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def fmt(value, spec):
    return format(value, spec) if isinstance(value, (int, float)) else "—"


def display_path(path):
    path = Path(path)
    return path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Summarize benchmark runs and compute speedups.")
    p.add_argument("--session", default=None,
                   help="Only use runs from this session ID (printed by run_all_benchmarks)")
    p.add_argument("--config-file", type=Path, default=CONFIG_PATH)
    p.add_argument("--no-save", action="store_true", help="Print only; do not write CSVs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        configs = load_all_configs(args.config_file)
        comparisons = load_comparisons(args.config_file)
    except ConfigError as e:
        print(f"ERROR: {e}")
        return 1
    any_cfg = next(iter(configs.values()))
    raw_dir, summary_dir = any_cfg.raw_results_dir, any_cfg.summary_dir

    runs = group_runs(read_raw_rows(raw_dir, configs))
    chosen = select_runs(runs, configs, args.session)
    summaries = {k: summarize_run(t) for k, t in chosen.items()}

    failures = read_failures(raw_dir)
    if args.session is not None:
        failures = [f for f in failures if f["session_id"] == args.session]
    last_error = {f["config_id"]: f["error"] for f in failures}

    # --- Summary table ---
    line = "=" * 92
    scope = f"session {args.session}" if args.session else "latest run of each configuration"
    print(line)
    print(f"ECE 4300 — Benchmark Summary ({scope})")
    print("Metric: elapsed workload processing time "
          "(preprocess + inference + postprocess), mean of measured trials")
    print(line)
    print(f"{'Cfg':<4}{'Name':<10}{'Device':<11}{'Res.':<9}{'Trials':>6}{'Images':>7}"
          f"{'Mean time (s)':>15}{'Std (s)':>9}{'ms/image':>10}{'images/s':>10}")
    summary_rows = []
    for key, cfg in configs.items():
        s = summaries.get(key)
        if s is None:
            status = "FAILED" if key in last_error else "MISSING"
            print(f"{key:<4}{cfg.name:<10}{cfg.device_label:<11}{cfg.resolution:<9}"
                  f"  {status}: {last_error.get(key, 'no successful run found')}")
            summary_rows.append({"config_id": key, "configuration": cfg.name,
                                 "device": cfg.device, "resolution": cfg.resolution,
                                 "status": status, "error": last_error.get(key, "")})
            continue
        status = "OK" if s["trials"] == cfg.num_trials else f"INCOMPLETE ({s['trials']} trials)"
        print(f"{key:<4}{cfg.name:<10}{cfg.device_label:<11}{cfg.resolution:<9}"
              f"{s['trials']:>6}{s['num_images']:>7}{s['mean_total_time_s']:>15.3f}"
              f"{s['stdev_total_time_s']:>9.3f}{s['mean_avg_time_ms_per_image']:>10.2f}"
              f"{s['mean_throughput_img_per_s']:>10.2f}")
        summary_rows.append({
            "config_id": key, "configuration": cfg.name, "device": s["device"],
            "resolution": cfg.resolution, "status": status, "error": "",
            **{k: s[k] for k in ("run_id", "session_id", "timestamp", "trials",
                                 "num_images", "warmup_images")},
            **{k: f"{s[k]:.6f}" for k in ("mean_total_time_s", "stdev_total_time_s",
                                          "min_total_time_s", "max_total_time_s")},
            "mean_avg_time_ms_per_image": f"{s['mean_avg_time_ms_per_image']:.4f}",
            "mean_throughput_img_per_s": f"{s['mean_throughput_img_per_s']:.4f}",
        })

    # --- Speedups ---
    print()
    print("Speedup = Baseline execution time / Comparison execution time")
    print("-" * 92)
    speedup_rows = []
    for comp in comparisons:
        base_cfg, comp_cfg = configs[comp["baseline"]], configs[comp["comparison"]]
        b, c = summaries.get(base_cfg.key), summaries.get(comp_cfg.key)
        label = f"{comp['title']}: {base_cfg.name} / {comp_cfg.name}"
        row = {"comparison": comp["title"], "kind": comp["kind"],
               "baseline": base_cfg.name, "comparison_config": comp_cfg.name}
        if b is None or c is None:
            missing = ", ".join(x.name for x, s in ((base_cfg, b), (comp_cfg, c)) if s is None)
            print(f"{label}\n    not calculated: no successful run for {missing}")
            row["interpretation"] = f"Not calculated: no successful run for {missing}"
        else:
            s = speedup(b["mean_total_time_s"], c["mean_total_time_s"])
            text = interpret(s, base_cfg.name, comp_cfg.name, comp["kind"])
            print(f"{label}\n    {b['mean_total_time_s']:.3f} s / "
                  f"{c['mean_total_time_s']:.3f} s = {s:.3f}\n    {text}")
            row.update(baseline_mean_time_s=f"{b['mean_total_time_s']:.6f}",
                       comparison_mean_time_s=f"{c['mean_total_time_s']:.6f}",
                       speedup=f"{s:.4f}", interpretation=text)
        speedup_rows.append(row)

    # --- Consistency checks ---
    warnings = check_consistency(summaries)
    print()
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    elif summaries:
        print("Check: all selected runs used the same model weights, image list, "
              "image count, and warm-up.")

    if not args.no_save and summaries:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"{stamp}_{args.session}" if args.session else stamp
        p1 = write_csv(summary_dir / f"summary_{suffix}.csv", SUMMARY_COLUMNS, summary_rows)
        p2 = write_csv(summary_dir / f"speedup_{suffix}.csv", SPEEDUP_COLUMNS, speedup_rows)
        print(f"Summary saved to:  {display_path(p1)}")
        print(f"Speedups saved to: {display_path(p2)}")
    print(line)
    return 0 if len(summaries) == len(configs) else 1


if __name__ == "__main__":
    sys.exit(main())
