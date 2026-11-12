"""
ECE 4300 - Step 6: Performance analysis and graph generation.

Reads the measured per-trial CSVs from Step 5 (results/raw/), checks that
they are complete and comparable, and produces:

    results/summaries/performance_summary_<session>.csv   per-configuration table
    results/summaries/speedup_<session>.csv               the four speedups
    results/summaries/analysis_report_<session>.md        short written interpretation
    results/graphs/execution_time_<session>.png           Graph 1
    results/graphs/throughput_<session>.png               Graph 2
    results/graphs/cpu_gpu_speedup_<session>.png          Graph 3

Only measured data is used. All four configurations must come from ONE
benchmark session (one run of src.run_all_benchmarks), so every number
was collected with the same code, settings, and conditions. If no such
session exists, nothing is plotted: the script reports what is missing and
the command that collects it.

Formulas (ECE 4300 Lecture 1):
    Mean execution time (s)        = sum of trial times / number of trials
    Avg. processing time (ms/img)  = mean execution time / number of images x 1000
    Trial throughput (images/s)    = number of images / trial execution time
    Mean throughput (images/s)     = average of the trial throughputs
    Range (s)                      = maximum trial time - minimum trial time
    Speedup                        = reference execution time / comparison execution time

Run from the project root:
    python -m src.analyze_results                   # latest complete session
    python -m src.analyze_results --session <id>    # a specific session
"""

import argparse
import csv
import math
import sys
from pathlib import Path

from src.benchmark_config import (
    CONFIG_PATH, PROJECT_ROOT, ConfigError, load_all_configs, load_comparisons,
    read_config_file,
)
from src.compare_results import group_runs, read_failures, read_raw_rows

EXPECTED_MEASUREMENT = "elapsed_workload_time"
STORED_VALUE_TOLERANCE_MS = 0.01      # stored ms/image must match the recomputed value

# Colors: validated categorical pair (blue = CPU, orange = GPU); a neutral gray
# for the speedup ratio, which belongs to neither device.
DEVICE_COLORS = {"cpu": "#2a78d6", "mps": "#eb6834"}
RATIO_COLOR = "#6b7280"
INK = "#1f2328"
MUTED_INK = "#57606a"
GRID = "#d8dee4"

SUMMARY_COLUMNS = [
    "configuration", "config_id", "processor", "resolution", "trials",
    "images_per_trial", "mean_execution_time_s", "avg_processing_time_ms_per_image",
    "mean_throughput_img_per_s", "min_execution_time_s", "max_execution_time_s",
    "execution_time_range_s", "run_id", "session_id",
]
SPEEDUP_COLUMNS = [
    "comparison", "kind", "reference", "comparison_config", "reference_mean_time_s",
    "comparison_mean_time_s", "speedup", "faster_configuration", "interpretation",
]


# ----------------------------------------------------------------------------
# Formulas
# ----------------------------------------------------------------------------
def _check_positive(*values):
    for v in values:
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise ValueError(f"Expected a positive number, got {v!r}")


def mean_execution_time(times):
    """Mean Execution Time = Sum of Trial Execution Times / Number of Trials."""
    if not times:
        raise ValueError("At least one trial is required.")
    _check_positive(*times)
    return sum(times) / len(times)


def avg_processing_time_ms(mean_time_s, num_images):
    """Average Processing Time = Mean Execution Time / Number of Images (in ms)."""
    _check_positive(mean_time_s, num_images)
    return mean_time_s / num_images * 1000.0


def trial_throughput(time_s, num_images):
    """Throughput = Number of Images / Execution Time (images/s), for one trial."""
    _check_positive(time_s, num_images)
    return num_images / time_s


def mean_throughput(times, num_images):
    """Throughput of each trial, then their average (images/s)."""
    values = [trial_throughput(t, num_images) for t in times]
    return sum(values) / len(values)


def time_range(times):
    """(minimum, maximum, range = maximum - minimum) of the trial times."""
    _check_positive(*times)
    return min(times), max(times), max(times) - min(times)


def speedup(reference_time_s, comparison_time_s):
    """Speedup = Reference Execution Time / Comparison Execution Time."""
    _check_positive(reference_time_s, comparison_time_s)
    return reference_time_s / comparison_time_s


def describe_speedup(s, reference, comparison):
    """Direction-aware wording: never calls a slowdown an improvement."""
    if math.isclose(s, 1.0, abs_tol=0.005):
        return (f"Speedup {s:.2f}: {comparison} and {reference} took about the same time.",
                "neither")
    if s > 1.0:
        return (f"Speedup {s:.2f} > 1: {comparison} was faster than {reference} "
                f"({s:.2f}x the performance).", comparison)
    return (f"Speedup {s:.2f} < 1: {comparison} was SLOWER than {reference} "
            f"(it took {1 / s:.2f}x as long).", reference)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate_run(trials, cfg):
    """
    Return a list of problems with one configuration's run (empty = valid).

    A run with any invalid trial is rejected as a whole; trials are never
    dropped silently.
    """
    problems = []
    if len(trials) < cfg.num_trials:
        problems.append(f"only {len(trials)} trials (need {cfg.num_trials})")
    numbers = [r.get("trial", "") for r in trials]
    if len(set(numbers)) != len(numbers):
        problems.append("duplicate trial numbers")

    for r in trials:
        label = f"trial {r.get('trial', '?')}"
        try:
            t = float(r["total_time_s"])
            n = int(r["num_images"])
        except (KeyError, ValueError):
            problems.append(f"{label}: missing or non-numeric time / image count")
            continue
        if not (math.isfinite(t) and t > 0):
            problems.append(f"{label}: execution time {r['total_time_s']} is not positive")
            continue
        if n != cfg.expected_images:
            problems.append(f"{label}: {n} images (expected {cfg.expected_images})")
        try:
            stored_ms = float(r["avg_time_ms_per_image"])
            if abs(stored_ms - t / n * 1000) > STORED_VALUE_TOLERANCE_MS:
                problems.append(f"{label}: stored ms/image {stored_ms} does not match its time")
        except (KeyError, ValueError, ZeroDivisionError):
            problems.append(f"{label}: missing ms/image value")
        if r.get("measurement") != EXPECTED_MEASUREMENT:
            problems.append(f"{label}: measurement is {r.get('measurement')!r}, "
                            f"not {EXPECTED_MEASUREMENT!r}")

    first = trials[0] if trials else {}
    if first.get("device") != cfg.device:
        problems.append(f"device is {first.get('device')!r}, expected {cfg.device!r}")
    if first.get("resolution") != cfg.resolution:
        problems.append(f"resolution is {first.get('resolution')!r}, expected {cfg.resolution!r}")
    return problems


def check_compatible(runs):
    """
    Return problems if the selected runs did not do identical work.

    `runs` maps config key -> trial rows. Every run must come from one
    session and record the same weights, image list, dataset, and warm-up.
    """
    problems = []
    fields = {
        "session_id": "benchmark session",
        "model": "model file",
        "model_sha256": "model weights (hash)",
        "image_list_sha256": "image list / order (hash)",
        "dataset": "dataset",
        "num_images": "number of images",
        "warmup_images": "warm-up",
        "cpu_threads": "CPU thread count",
    }
    for field, label in fields.items():
        values = {k: trials[0].get(field, "") for k, trials in runs.items()}
        if any(v in ("", None) for v in values.values()):
            missing = ", ".join(sorted(k for k, v in values.items() if v in ("", None)))
            problems.append(f"{label} not recorded for {missing} "
                            "(run predates Step 5, so it cannot be confirmed comparable)")
        elif len(set(values.values())) > 1:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(values.items()))
            problems.append(f"runs differ in {label}: {detail}")
    return problems


def find_sessions(runs_by_id, configs):
    """{session_id: {config_key: trials}} for every session in the raw data."""
    sessions = {}
    for trials in runs_by_id.values():
        first = trials[0]
        key = first["config_id"]
        if key not in configs:
            continue
        slot = sessions.setdefault(first["session_id"], {})
        if key not in slot or first["timestamp"] > slot[key][0]["timestamp"]:
            slot[key] = trials
    return sessions


def select_session(sessions, configs, session_id=None):
    """
    Return (session_id, runs) for the requested session, or the most recent
    session containing all configurations. (None, None) if there is none.
    """
    if session_id is not None:
        runs = sessions.get(session_id)
        return (session_id, runs) if runs else (None, None)
    complete = {sid: runs for sid, runs in sessions.items() if set(runs) == set(configs)}
    if not complete:
        return None, None
    latest = max(complete, key=lambda sid: max(t[0]["timestamp"] for t in complete[sid].values()))
    return latest, complete[latest]


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def analyze_run(trials, cfg):
    times = [float(r["total_time_s"]) for r in trials]
    n = int(trials[0]["num_images"])
    mean_t = mean_execution_time(times)
    t_min, t_max, t_range = time_range(times)
    throughputs = [trial_throughput(t, n) for t in times]
    return {
        "cfg": cfg,
        "label": graph_label(cfg),
        "processor": trials[0].get("processor", ""),
        "run_id": trials[0]["run_id"],
        "session_id": trials[0]["session_id"],
        "timestamp": trials[0]["timestamp"],
        "trials": len(times),
        "num_images": n,
        "times": times,
        "mean_time_s": mean_t,
        "avg_ms_per_image": avg_processing_time_ms(mean_t, n),
        "mean_throughput": mean_throughput(times, n),
        "min_throughput": min(throughputs),
        "max_throughput": max(throughputs),
        "min_time_s": t_min,
        "max_time_s": t_max,
        "range_s": t_range,
        "range_pct": t_range / mean_t * 100,
    }


def graph_label(cfg):
    """'CPU 640×640' / 'GPU 320×320' (consistent names on every graph)."""
    device = "GPU" if cfg.device == "mps" else "CPU"
    return f"{device} {cfg.image_size}×{cfg.image_size}"


def compute_speedups(results, configs, comparisons):
    rows = []
    for comp in comparisons:
        ref, cmp_ = results[comp["baseline"]], results[comp["comparison"]]
        s = speedup(ref["mean_time_s"], cmp_["mean_time_s"])
        text, faster = describe_speedup(s, ref["cfg"].name, cmp_["cfg"].name)
        rows.append({
            "comparison": comp["title"], "kind": comp["kind"],
            "reference": ref["cfg"].name, "comparison_config": cmp_["cfg"].name,
            "reference_mean_time_s": ref["mean_time_s"],
            "comparison_mean_time_s": cmp_["mean_time_s"],
            "speedup": s, "faster_configuration": faster, "interpretation": text,
        })
    return rows


# ----------------------------------------------------------------------------
# Output: CSV
# ----------------------------------------------------------------------------
def write_summary_csv(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for key, r in results.items():
            cfg = r["cfg"]
            w.writerow({
                "configuration": cfg.name, "config_id": key, "processor": r["processor"],
                "resolution": cfg.resolution, "trials": r["trials"],
                "images_per_trial": r["num_images"],
                "mean_execution_time_s": f"{r['mean_time_s']:.6f}",
                "avg_processing_time_ms_per_image": f"{r['avg_ms_per_image']:.4f}",
                "mean_throughput_img_per_s": f"{r['mean_throughput']:.4f}",
                "min_execution_time_s": f"{r['min_time_s']:.6f}",
                "max_execution_time_s": f"{r['max_time_s']:.6f}",
                "execution_time_range_s": f"{r['range_s']:.6f}",
                "run_id": r["run_id"], "session_id": r["session_id"],
            })
    return path


def write_speedup_csv(speedups, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SPEEDUP_COLUMNS)
        w.writeheader()
        for s in speedups:
            w.writerow({**s,
                        "reference_mean_time_s": f"{s['reference_mean_time_s']:.6f}",
                        "comparison_mean_time_s": f"{s['comparison_mean_time_s']:.6f}",
                        "speedup": f"{s['speedup']:.4f}"})
    return path


# ----------------------------------------------------------------------------
# Output: graphs
# ----------------------------------------------------------------------------
def _new_axes(plt):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_INK)
    ax.tick_params(colors=INK, labelsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, title, ylabel, note, path, dpi):
    ax.set_title(title, fontsize=14, color=INK, pad=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, color=INK)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8.5, color=MUTED_INK, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    return path


def _device_legend(ax, results):
    from matplotlib.patches import Patch
    devices = {r["cfg"].device for r in results.values()}
    handles = [Patch(color=DEVICE_COLORS[d], label="CPU" if d == "cpu" else "GPU (MPS)")
               for d in ("cpu", "mps") if d in devices]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="best")


def _bar_chart(plt, results, values, lows, highs, title, ylabel, value_fmt, note, path, dpi):
    fig, ax = _new_axes(plt)
    keys = list(results)
    x = range(len(keys))
    colors = [DEVICE_COLORS[results[k]["cfg"].device] for k in keys]
    ax.bar(x, values, width=0.55, color=colors, edgecolor="white", linewidth=2, zorder=2)
    yerr = [[v - lo for v, lo in zip(values, lows)], [hi - v for v, hi in zip(values, highs)]]
    ax.errorbar(x, values, yerr=yerr, fmt="none", ecolor=INK, elinewidth=1, capsize=4, zorder=3)
    top = max(highs)
    for xi, v, hi in zip(x, values, highs):
        ax.text(xi, hi + top * 0.02, value_fmt(v), ha="center", va="bottom",
                fontsize=10, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([results[k]["label"] for k in keys])
    ax.set_ylim(0, top * 1.18)
    _device_legend(ax, results)
    return _finish(fig, ax, title, ylabel, note, path, dpi)


def plot_execution_time(plt, results, path, dpi, trials_note):
    return _bar_chart(
        plt, results,
        [r["mean_time_s"] for r in results.values()],
        [r["min_time_s"] for r in results.values()],
        [r["max_time_s"] for r in results.values()],
        "Average Object Detection Execution Time",
        "Average execution time (seconds)",
        lambda v: f"{v:.3f} s",
        f"Bars: mean of {trials_note} (100 images each); whiskers: minimum–maximum trial. "
        "Lower is faster.",
        path, dpi)


def plot_throughput(plt, results, path, dpi, trials_note):
    return _bar_chart(
        plt, results,
        [r["mean_throughput"] for r in results.values()],
        [r["min_throughput"] for r in results.values()],
        [r["max_throughput"] for r in results.values()],
        "Object Detection Throughput Comparison",
        "Throughput (images processed per second)",
        lambda v: f"{v:.1f}",
        f"Bars: mean trial throughput over {trials_note}; whiskers: minimum–maximum trial. "
        "Higher is better.",
        path, dpi)


def plot_cpu_gpu_speedup(plt, cpu_gpu, path, dpi):
    """cpu_gpu: [(resolution label, speedup)] in display order (320, then 640)."""
    fig, ax = _new_axes(plt)
    labels = [lbl for lbl, _ in cpu_gpu]
    values = [s for _, s in cpu_gpu]
    x = range(len(values))
    ax.bar(x, values, width=0.45, color=RATIO_COLOR, edgecolor="white", linewidth=2, zorder=2)
    top = max(max(values), 1.0)
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.2, zorder=3,
               label="1.0× = equal performance")
    ax.legend(frameon=False, fontsize=10, loc="best")
    for xi, v in zip(x, values):
        ax.text(xi, v + top * 0.02, f"{v:.2f}×", ha="center", va="bottom",
                fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Input resolution", fontsize=12, color=INK)
    ax.set_ylim(0, top * 1.22)
    return _finish(fig, ax, "CPU-to-GPU Speedup by Resolution", "Speedup factor (×)",
                   "Speedup = CPU mean execution time / GPU mean execution time. "
                   "Above 1.0×: GPU faster. Below 1.0×: CPU faster.",
                   path, dpi)


def make_graphs(results, cpu_gpu, graphs_dir, session_id, dpi):
    import matplotlib
    matplotlib.use("Agg")                     # file output only, no window
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK})
    trials = {r["trials"] for r in results.values()}
    trials_note = f"{trials.pop()} trials" if len(trials) == 1 else "the measured trials"
    paths = [
        plot_execution_time(plt, results, graphs_dir / f"execution_time_{session_id}.png",
                            dpi, trials_note),
        plot_throughput(plt, results, graphs_dir / f"throughput_{session_id}.png",
                        dpi, trials_note),
        plot_cpu_gpu_speedup(plt, cpu_gpu, graphs_dir / f"cpu_gpu_speedup_{session_id}.png", dpi),
    ]
    plt.close("all")
    return paths


# ----------------------------------------------------------------------------
# Output: written interpretation
# ----------------------------------------------------------------------------
def build_report(results, speedups, session_id, threshold_pct):
    """Markdown report; every number comes from `results` / `speedups`."""
    by_name = {r["cfg"].name: r for r in results.values()}
    fastest = min(results.values(), key=lambda r: r["mean_time_s"])
    best_tp = max(results.values(), key=lambda r: r["mean_throughput"])
    sp = {s["comparison"]: s for s in speedups}
    res_rows = [s for s in speedups if s["kind"] == "resolution"]
    dev_rows = [s for s in speedups if s["kind"] == "device"]

    L = []
    L.append(f"# Step 6 Performance Analysis — session `{session_id}`")
    L.append("")
    procs = sorted({r["processor"] for r in results.values()})
    first = min(r["timestamp"] for r in results.values())
    L.append(f"Measured on {first[:10]} ({'; '.join(procs)}). Each configuration processed "
             f"the same {fastest['num_images']} COCO 2017 images per trial with YOLO11n, "
             "batch size 1. Times are elapsed workload processing time "
             "(preprocessing + inference + postprocessing).")
    L.append("")
    L.append("## Measured results")
    L.append("")
    L.append("| Configuration | Trials | Mean time (s) | ms/image | Mean throughput (images/s) "
             "| Min (s) | Max (s) | Range (s) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results.values():
        L.append(f"| {r['label']} | {r['trials']} | {r['mean_time_s']:.3f} | "
                 f"{r['avg_ms_per_image']:.2f} | {r['mean_throughput']:.2f} | "
                 f"{r['min_time_s']:.3f} | {r['max_time_s']:.3f} | {r['range_s']:.3f} |")
    L.append("")
    L.append("## Speedups (reference time / comparison time)")
    L.append("")
    L.append("| Comparison | Reference → Comparison | Speedup | Result |")
    L.append("|---|---|---|---|")
    for s in speedups:
        L.append(f"| {s['comparison']} | {s['reference']} → {s['comparison_config']} | "
                 f"{s['speedup']:.2f} | {s['interpretation']} |")
    L.append("")

    L.append("## Observations (measured)")
    L.append("")
    L.append(f"- **Shortest mean execution time:** {fastest['label']}, "
             f"{fastest['mean_time_s']:.3f} s for {fastest['num_images']} images "
             f"({fastest['avg_ms_per_image']:.2f} ms/image).")
    L.append(f"- **Highest throughput:** {best_tp['label']}, "
             f"{best_tp['mean_throughput']:.2f} images/s.")
    for s in res_rows:
        ref, cmp_ = by_name[s["reference"]], by_name[s["comparison_config"]]
        change = (cmp_["mean_time_s"] - ref["mean_time_s"]) / ref["mean_time_s"] * 100
        L.append(f"- **{s['comparison']}:** {ref['label']} → {cmp_['label']} changed the mean "
                 f"time from {ref['mean_time_s']:.3f} s to {cmp_['mean_time_s']:.3f} s "
                 f"({change:+.1f}%, speedup {s['speedup']:.2f}). The 320×320 input has "
                 "one quarter of the pixels, so this is a change in workload, not in hardware.")
    for s in dev_rows:
        L.append(f"- **{s['comparison']}:** {s['interpretation']} "
                 "(same workload, different processor)")
    spread = ", ".join(f"{r['label']} {r['range_pct']:.1f}%" for r in results.values())
    inconsistent = [r for r in results.values() if r["range_pct"] > threshold_pct]
    verdict = ("All configurations were consistent across trials"
               if not inconsistent else
               "Not all configurations were consistent across trials")
    L.append(f"- **Consistency:** {verdict} (range as a percentage of the mean: {spread}; "
             f"threshold used: {threshold_pct}%).")
    L.append("")

    L.append("## Unexpected results")
    L.append("")
    unexpected = []
    for s in dev_rows:
        if s["speedup"] < 1.0:
            unexpected.append(f"{s['comparison']}: the GPU was slower than the CPU "
                              f"(speedup {s['speedup']:.2f}).")
    for s in res_rows:
        if s["speedup"] <= 1.0:
            unexpected.append(f"{s['comparison']}: the lower resolution was not faster "
                              f"(speedup {s['speedup']:.2f}).")
    if len(dev_rows) == 2 and (dev_rows[0]["speedup"] > 1) != (dev_rows[1]["speedup"] > 1):
        unexpected.append("The faster processor was different at the two resolutions.")
    for r in inconsistent:
        unexpected.append(f"{r['label']}: trial times varied by {r['range_pct']:.1f}% of the "
                          f"mean ({r['min_time_s']:.3f}–{r['max_time_s']:.3f} s).")
    for r in results.values():
        others = r["times"][1:]
        if others and r["times"][0] > max(others) and \
                (r["times"][0] - max(others)) / r["mean_time_s"] * 100 > threshold_pct:
            unexpected.append(f"{r['label']}: trial 1 ({r['times'][0]:.3f} s) was much slower "
                              f"than the other trials (≤ {max(others):.3f} s).")
    if unexpected:
        L.extend(f"- {u}" for u in unexpected)
    else:
        L.append("- None of the checks flagged an unexpected result.")
    L.append("")

    L.append("## Possible explanations (hypotheses — not measured)")
    L.append("")
    L.append("These are possible reasons only. This project measured total elapsed time; "
             "it did not measure the individual parts below.")
    L.append("")
    hyps = []
    gpu_res = sp.get("Resolution effect on GPU")
    cpu_res = sp.get("Resolution effect on CPU")
    if any(s["speedup"] < 1.0 for s in dev_rows) or (
            gpu_res and cpu_res and gpu_res["speedup"] < cpu_res["speedup"]):
        hyps.append("With batch size 1 and a small model, each image carries fixed costs that "
                    "do not shrink with a faster processor or a smaller input (CPU-side "
                    "preprocessing and postprocessing, launching GPU work, and waiting for it "
                    "to finish). If these are a large share of each image's time, the GPU's "
                    "advantage and the benefit of a lower resolution would both be reduced.")
    if len(dev_rows) == 2:
        s320 = sp.get("CPU vs. GPU at 320x320")
        s640 = sp.get("CPU vs. GPU at 640x640")
        if s320 and s640 and s640["speedup"] > s320["speedup"]:
            hyps.append("The CPU-to-GPU speedup was larger at 640×640. A larger input gives "
                        "the GPU more work per image to run in parallel, which may make better "
                        "use of it relative to the fixed per-image costs.")
    if inconsistent or unexpected:
        hyps.append("Trial-to-trial variation may come from macOS background activity, power "
                    "or temperature changes, or other apps sharing the unified memory.")
    if hyps:
        L.extend(f"- {h}" for h in hyps)
    else:
        L.append("- No hypotheses needed.")
    L.append("")
    L.append("## Experimental conditions")
    L.append("")
    L.append("These results apply to this Mac, software versions, and conditions (power "
             "source, temperature, background apps). Other hardware, other model sizes, or "
             "other batch sizes may give different results.")
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def display_path(path):
    path = Path(path)
    return path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path


def report_missing(configs, sessions, failures, session_id, raw_dir):
    print("ERROR: No complete, valid benchmark session is available for analysis.")
    print(f"Searched: {display_path(raw_dir)}")
    if session_id is not None:
        print(f"Session {session_id!r} was not found.")
    if not sessions:
        print("No Step 5 benchmark results were found.")
    else:
        print("Sessions found (configurations with measurements):")
        for sid, runs in sorted(sessions.items()):
            have = ", ".join(configs[k].name for k in configs if k in runs) or "none"
            print(f"  {sid}: {have}")
    if failures:
        print("Recorded failures (results/raw/failed_runs.csv):")
        for f in failures[-4:]:
            print(f"  {f['session_id']} {f['configuration']}: {f['error']}")
    print()
    print("Collect all four configurations in one session (on the Apple M2 Mac):")
    print("  source .venv/bin/activate")
    print("  python -m src.run_all_benchmarks")
    print("Then run the analysis again:")
    print("  python -m src.analyze_results")
    print("No summary, report, or graphs were written.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Analyze Step 5 measurements and draw graphs.")
    p.add_argument("--session", default=None,
                   help="Session ID to analyze (default: latest session with all four configurations)")
    p.add_argument("--config-file", type=Path, default=CONFIG_PATH)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        configs = load_all_configs(args.config_file)
        comparisons = load_comparisons(args.config_file)
        settings = read_config_file(args.config_file).get("analysis", {})
    except ConfigError as e:
        print(f"ERROR: {e}")
        return 1
    any_cfg = next(iter(configs.values()))
    raw_dir = any_cfg.raw_results_dir
    summaries_dir = PROJECT_ROOT / settings.get("summaries_dir", "results/summaries")
    graphs_dir = PROJECT_ROOT / settings.get("graphs_dir", "results/graphs")
    dpi = int(settings.get("graph_dpi", 300))
    threshold = float(settings.get("consistency_threshold_pct", 10))

    # --- Task 1: locate and verify the measurements ---
    runs_by_id = group_runs(read_raw_rows(raw_dir, configs))
    sessions = find_sessions(runs_by_id, configs)
    failures = read_failures(raw_dir)
    session_id, runs = select_session(sessions, configs, args.session)
    if runs is None:
        report_missing(configs, sessions, failures, args.session, raw_dir)
        return 1

    problems = []
    missing = [configs[k].name for k in configs if k not in runs]
    if missing:
        problems.append(f"session has no measurements for {', '.join(missing)}")
    for key, trials in runs.items():
        problems += [f"{configs[key].name}: {p}" for p in validate_run(trials, configs[key])]
    if any(c.batch_size != 1 for c in configs.values()):
        problems.append("batch size is not 1 in the configuration file")
    problems += check_compatible(runs)
    if problems:
        print(f"ERROR: Session {session_id} cannot be analyzed:")
        for p in problems:
            print(f"  - {p}")
        print("\nNothing was written. Re-collect all four configurations in one session:")
        print("  python -m src.run_all_benchmarks\n  python -m src.analyze_results")
        return 1

    print(f"Verified session {session_id}: 4 configurations, "
          f"{', '.join(str(len(runs[k])) for k in configs)} trials, "
          f"{runs[next(iter(configs))][0]['num_images']} images per trial, same model weights, "
          "image list, warm-up, and timing boundary (elapsed workload time); batch size 1.")

    # --- Task 2: calculations ---
    results = {k: analyze_run(runs[k], configs[k]) for k in configs}
    speedups = compute_speedups(results, configs, comparisons)
    by_title = {s["comparison"]: s for s in speedups}
    cpu_gpu = [("320×320", by_title["CPU vs. GPU at 320x320"]["speedup"]),
               ("640×640", by_title["CPU vs. GPU at 640x640"]["speedup"])]

    # --- Tasks 3-6: outputs ---
    summary_path = write_summary_csv(results, summaries_dir / f"performance_summary_{session_id}.csv")
    speedup_path = write_speedup_csv(speedups, summaries_dir / f"speedup_{session_id}.csv")
    report_path = summaries_dir / f"analysis_report_{session_id}.md"
    report_path.write_text(build_report(results, speedups, session_id, threshold), encoding="utf-8")
    graph_paths = make_graphs(results, cpu_gpu, graphs_dir, session_id, dpi)

    print()
    print(f"{'Configuration':<14}{'Mean (s)':>10}{'ms/image':>10}{'images/s':>10}"
          f"{'Min (s)':>9}{'Max (s)':>9}{'Range (s)':>11}")
    for r in results.values():
        print(f"{r['label']:<14}{r['mean_time_s']:>10.3f}{r['avg_ms_per_image']:>10.2f}"
              f"{r['mean_throughput']:>10.2f}{r['min_time_s']:>9.3f}{r['max_time_s']:>9.3f}"
              f"{r['range_s']:>11.3f}")
    print()
    for s in speedups:
        print(f"{s['comparison']}: {s['reference']} / {s['comparison_config']} = "
              f"{s['speedup']:.3f}  ->  {s['interpretation']}")
    print()
    for p in (summary_path, speedup_path, report_path, *graph_paths):
        print(f"Saved: {display_path(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
