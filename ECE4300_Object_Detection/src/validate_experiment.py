"""
ECE 4300 - Step 7: Validate the recorded benchmark measurements.

Reads the raw Step 5 trial CSVs (results/raw/) directly with the csv module,
recalculates every metric independently, and checks the Step 5 and Step 6
outputs against those recalculated values. Nothing is benchmarked here and
no existing result file is changed; the findings are printed and saved to a
NEW file:

    results/validation/data_validation_<session>_<time>.md

Checks:
  - one run per configuration A-D in the session, each with >= 5 trials
    numbered 1..N, positive execution times, and 100 images per trial
  - stored ms/image and images/s match each trial's measured time
  - device, resolution, timing boundary (measurement column), warm-up, CPU
    threads, model weights hash, and image-list hash are the same where they
    must be, and match the configuration file / current files on disk
  - the four speedups use the right numerator and denominator
  - Step 5 summary / speedup CSVs and Step 6 summary / speedup CSVs and the
    written report agree with the independent recalculation
  - the Step 6 graphs exist, are 300-dpi PNGs, and are byte-identical to a
    fresh render of the measured data (needs matplotlib; otherwise SKIP)
  - the model input shapes produced by YOLO's letterbox resize

Formulas (ECE 4300 Lecture 1):
    Mean execution time (s)       = sum of trial times / number of trials
    Avg. processing time (ms/img) = mean execution time / number of images x 1000
    Throughput (images/s)         = number of images / trial time (then averaged)
    Range (s)                     = maximum trial time - minimum trial time
    Speedup                       = reference mean time / comparison mean time

Run from the project root (on the Mac, after Steps 5 and 6):
    python -m src.validate_experiment                     # latest complete session
    python -m src.validate_experiment --session <id>
"""

import argparse
import csv
import hashlib
import math
import struct
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.benchmark_config import (
    CONFIG_PATH, PROJECT_ROOT, ConfigError, load_all_configs, load_comparisons,
    read_config_file,
)

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "SKIP", "INFO"
EXPECTED_MEASUREMENT = "elapsed_workload_time"
FAILURES_FILE = "failed_runs.csv"
STRIDE = 32                      # YOLO11n maximum stride (letterbox padding multiple)
STORED_TOL = 1e-3                # per-trial CSV values have 4 decimals
REQUIRED_COLUMNS = (
    "run_id", "session_id", "timestamp", "configuration", "config_id", "device",
    "processor", "resolution", "model", "model_sha256", "dataset", "image_list_sha256",
    "num_images", "cpu_threads", "warmup_images", "measurement", "trial",
    "total_time_s", "avg_time_ms_per_image", "throughput_img_per_s",
)


def close(a, b):
    """Values written to CSV with 4-6 decimals compared with recalculated ones."""
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-4)


class Checks:
    """Ordered list of (section, check, status, detail)."""

    def __init__(self):
        self.items = []

    def add(self, section, name, status, detail=""):
        self.items.append((section, name, status, detail))

    def expect(self, section, name, ok, detail_ok="", detail_fail=""):
        self.add(section, name, PASS if ok else FAIL, detail_ok if ok else detail_fail)
        return ok

    def count(self, status):
        return sum(1 for item in self.items if item[2] == status)


# ----------------------------------------------------------------------------
# Independent formulas (deliberately not imported from src.analyze_results)
# ----------------------------------------------------------------------------
def recalculate(times, num_images):
    n = len(times)
    mean_t = sum(times) / n
    throughputs = [num_images / t for t in times]
    return {
        "trials": n,
        "num_images": num_images,
        "mean_time_s": mean_t,
        "ms_per_image": mean_t / num_images * 1000.0,
        "mean_throughput": sum(throughputs) / n,
        "min_time_s": min(times),
        "max_time_s": max(times),
        "range_s": max(times) - min(times),
    }


def speedup_direction(s):
    if math.isclose(s, 1.0, abs_tol=0.005):
        return "same"
    return "comparison faster" if s > 1 else "comparison SLOWER"


def letterbox_shape(width, height, size, stride=STRIDE):
    """
    Model input (height, width) for one image, following Ultralytics' LetterBox
    with auto=True (minimum stride-multiple padding, used by predict()).
    """
    r = min(size / height, size / width)
    new_w, new_h = int(round(width * r)), int(round(height * r))
    return new_h + (size - new_h) % stride, new_w + (size - new_w) % stride


def image_list_fingerprint(manifest_rows):
    """Same definition as src.benchmark.image_list_fingerprint (stdlib only)."""
    text = "\n".join(f"{r['file_name']},{r.get('sha256', '')}" for r in manifest_rows)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def file_sha256_short(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def png_info(path):
    """(width, height, dpi or None) read from the PNG header; None if not a PNG."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", data[16:24])
    dpi, pos = None, 8
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if kind == b"pHYs":
            ppm_x, _, unit = struct.unpack(">IIB", data[pos + 8:pos + 17])
            if unit == 1:                          # pixels per metre
                dpi = round(ppm_x * 0.0254)
            break
        if kind == b"IDAT":
            break
        pos += 12 + length
    return width, height, dpi


# ----------------------------------------------------------------------------
# Reading raw data
# ----------------------------------------------------------------------------
def read_raw(raw_dir):
    """{run_id: [rows]} exactly as stored in results/raw/*.csv (nothing filled in)."""
    runs = {}
    for path in sorted(Path(raw_dir).glob("*.csv")):
        if path.name == FAILURES_FILE:
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_file"] = path.name
                runs.setdefault(row.get("run_id", ""), []).append(row)
    return runs


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_sessions(runs):
    """{session_id: {config_id: [run_id, ...]}}; Step 4 runs (no session) are skipped."""
    sessions = {}
    for run_id, rows in runs.items():
        sid = rows[0].get("session_id") or ""
        if sid:
            slot = sessions.setdefault(sid, {})
            slot.setdefault(rows[0].get("config_id", ""), []).append(run_id)
    return sessions


def pick_session(sessions, runs, keys, wanted=None):
    if wanted is not None:
        return wanted if wanted in sessions else None
    complete = [sid for sid, slot in sessions.items() if set(keys) <= set(slot)]
    if not complete:
        return None
    return max(complete, key=lambda sid: max(runs[r][0].get("timestamp", "")
                                             for ids in sessions[sid].values() for r in ids))


# ----------------------------------------------------------------------------
# Checks on the raw measurements
# ----------------------------------------------------------------------------
def check_run(checks, cfg, rows):
    """Validate one configuration's trial rows. Returns the trial times if usable."""
    sec = f"Raw trials: {cfg.name}"
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if not checks.expect(sec, "All Step 5 CSV columns present", not missing_cols,
                         f"{len(REQUIRED_COLUMNS)} columns ({rows[0]['_file']})",
                         f"missing {missing_cols} in {rows[0]['_file']}"):
        return None

    try:
        numbers = sorted(int(r["trial"]) for r in rows)
    except ValueError:
        numbers = []
    checks.expect(sec, f"At least {cfg.num_trials} trials", len(rows) >= cfg.num_trials,
                  f"{len(rows)} trials", f"only {len(rows)} trials")
    checks.expect(sec, "Trial numbers are 1..N with no duplicates or gaps",
                  numbers == list(range(1, len(rows) + 1)),
                  f"trials {numbers}", f"trial numbers found: {[r['trial'] for r in rows]}")

    times, bad_time, bad_n, bad_stored = [], [], [], []
    for r in rows:
        try:
            t = float(r["total_time_s"])
            n = int(r["num_images"])
        except ValueError:
            bad_time.append(r["trial"])
            continue
        if not (math.isfinite(t) and t > 0):
            bad_time.append(r["trial"])
            continue
        times.append(t)
        if n != cfg.expected_images:
            bad_n.append(f"trial {r['trial']}: {n}")
        try:
            if not (abs(float(r["avg_time_ms_per_image"]) - t / n * 1000) <= STORED_TOL and
                    abs(float(r["throughput_img_per_s"]) - n / t) <= STORED_TOL):
                bad_stored.append(r["trial"])
        except (ValueError, ZeroDivisionError):
            bad_stored.append(r["trial"])
    checks.expect(sec, "Every execution time is a positive number (seconds)", not bad_time,
                  f"{min(times):.3f}–{max(times):.3f} s" if times else "",
                  f"invalid time in trial(s) {bad_time}")
    checks.expect(sec, f"Every trial processed {cfg.expected_images} images", not bad_n,
                  f"{cfg.expected_images} images x {len(rows)} trials", "; ".join(bad_n))
    checks.expect(sec, "Stored ms/image and images/s match each trial's time", not bad_stored,
                  "all trials", f"mismatch in trial(s) {bad_stored}")

    def same(field, expected, label):
        values = sorted({r[field] for r in rows})
        checks.expect(sec, label, values == [str(expected)], f"{field} = {expected}",
                      f"{field} values {values}, expected {expected!r}")

    same("configuration", cfg.name, "Configuration name")
    same("device", cfg.device, "Device recorded")
    same("resolution", cfg.resolution, "Input resolution recorded")
    same("measurement", EXPECTED_MEASUREMENT, "Timing boundary (measurement column)")
    same("warmup_images", cfg.num_warmup, "Warm-up images (untimed)")
    same("cpu_threads", cfg.cpu_threads, "PyTorch CPU threads")
    same("dataset", cfg.dataset_id, "Dataset ID")
    same("model", cfg.model_path.name, "Model file")
    want = "GPU (MPS)" if cfg.device == "mps" else "CPU"
    procs = sorted({r["processor"] for r in rows})
    checks.expect(sec, "Processor label matches device",
                  all(p.endswith(want) for p in procs), procs[0] if procs else "",
                  f"processor {procs}, expected '... {want}'")
    for field in ("run_id", "session_id", "model_sha256", "image_list_sha256"):
        values = {r[field] for r in rows}
        checks.expect(sec, f"{field} identical for every trial", len(values) == 1 and "" not in values,
                      "", f"values {sorted(values)}")
    if bad_time or not times:
        return None
    return times


def check_session_consistency(checks, configs, chosen, manifest_rows, model_path):
    sec = "Same workload in every configuration"
    first = {k: rows[0] for k, rows in chosen.items()}
    for field, label in (("model_sha256", "Same model weights (SHA-256)"),
                         ("image_list_sha256", "Same 100 images in the same order (SHA-256)"),
                         ("dataset", "Same dataset ID"), ("num_images", "Same image count"),
                         ("warmup_images", "Same warm-up"), ("cpu_threads", "Same CPU threads"),
                         ("measurement", "Same timing boundary")):
        values = {k: r[field] for k, r in first.items()}
        checks.expect(sec, label, len(set(values.values())) == 1,
                      f"{next(iter(values.values()))}",
                      ", ".join(f"{configs[k].name}={v}" for k, v in values.items()))

    image_fp = next(iter(first.values()))["image_list_sha256"]
    current = image_list_fingerprint(manifest_rows)
    checks.expect(sec, "Image list hash matches the current manifest", image_fp == current,
                  f"{current}", f"CSV {image_fp} vs manifest {current}")
    model_fp = next(iter(first.values()))["model_sha256"]
    if model_path.exists():
        current = file_sha256_short(model_path)
        checks.expect(sec, "Model hash matches models/yolo11n.pt on disk", model_fp == current,
                      f"{current}", f"CSV {model_fp} vs file {current}")
    else:
        checks.add(sec, "Model hash matches models/yolo11n.pt on disk", SKIP,
                   f"{model_path.name} not present on this computer")


def check_config_file(checks, configs, comparisons):
    sec = "Configuration file"
    combos = sorted((c.device, c.image_size) for c in configs.values())
    checks.expect(sec, "Exactly CPU/GPU x 640/320", combos == [("cpu", 320), ("cpu", 640),
                                                              ("mps", 320), ("mps", 640)],
                  ", ".join(f"{c.key}={c.name}" for c in configs.values()), f"found {combos}")
    shared = ("batch_size", "conf_threshold", "iou_threshold", "num_trials", "num_warmup",
              "cpu_threads", "expected_images", "dataset_id", "model_path", "manifest_path")
    differing = [f for f in shared if len({getattr(c, f) for c in configs.values()}) > 1]
    checks.expect(sec, "Only device and resolution differ between configurations", not differing,
                  "batch 1, conf 0.25, IoU 0.7, same model/manifest/trials/warm-up",
                  f"also differ: {differing}")
    checks.expect(sec, "Batch size 1", all(c.batch_size == 1 for c in configs.values()), "1",
                  "batch size is not 1")
    checks.expect(sec, "At least 5 trials configured", all(c.num_trials >= 5 for c in configs.values()),
                  f"{next(iter(configs.values())).num_trials}", "fewer than 5 trials")

    bad = []
    for comp in comparisons:
        ref, cmp_ = configs[comp["baseline"]], configs[comp["comparison"]]
        if comp["kind"] == "device":
            ok = (ref.device, cmp_.device) == ("cpu", "mps") and ref.image_size == cmp_.image_size
        else:
            ok = ref.device == cmp_.device and (ref.image_size, cmp_.image_size) == (640, 320)
        if not ok:
            bad.append(comp["title"])
    checks.expect(sec, "Speedup pairs: CPU/GPU at equal size, 640 -> 320 on equal device",
                  not bad and len(comparisons) == 4,
                  "; ".join(f"{c['title']} = {configs[c['baseline']].name} / "
                            f"{configs[c['comparison']].name}" for c in comparisons),
                  f"wrong pairs: {bad}")


def check_letterbox(checks, configs, manifest_rows):
    sec = "Input resolution actually applied"
    sizes = sorted({c.image_size for c in configs.values()}, reverse=True)
    dims = [(int(r["width"]), int(r["height"])) for r in manifest_rows]
    for size in sizes:
        shapes = Counter(letterbox_shape(w, h, size) for w, h in dims)
        ok = all(max(s) == size and s[0] % STRIDE == 0 and s[1] % STRIDE == 0 and min(s) <= size
                 for s in shapes)
        checks.expect(sec, f"{size}x{size}: longest side = {size}, padded to a multiple of {STRIDE}",
                      ok, f"{len(shapes)} distinct input shapes (rect letterbox), "
                          f"e.g. {shapes.most_common(1)[0][0]} (h, w)",
                      f"unexpected shapes {sorted(shapes)}")
    try:                                        # cross-check with Ultralytics itself (Mac)
        import numpy as np
        from ultralytics.data.augment import LetterBox
    except Exception as e:                      # noqa: BLE001 - optional dependency
        checks.add(sec, "Shapes match Ultralytics' own LetterBox", SKIP,
                   f"ultralytics/numpy not importable here ({type(e).__name__})")
        return
    mismatches = 0
    for size in sizes:
        lb = LetterBox(new_shape=(size, size), auto=True, stride=STRIDE)
        for w, h in set(dims):
            out = lb(image=np.zeros((h, w, 3), dtype=np.uint8))
            if tuple(out.shape[:2]) != letterbox_shape(w, h, size):
                mismatches += 1
    checks.expect(sec, "Shapes match Ultralytics' own LetterBox", mismatches == 0,
                  "all image sizes, both resolutions", f"{mismatches} mismatches")


# ----------------------------------------------------------------------------
# Checks on Step 5 / Step 6 outputs
# ----------------------------------------------------------------------------
def check_summary_files(checks, configs, recalc, speedups, session_id, summary_dir, summaries_dir):
    # --- Step 6 summary CSV ---
    sec = "Step 6 summary and speedup CSVs"
    path = summaries_dir / f"performance_summary_{session_id}.csv"
    if checks.expect(sec, f"{path.name} exists", path.exists(), "", "not found; run "
                     f"python -m src.analyze_results --session {session_id}"):
        rows = {r["config_id"]: r for r in read_csv(path)}
        fields = (("mean_execution_time_s", "mean_time_s"),
                  ("avg_processing_time_ms_per_image", "ms_per_image"),
                  ("mean_throughput_img_per_s", "mean_throughput"),
                  ("min_execution_time_s", "min_time_s"), ("max_execution_time_s", "max_time_s"),
                  ("execution_time_range_s", "range_s"))
        bad = []
        for key, cfg in configs.items():
            row = rows.get(key)
            if row is None:
                bad.append(f"{cfg.name} missing")
                continue
            for col, metric in fields:
                if not close(float(row[col]), recalc[key][metric]):
                    bad.append(f"{cfg.name} {col}: file {row[col]} vs {recalc[key][metric]:.6f}")
            if int(row["trials"]) != recalc[key]["trials"] or \
                    int(row["images_per_trial"]) != recalc[key]["num_images"]:
                bad.append(f"{cfg.name} trial/image count")
        checks.expect(sec, "Summary values = independent recalculation", not bad,
                      "mean, ms/image, throughput, min, max, range for all 4", "; ".join(bad))

    path = summaries_dir / f"speedup_{session_id}.csv"
    if checks.expect(sec, f"{path.name} exists", path.exists(), "", "not found"):
        rows = {r["comparison"]: r for r in read_csv(path)}
        bad = []
        for s in speedups:
            row = rows.get(s["title"])
            if row is None:
                bad.append(f"{s['title']} missing")
                continue
            if not close(float(row["speedup"]), s["speedup"]):
                bad.append(f"{s['title']}: file {row['speedup']} vs {s['speedup']:.4f}")
            if (row["reference"], row["comparison_config"]) != (s["reference"], s["comparison"]):
                bad.append(f"{s['title']}: numerator/denominator swapped")
            faster = {"same": "neither", "comparison faster": s["comparison"],
                      "comparison SLOWER": s["reference"]}[s["direction"]]
            if row["faster_configuration"] != faster:
                bad.append(f"{s['title']}: says {row['faster_configuration']} was faster")
        checks.expect(sec, "Speedups and direction = independent recalculation", not bad,
                      "; ".join(f"{s['title']} {s['speedup']:.3f}" for s in speedups),
                      "; ".join(bad))

    path = summaries_dir / f"analysis_report_{session_id}.md"
    if checks.expect(sec, f"{path.name} exists", path.exists(), "", "not found"):
        text = path.read_text(encoding="utf-8")
        fastest = min(recalc.values(), key=lambda r: r["mean_time_s"])
        best = max(recalc.values(), key=lambda r: r["mean_throughput"])
        claims = [f"**Shortest mean execution time:** {fastest['label']}, "
                  f"{fastest['mean_time_s']:.3f} s",
                  f"**Highest throughput:** {best['label']}, {best['mean_throughput']:.2f} images/s"]
        missing = [c for c in claims if c not in text]
        checks.expect(sec, "Report's fastest / highest-throughput claims match the data",
                      not missing, f"{fastest['label']} fastest; {best['label']} highest throughput",
                      f"not found in report: {missing}")

    # --- Step 5 summary CSVs (one per compare_results call for this session) ---
    sec = "Step 5 summary CSVs"
    files = sorted(summary_dir.glob(f"summary_*_{session_id}.csv"))
    if not files:
        checks.add(sec, "Step 5 summary for this session", SKIP, "no summary_*_<session>.csv found")
    for path in files:
        bad = []
        for row in read_csv(path):
            key = row["config_id"]
            if key not in recalc or row.get("status") != "OK":
                bad.append(f"{row['configuration']} status {row.get('status')}")
                continue
            for col, metric in (("mean_total_time_s", "mean_time_s"),
                                ("min_total_time_s", "min_time_s"),
                                ("max_total_time_s", "max_time_s"),
                                ("mean_avg_time_ms_per_image", "ms_per_image"),
                                ("mean_throughput_img_per_s", "mean_throughput")):
                if not close(float(row[col]), recalc[key][metric]):
                    bad.append(f"{row['configuration']} {col}")
        checks.expect(sec, f"{path.name} = recalculation", not bad, "all 4 configurations OK",
                      "; ".join(bad))
    for path in sorted(summary_dir.glob(f"speedup_*_{session_id}.csv")):
        rows = {r["comparison"]: r for r in read_csv(path)}
        bad = [s["title"] for s in speedups
               if s["title"] not in rows or not rows[s["title"]]["speedup"]
               or not close(float(rows[s["title"]]["speedup"]), s["speedup"])]
        checks.expect(sec, f"{path.name} = recalculation", not bad, "4 speedups", f"differ: {bad}")


def check_graphs(checks, configs, comparisons, chosen, session_id, graphs_dir, dpi, recalc):
    sec = "Step 6 graphs"
    names = [f"execution_time_{session_id}.png", f"throughput_{session_id}.png",
             f"cpu_gpu_speedup_{session_id}.png"]
    present = []
    for name in names:
        path = graphs_dir / name
        if not path.exists():
            checks.add(sec, f"{name} exists", FAIL,
                       f"not found; run python -m src.analyze_results --session {session_id}")
            continue
        info = png_info(path)
        ok = info is not None and info[2] is not None and info[2] >= dpi
        checks.expect(sec, f"{name}: PNG, >= {dpi} dpi", ok,
                      f"{info[0]}x{info[1]} px, {info[2]} dpi" if info else "",
                      f"header: {info}")
        present.append(path)

    # The values the graphs plot, taken from analyze_results, must equal the recalculation.
    from src import analyze_results as ar
    plotted = {k: ar.analyze_run(chosen[k], configs[k]) for k in configs}
    bad = [configs[k].name for k in configs
           if not (close(plotted[k]["mean_time_s"], recalc[k]["mean_time_s"]) and
                   close(plotted[k]["mean_throughput"], recalc[k]["mean_throughput"]))]
    checks.expect(sec, "Plotted bar values = independent recalculation", not bad,
                  "execution time and throughput bars", f"differ for {bad}")
    labels = [ar.graph_label(configs[k]) for k in configs]
    checks.expect(sec, "Configuration labels", labels == ["CPU 640×640", "CPU 320×320",
                                                          "GPU 640×640", "GPU 320×320"],
                  ", ".join(labels), f"labels {labels}")

    if not present:
        return
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        checks.add(sec, "Graphs identical to a fresh render of the measured data", SKIP,
                   "matplotlib not installed here")
        return
    speedups = {s["comparison"]: s["speedup"]
                for s in ar.compute_speedups(plotted, configs, comparisons)}
    cpu_gpu = [("320×320", speedups["CPU vs. GPU at 320x320"]),
               ("640×640", speedups["CPU vs. GPU at 640x640"])]
    with tempfile.TemporaryDirectory() as tmp:
        fresh = {Path(p).name: Path(p) for p in
                 ar.make_graphs(plotted, cpu_gpu, Path(tmp), session_id, dpi)}
        for path in present:
            same = path.read_bytes() == fresh[path.name].read_bytes()
            checks.expect(sec, f"{path.name} identical to a fresh render", same,
                          "byte-identical (graph shows the measured data)",
                          "differs (older data or another matplotlib version); regenerate with "
                          f"python -m src.analyze_results --session {session_id}")


def check_environment(checks):
    sec = "Environment (this computer)"
    try:
        import torch
    except ImportError:
        checks.add(sec, "PyTorch / MPS", SKIP, "torch not installed here")
        return
    import os
    import platform
    checks.add(sec, "Python / PyTorch", INFO,
               f"Python {platform.python_version()}, PyTorch {torch.__version__}, "
               f"{platform.system()} {platform.machine()}")
    checks.add(sec, "MPS built / available", INFO,
               f"{torch.backends.mps.is_built()} / {torch.backends.mps.is_available()}")
    checks.expect(sec, "PYTORCH_ENABLE_MPS_FALLBACK not set (no silent CPU fallback)",
                  os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1", "not set",
                  "set to 1; GPU runs would refuse to start")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def validate(session=None, config_file=CONFIG_PATH):
    """Run every check; return (Checks, session_id or None)."""
    checks = Checks()
    configs = load_all_configs(config_file)
    comparisons = load_comparisons(config_file)
    settings = read_config_file(config_file).get("analysis", {})
    cfg0 = next(iter(configs.values()))
    summaries_dir = PROJECT_ROOT / settings.get("summaries_dir", "results/summaries")
    graphs_dir = PROJECT_ROOT / settings.get("graphs_dir", "results/graphs")
    dpi = int(settings.get("graph_dpi", 300))
    manifest_rows = read_csv(cfg0.manifest_path)

    check_config_file(checks, configs, comparisons)
    check_letterbox(checks, configs, manifest_rows)
    check_environment(checks)

    runs = read_raw(cfg0.raw_results_dir)
    sessions = group_sessions(runs)
    sid = pick_session(sessions, runs, list(configs), session)
    sec = "Benchmark session"
    if sid is None:
        checks.add(sec, "A session with all four configurations", FAIL,
                   f"none found in {cfg0.raw_results_dir} (sessions: {sorted(sessions) or 'none'}). "
                   "Collect one with: python -m src.run_all_benchmarks")
        return checks, None
    checks.add(sec, "Session validated", INFO, sid)
    other = sorted(s for s in sessions if s != sid)
    if other:
        checks.add(sec, "Other sessions in results/raw (not combined)", INFO, ", ".join(other))

    slot = sessions[sid]
    chosen = {}
    for key, cfg in configs.items():
        ids = slot.get(key, [])
        checks.expect(sec, f"Exactly one run of {cfg.name} in this session", len(ids) == 1,
                      ids[0] if ids else "", f"{len(ids)} runs: {ids}")
        if ids:   # analyze_results uses the latest; the FAIL above flags any ambiguity
            chosen[key] = runs[max(ids, key=lambda r: runs[r][0]["timestamp"])]
    failures = []
    fail_path = Path(cfg0.raw_results_dir) / FAILURES_FILE
    if fail_path.exists():
        failures = [f for f in read_csv(fail_path) if f.get("session_id") == sid]
    checks.add(sec, "Failed runs recorded for this session", INFO,
               "; ".join(f"{f['configuration']}: {f['error']}" for f in failures) or "none")

    times = {}
    for key, cfg in configs.items():
        if chosen.get(key):
            t = check_run(checks, cfg, chosen[key])
            if t is not None:
                times[key] = t
    if len(times) != len(configs):
        checks.add("Recalculation", "All four configurations usable", FAIL,
                   "invalid trials above; nothing recalculated (no trial is dropped silently)")
        return checks, sid
    check_session_consistency(checks, configs, chosen, manifest_rows, cfg0.model_path)

    # --- Independent recalculation ---
    sec = "Recalculation"
    recalc = {}
    for key, cfg in configs.items():
        r = recalculate(times[key], int(chosen[key][0]["num_images"]))
        r["label"] = f"{'GPU' if cfg.device == 'mps' else 'CPU'} {cfg.image_size}×{cfg.image_size}"
        recalc[key] = r
        checks.add(sec, cfg.name, INFO,
                   f"mean {r['mean_time_s']:.3f} s, {r['ms_per_image']:.2f} ms/image, "
                   f"{r['mean_throughput']:.2f} images/s, min {r['min_time_s']:.3f}, "
                   f"max {r['max_time_s']:.3f}, range {r['range_s']:.3f} s "
                   f"({r['range_s'] / r['mean_time_s'] * 100:.1f}% of mean)")
    speedups = []
    for comp in comparisons:
        ref, cmp_ = configs[comp["baseline"]], configs[comp["comparison"]]
        s = recalc[ref.key]["mean_time_s"] / recalc[cmp_.key]["mean_time_s"]
        speedups.append({"title": comp["title"], "kind": comp["kind"], "reference": ref.name,
                         "comparison": cmp_.name, "speedup": s, "direction": speedup_direction(s)})
        checks.add(sec, f"Speedup: {comp['title']}", INFO,
                   f"{ref.name} {recalc[ref.key]['mean_time_s']:.3f} s / {cmp_.name} "
                   f"{recalc[cmp_.key]['mean_time_s']:.3f} s = {s:.3f} ({speedup_direction(s)}; "
                   f"{'same workload, different processor' if comp['kind'] == 'device' else 'different workload (resolution)'})")

    check_summary_files(checks, configs, recalc, speedups, sid, cfg0.summary_dir, summaries_dir)
    check_graphs(checks, configs, comparisons, chosen, sid, graphs_dir, dpi, recalc)
    return checks, sid


def to_markdown(checks, session_id):
    lines = [f"# Step 7 Data Validation — session `{session_id or 'none'}`", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')} by "
             "`python -m src.validate_experiment`.", "",
             f"**{checks.count(PASS)} passed, {checks.count(FAIL)} failed, "
             f"{checks.count(SKIP)} skipped.**", ""]
    section = None
    for sec, name, status, detail in checks.items:
        if sec != section:
            lines += ["", f"## {sec}", "", "| Check | Result | Details |", "|---|---|---|"]
            section = sec
        lines.append(f"| {name} | {status} | {detail.replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validate the recorded benchmark measurements.")
    p.add_argument("--session", default=None,
                   help="Session ID (default: latest session with all four configurations)")
    p.add_argument("--config-file", type=Path, default=CONFIG_PATH)
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "validation")
    p.add_argument("--no-save", action="store_true", help="Print only")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        checks, sid = validate(args.session, args.config_file)
    except (ConfigError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return 1

    section = None
    for sec, name, status, detail in checks.items:
        if sec != section:
            print(f"\n== {sec} ==")
            section = sec
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    print(f"\n{checks.count(PASS)} passed, {checks.count(FAIL)} failed, "
          f"{checks.count(SKIP)} skipped.")
    if not args.no_save:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = args.out_dir / f"data_validation_{sid or 'none'}_{stamp}.md"
        with open(path, "x", encoding="utf-8") as f:      # never overwrite earlier output
            f.write(to_markdown(checks, sid))
        print(f"Saved: {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
    return 0 if checks.count(FAIL) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
