"""
ECE 4300 - Steps 4 and 5: CPU / GPU (MPS) performance benchmark.

Measures the elapsed time for YOLO11n to process the fixed 100-image COCO
2017 benchmark set, for one of four configurations defined in
config/benchmark_config.json:

    A  CPU-640   Apple M2 CPU        640x640   (baseline)
    B  CPU-320   Apple M2 CPU        320x320
    C  GPU-640   Apple M2 GPU (MPS)  640x640
    D  GPU-320   Apple M2 GPU (MPS)  320x320

Metrics (ECE 4300 Lecture 1: execution time and throughput):
    Total execution time  = elapsed seconds for all images in one trial
    Average time / image  = total time / number of images      (ms/image)
    Throughput            = number of images / total time      (images/s)

Timing boundary (identical for all four configurations): the timer wraps
ONLY the sequential loop of model.predict() calls over images already
decoded in memory. This includes YOLO's own preprocessing (letterbox resize,
normalization), the neural-network forward pass, and postprocessing (NMS).
On the GPU, torch.mps.synchronize() is called after every image so the timer
never stops while GPU work is still queued; on the CPU that call is a no-op.
Excluded: model loading, manifest reading, image decoding from disk,
warm-up, printing, and CSV writing.

Run from the project root:
    python -m src.benchmark                 # configuration A (CPU-640 baseline)
    python -m src.benchmark --config B      # or CPU-320, C / GPU-640, D / GPU-320
"""

import argparse
import csv
import hashlib
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import torch

from src.benchmark_config import CONFIG_PATH, ConfigError, load_config
from src.detector import PROJECT_ROOT, load_model

# ----------------------------------------------------------------------------
# Baseline settings (configuration A), kept as module constants for
# backward compatibility with Step 4. Values come from the config file.
# ----------------------------------------------------------------------------
BASELINE = load_config()

DEVICE = BASELINE.device
IMAGE_SIZE = BASELINE.image_size
BATCH_SIZE = BASELINE.batch_size
CONF_THRESHOLD = BASELINE.conf_threshold
IOU_THRESHOLD = BASELINE.iou_threshold
NUM_TRIALS = BASELINE.num_trials
NUM_WARMUP = BASELINE.num_warmup
# PyTorch CPU threads. Set explicitly because Ultralytics changes the thread
# count during its first predict() call (to os.cpu_count() - 1 = 7 on the M2).
# 4 = number of M2 performance cores (PyTorch's own default on this Mac).
# Used for every configuration: on the GPU, pre/postprocessing still run on the CPU.
NUM_THREADS = BASELINE.cpu_threads
EXPECTED_IMAGES = BASELINE.expected_images

CONFIG_NAME = BASELINE.name
DATASET_ID = BASELINE.dataset_id
MANIFEST_PATH = BASELINE.manifest_path
RAW_RESULTS_DIR = BASELINE.raw_results_dir

MEASUREMENT_LABEL = "elapsed_workload_time"   # preprocess + inference + postprocess
FAILURES_FILE = "failed_runs.csv"

CSV_COLUMNS = [
    "run_id", "session_id", "timestamp", "configuration", "config_id", "device",
    "processor", "resolution", "model", "model_sha256", "dataset", "image_list_sha256",
    "num_images", "cpu_threads", "warmup_images", "measurement", "trial",
    "total_time_s", "avg_time_ms_per_image", "throughput_img_per_s",
]
FAILURE_COLUMNS = [
    "timestamp", "session_id", "configuration", "config_id", "device",
    "resolution", "error",
]


class DeviceError(RuntimeError):
    """The requested device cannot be used, or the model is not running on it."""


# ----------------------------------------------------------------------------
# Metric formulas
# ----------------------------------------------------------------------------
def average_time_ms(total_time_s, num_images):
    """Average processing time per image, in milliseconds."""
    if total_time_s <= 0 or num_images <= 0:
        raise ValueError("Time and image count must be positive.")
    return total_time_s / num_images * 1000.0


def throughput(total_time_s, num_images):
    """Images processed per second."""
    if total_time_s <= 0 or num_images <= 0:
        raise ValueError("Time and image count must be positive.")
    return num_images / total_time_s


# ----------------------------------------------------------------------------
# Dataset loading (outside the timed region)
# ----------------------------------------------------------------------------
def read_manifest(path=MANIFEST_PATH):
    """Return manifest rows in file order (sorted by image_id)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_images(rows, root=PROJECT_ROOT):
    """
    Decode every manifest image into memory, in manifest order.

    Returns a list of (file_name, BGR numpy array). Raises if any image is
    missing or unreadable, so a benchmark never runs on a partial dataset.
    """
    images = []
    for row in rows:
        path = root / row["relative_path"]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Missing or unreadable benchmark image: {path}")
        images.append((row["file_name"], image))
    return images


def image_list_fingerprint(rows):
    """
    Short SHA-256 of the ordered (file name, file checksum) list.

    Identical values in two CSVs prove both runs used the same images in the
    same order.
    """
    text = "\n".join(f"{r['file_name']},{r.get('sha256', '')}" for r in rows)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def file_sha256(path):
    """Short SHA-256 of a file (used to prove every run used the same weights)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------
# Device checks (outside the timed region)
# ----------------------------------------------------------------------------
def check_device_available(device):
    """
    Raise DeviceError with an understandable message if `device` cannot be used.

    Ultralytics quietly switches to the CPU when MPS is requested but not
    available, so this check must happen before any inference.
    """
    if device == "cpu":
        return
    if device != "mps":
        raise DeviceError(f"Unsupported device {device!r}; use 'cpu' or 'mps'.")
    if not torch.backends.mps.is_built():
        raise DeviceError(
            "This PyTorch installation was built without MPS (Apple GPU) support. "
            "GPU configurations need the macOS arm64 build of PyTorch on an "
            "Apple Silicon Mac."
        )
    if not torch.backends.mps.is_available():
        raise DeviceError(
            "The Apple GPU (MPS backend) is not available on this machine. "
            "MPS needs an Apple Silicon Mac running macOS 12.3 or later. "
            f"(Detected: {platform.system()} {platform.machine()}.) "
            "The benchmark will not fall back to the CPU."
        )
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise DeviceError(
            "PYTORCH_ENABLE_MPS_FALLBACK=1 is set, which lets PyTorch run "
            "unsupported operations on the CPU without warning. Unset it "
            "(unset PYTORCH_ENABLE_MPS_FALLBACK) so the GPU timing is pure MPS."
        )


def synchronize(device):
    """
    Block until all queued GPU work has finished.

    MPS runs kernels asynchronously: a Python call can return before the GPU
    is done. Synchronizing before reading the clock makes GPU timings
    comparable with CPU timings. On the CPU, work is already finished when
    predict() returns, so nothing is needed.
    """
    if device == "mps":
        torch.mps.synchronize()


def weights_device(predictor_model):
    """
    Device type of the weights of the network the predictor actually runs.

    Ultralytics wraps the network in AutoBackend; in Ultralytics 8.4 the
    weights are at AutoBackend.backend.model, and the wrapper itself has none.
    """
    backend = getattr(predictor_model, "backend", None)
    for net in (getattr(backend, "model", None), predictor_model):
        if isinstance(net, torch.nn.Module):
            param = next(net.parameters(), None)
            if param is not None:
                return param.device.type
    raise DeviceError("Could not locate the model weights to check which device they are on.")


def verify_model_device(model, result, device):
    """
    Confirm the model's weights, the predictor, and the output tensors are on
    the requested device. Raises DeviceError if any of them is elsewhere.
    """
    found = {
        "predictor": model.predictor.device.type,
        "model weights": weights_device(model.predictor.model),
        "output boxes": result.boxes.data.device.type,
    }
    wrong = {k: v for k, v in found.items() if v != device}
    if wrong:
        details = ", ".join(f"{k} on {v}" for k, v in wrong.items())
        raise DeviceError(f"Requested device '{device}', but found {details}. "
                          "Refusing to benchmark on the wrong device.")
    return found


# ----------------------------------------------------------------------------
# Timed inference
# ----------------------------------------------------------------------------
def predict_one(model, image, cfg=None):
    """One inference call with the fixed benchmark settings."""
    cfg = cfg or BASELINE
    return model.predict(
        source=image,
        device=cfg.device,
        imgsz=cfg.image_size,
        batch=cfg.batch_size,
        conf=cfg.conf_threshold,
        iou=cfg.iou_threshold,
        verbose=False,
    )


def warm_up(model, images, count=None, cfg=None):
    """Untimed inferences so one-time setup cost is not measured."""
    cfg = cfg or BASELINE
    count = cfg.num_warmup if count is None else count
    for _, image in images[:count]:
        predict_one(model, image, cfg)
    synchronize(cfg.device)


def run_trial(model, images, cfg=None):
    """
    Process all images sequentially and return elapsed seconds.

    The model must already be loaded: only the inference loop is timed.
    Each image is fully processed (including any queued GPU work) before the
    next one starts, matching the CPU's sequential behaviour.
    """
    cfg = cfg or BASELINE
    device = cfg.device
    start = time.perf_counter()
    for _, image in images:
        predict_one(model, image, cfg)
        synchronize(device)
    end = time.perf_counter()
    return end - start


def run_benchmark(model, images, num_trials=None, cfg=None):
    """Run repeated trials on the same model and images; return per-trial records."""
    cfg = cfg or BASELINE
    num_trials = cfg.num_trials if num_trials is None else num_trials
    records = []
    for trial in range(1, num_trials + 1):
        total = run_trial(model, images, cfg)
        records.append({
            "trial": trial,
            "total_time_s": total,
            "avg_time_ms_per_image": average_time_ms(total, len(images)),
            "throughput_img_per_s": throughput(total, len(images)),
        })
        print(f"Trial {trial}: {total:.3f} s")
    return records


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
def get_processor_name():
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return platform.processor() or "Unknown"


def display_path(path):
    """Path relative to the project root when possible (for console output)."""
    return path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path


def processor_label(processor, cfg):
    """e.g. 'Apple M2 CPU' or 'Apple M2 GPU (MPS)'."""
    return f"{processor} {cfg.device_label}"


def new_run_id(now):
    """Unique session ID: timestamp plus a short random suffix."""
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def save_results(records, run_id, timestamp, processor, num_images, out_dir=None,
                 cfg=None, session_id=None, model_sha256="", image_list_sha256="",
                 cpu_threads=None):
    """
    Write one CSV row per trial to a new file named after the run ID.

    Opens with mode 'x' (exclusive create), so an existing file is never
    overwritten.
    """
    cfg = cfg or BASELINE
    out_dir = cfg.raw_results_dir if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg.name.lower()}_{run_id}.csv"
    threads = torch.get_num_threads() if cpu_threads is None else cpu_threads
    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "run_id": run_id,
                "session_id": session_id or run_id,
                "timestamp": timestamp,
                "configuration": cfg.name,
                "config_id": cfg.key,
                "device": cfg.device,
                "processor": processor_label(processor, cfg),
                "resolution": cfg.resolution,
                "model": cfg.model_path.name,
                "model_sha256": model_sha256,
                "dataset": cfg.dataset_id,
                "image_list_sha256": image_list_sha256,
                "num_images": num_images,
                "cpu_threads": threads,
                "warmup_images": cfg.num_warmup,
                "measurement": MEASUREMENT_LABEL,
                "trial": r["trial"],
                "total_time_s": f"{r['total_time_s']:.6f}",
                "avg_time_ms_per_image": f"{r['avg_time_ms_per_image']:.4f}",
                "throughput_img_per_s": f"{r['throughput_img_per_s']:.4f}",
            })
    return path


def record_failure(cfg, session_id, error, out_dir=None):
    """
    Append a failed run to results/raw/failed_runs.csv so errors are kept
    alongside the measurements (no timing data is written for a failed run).
    """
    out_dir = cfg.raw_results_dir if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FAILURES_FILE
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "configuration": cfg.name,
            "config_id": cfg.key,
            "device": cfg.device,
            "resolution": cfg.resolution,
            "error": " ".join(str(error).split()),
        })
    return path


def summarize(records):
    times = [r["total_time_s"] for r in records]
    return {
        "mean_time_s": statistics.mean(times),
        "mean_latency_ms": statistics.mean(r["avg_time_ms_per_image"] for r in records),
        "mean_throughput": statistics.mean(r["throughput_img_per_s"] for r in records),
        "min_time_s": min(times),
        "max_time_s": max(times),
        "stdev_time_s": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ECE 4300 YOLO11n benchmark (one configuration).")
    p.add_argument("--config", default=None,
                   help="Configuration key or name: A/CPU-640 (default), B/CPU-320, "
                        "C/GPU-640, D/GPU-320")
    p.add_argument("--config-file", type=Path, default=CONFIG_PATH,
                   help="Path to the benchmark configuration JSON file")
    p.add_argument("--session-id", default=None,
                   help="Groups runs started together by run_all_benchmarks")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, path=args.config_file)
    except ConfigError as e:
        print(f"ERROR: {e}")
        return 1

    now = datetime.now()
    run_id = new_run_id(now)
    session_id = args.session_id or run_id

    line = "-" * 41
    processor = get_processor_name()
    title = "CPU Baseline Benchmark" if cfg.is_baseline else "Benchmark"
    print(f"ECE 4300 — {title}: configuration {cfg.key} ({cfg.name})")
    print(line)
    print(f"Model:       YOLO11n ({cfg.model_path.name})")
    print(f"Device:      {processor_label(processor, cfg)} "
          f"({cfg.cpu_threads} PyTorch CPU threads)")
    print(f"Resolution:  {cfg.image_size} x {cfg.image_size}")
    print(f"Dataset:     COCO 2017 val ({cfg.expected_images} images)")
    print(f"Batch Size:  {cfg.batch_size}")
    print(f"Trials:      {cfg.num_trials}")
    print()

    # --- Device check (not timed): never fall back silently ---
    try:
        check_device_available(cfg.device)
    except DeviceError as e:
        print(f"ERROR: {e}")
        path = record_failure(cfg, session_id, e)
        print(f"Failure recorded in {display_path(path)}")
        return 2

    # --- Setup (not timed) ---
    try:
        rows = read_manifest(cfg.manifest_path)
    except FileNotFoundError:
        print(f"ERROR: Manifest not found: {cfg.manifest_path}")
        print("Run: python src/prepare_benchmark_dataset.py")
        return 1
    if len(rows) != cfg.expected_images:
        print(f"ERROR: Manifest lists {len(rows)} images, expected {cfg.expected_images}.")
        return 1

    try:
        images = load_images(rows)
    except FileNotFoundError as e:
        print(f"ERROR: {e}\nRun: python tests/validate_benchmark_dataset.py")
        return 1
    megabytes = sum(img.nbytes for _, img in images) / 1e6
    print(f"Preloaded {len(images)} images into memory ({megabytes:.0f} MB decoded).")

    try:
        model = load_model(cfg.model_path)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    model_sha = file_sha256(cfg.model_path)
    image_sha = image_list_fingerprint(rows)

    try:
        # One untimed inference to confirm the model really runs on the device.
        result = predict_one(model, images[0][1], cfg)[0]
        synchronize(cfg.device)
        verify_model_device(model, result, cfg.device)
        print(f"Device check: model weights and outputs are on '{cfg.device}'.")

        warm_up(model, images, cfg=cfg)
        # Must come AFTER warm-up: the first predict() call lets Ultralytics
        # reset the thread count, so we set our fixed value afterwards.
        torch.set_num_threads(cfg.cpu_threads)
        print(f"Warm-up complete ({min(cfg.num_warmup, len(images))} untimed inferences).")
        print()

        # --- Timed trials ---
        print("Elapsed workload processing time per trial "
              "(preprocess + inference + postprocess):")
        records = run_benchmark(model, images, cfg=cfg)
        threads = torch.get_num_threads()
        if threads != cfg.cpu_threads:
            print(f"WARNING: thread count changed to {threads} during trials.")
        # Re-check after the trials: the device must not have changed.
        verify_model_device(model, result, cfg.device)
    except Exception as e:  # e.g. an operation not supported on MPS
        print(f"ERROR: Benchmark {cfg.name} failed: {type(e).__name__}: {e}")
        path = record_failure(cfg, session_id, f"{type(e).__name__}: {e}")
        print(f"Failure recorded in {display_path(path)}")
        return 1

    # --- Results (not timed) ---
    s = summarize(records)
    path = save_results(records, run_id, now.isoformat(timespec="seconds"),
                        processor, len(images), cfg=cfg, session_id=session_id,
                        model_sha256=model_sha, image_list_sha256=image_sha,
                        cpu_threads=threads)

    if cfg.device == "mps":
        torch.mps.empty_cache()   # release cached GPU memory (8 GB unified memory)

    print()
    print(f"Average execution time: {s['mean_time_s']:.3f} s")
    print(f"Average latency:        {s['mean_latency_ms']:.2f} ms/image")
    print(f"Average throughput:     {s['mean_throughput']:.2f} images/s")
    print()
    print(f"Minimum time:           {s['min_time_s']:.3f} s")
    print(f"Maximum time:           {s['max_time_s']:.3f} s")
    print(f"Std. deviation:         {s['stdev_time_s']:.3f} s")
    print()
    print(f"Run ID:                 {run_id}")
    print(f"Results saved to:       {display_path(path)}")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
