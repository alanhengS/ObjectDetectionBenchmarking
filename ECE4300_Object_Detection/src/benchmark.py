"""
ECE 4300 - Step 4: CPU baseline performance benchmark.

Measures the elapsed time for YOLO11n to process the fixed 100-image COCO
2017 benchmark set on the Apple M2 CPU at 640x640, repeated for 5 trials.

Metrics (ECE 4300 Lecture 1: execution time and throughput):
    Total execution time  = elapsed seconds for all images in one trial
    Average time / image  = total time / number of images      (ms/image)
    Throughput            = number of images / total time      (images/s)

Timing boundary: the timer wraps ONLY the sequential loop of model.predict()
calls over images already decoded in memory. This includes YOLO's own
preprocessing (letterbox resize, normalization), the neural-network forward
pass, and postprocessing (NMS). It excludes model loading, manifest reading,
image decoding from disk, warm-up, printing, and CSV writing.

Run from the project root:
    python -m src.benchmark
"""

import csv
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

from src.detector import MODEL_NAME, PROJECT_ROOT, load_model

# ----------------------------------------------------------------------------
# Benchmark configuration (fixed for the CPU baseline)
# ----------------------------------------------------------------------------
DEVICE = "cpu"
IMAGE_SIZE = 640          # 640x640 model input
BATCH_SIZE = 1            # one image per predict() call
CONF_THRESHOLD = 0.25     # Ultralytics default, set explicitly for consistency
IOU_THRESHOLD = 0.7       # NMS IoU threshold (Ultralytics default)
NUM_TRIALS = 5
NUM_WARMUP = 3            # warm-up inferences before timing (not recorded)
# PyTorch CPU threads. Set explicitly because Ultralytics changes the thread
# count during its first predict() call (to os.cpu_count() - 1 = 7 on the M2).
# 4 = number of M2 performance cores (PyTorch's own default on this Mac).
NUM_THREADS = 4
EXPECTED_IMAGES = 100

CONFIG_NAME = "CPU-640"
DATASET_ID = "COCO2017-val-first100-by-id"
MANIFEST_PATH = PROJECT_ROOT / "data" / "benchmark" / "image_manifest.csv"
RAW_RESULTS_DIR = PROJECT_ROOT / "results" / "raw"

CSV_COLUMNS = [
    "run_id", "timestamp", "configuration", "processor", "resolution",
    "model", "dataset", "num_images", "cpu_threads", "trial",
    "total_time_s", "avg_time_ms_per_image", "throughput_img_per_s",
]


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


# ----------------------------------------------------------------------------
# Timed inference
# ----------------------------------------------------------------------------
def predict_one(model, image):
    """One inference call with the fixed benchmark settings."""
    return model.predict(
        source=image,
        device=DEVICE,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )


def warm_up(model, images, count=NUM_WARMUP):
    """Untimed inferences so one-time setup cost is not measured."""
    for _, image in images[:count]:
        predict_one(model, image)


def run_trial(model, images):
    """
    Process all images sequentially and return elapsed seconds.

    The model must already be loaded: only the inference loop is timed.
    """
    start = time.perf_counter()
    for _, image in images:
        predict_one(model, image)
    end = time.perf_counter()
    return end - start


def run_benchmark(model, images, num_trials=NUM_TRIALS):
    """Run repeated trials on the same model and images; return per-trial records."""
    records = []
    for trial in range(1, num_trials + 1):
        total = run_trial(model, images)
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


def new_run_id(now):
    """Unique session ID: timestamp plus a short random suffix."""
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def save_results(records, run_id, timestamp, processor, num_images, out_dir=RAW_RESULTS_DIR):
    """
    Write one CSV row per trial to a new file named after the run ID.

    Opens with mode 'x' (exclusive create), so an existing file is never
    overwritten.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{CONFIG_NAME.lower()}_{run_id}.csv"
    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "run_id": run_id,
                "timestamp": timestamp,
                "configuration": CONFIG_NAME,
                "processor": f"{processor} CPU",
                "resolution": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
                "model": MODEL_NAME,
                "dataset": DATASET_ID,
                "num_images": num_images,
                "cpu_threads": torch.get_num_threads(),
                "trial": r["trial"],
                "total_time_s": f"{r['total_time_s']:.6f}",
                "avg_time_ms_per_image": f"{r['avg_time_ms_per_image']:.4f}",
                "throughput_img_per_s": f"{r['throughput_img_per_s']:.4f}",
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


def main():
    line = "-" * 41
    processor = get_processor_name()
    print("ECE 4300 — CPU Baseline Benchmark")
    print(line)
    print("Model:       YOLO11n")
    print(f"Processor:   {processor} CPU ({NUM_THREADS} PyTorch threads)")
    print(f"Resolution:  {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Dataset:     COCO 2017 val ({EXPECTED_IMAGES} images)")
    print(f"Batch Size:  {BATCH_SIZE}")
    print()

    # --- Setup (not timed) ---
    try:
        rows = read_manifest()
    except FileNotFoundError:
        print(f"ERROR: Manifest not found: {MANIFEST_PATH}")
        print("Run: python src/prepare_benchmark_dataset.py")
        return 1
    if len(rows) != EXPECTED_IMAGES:
        print(f"ERROR: Manifest lists {len(rows)} images, expected {EXPECTED_IMAGES}.")
        return 1

    try:
        images = load_images(rows)
    except FileNotFoundError as e:
        print(f"ERROR: {e}\nRun: python tests/validate_benchmark_dataset.py")
        return 1
    megabytes = sum(img.nbytes for _, img in images) / 1e6
    print(f"Preloaded {len(images)} images into memory ({megabytes:.0f} MB decoded).")

    try:
        model = load_model()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    warm_up(model, images)
    # Must come AFTER warm-up: the first predict() call lets Ultralytics
    # reset the thread count, so we set our fixed value afterwards.
    torch.set_num_threads(NUM_THREADS)
    print(f"Warm-up complete ({NUM_WARMUP} untimed inferences).")
    print()

    # --- Timed trials ---
    now = datetime.now()
    run_id = new_run_id(now)
    records = run_benchmark(model, images)
    if torch.get_num_threads() != NUM_THREADS:
        print(f"WARNING: thread count changed to {torch.get_num_threads()} during trials.")

    # --- Results (not timed) ---
    s = summarize(records)
    path = save_results(records, run_id, now.isoformat(timespec="seconds"),
                        processor, len(images))

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
    print(f"Results saved to:       {path.relative_to(PROJECT_ROOT)}")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
