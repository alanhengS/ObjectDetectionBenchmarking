# ECE 4300 — Benchmark Demonstration Guide

This guide shows how to demonstrate how the performance measurements are
obtained, using the project's own benchmark program (`src/benchmark.py`).
There is no separate demo program and no graphical interface.

There are two kinds of run:

| | Quick demonstration | Formal experiment |
|---|---|---|
| Purpose | Show the measurement procedure in class | The results in the report |
| Images | First N of the 100 benchmark images (e.g. 20) | All 100 |
| Trials | 2 (default; `--demo-trials` changes it) | 5 |
| Command | `python -m src.benchmark --config C --demo 20` | `python -m src.run_all_benchmarks` |
| CSV saved in | `results/demo/` | `results/raw/` |
| Used in the analysis? | **No.** Never read by the analysis or validation scripts | Yes |

The quick demonstration uses the same model, device, resolution, prediction
settings, warm-up rule, and timer as the formal experiment. Only the number of
images and trials is smaller. **Its numbers are not the formal results** and
should not be compared directly with them: a smaller image subset is a
different workload.

---

## 1. Prerequisites

- An Apple Silicon Mac (the GPU configurations C and D need Apple's MPS).
- The project folder `ECE4300_Object_Detection` with its virtual environment
  `.venv`. To create it from scratch, see "Setup" in the README.
- Internet access only the first time, if `models/yolo11n.pt` has not been
  downloaded yet.
- Mac plugged into power, other apps closed.

## 2. Activate the Python environment

```bash
cd ECE4300_Object_Detection
source .venv/bin/activate
python tests/check_environment.py        # optional: shows Python/PyTorch versions and tests CPU + MPS
```

## 3. Required input files

| File | Purpose | If missing |
|------|---------|-----------|
| `config/benchmark_config.json` | Configurations A–D and shared settings | In git |
| `data/benchmark/image_manifest.csv` | Fixed, ordered list of the 100 images | In git, or `python src/prepare_benchmark_dataset.py` |
| `data/benchmark/images/*.jpg` | The 100 COCO 2017 images | In git, or `python src/prepare_benchmark_dataset.py` |
| `models/yolo11n.pt` | Pretrained YOLO11n weights | Downloaded automatically on the first run |

Check the dataset before the demo (no YOLO needed; it should print `RESULT: PASS`):

```bash
python tests/validate_benchmark_dataset.py
```

## 4. Demonstration commands

```bash
# GPU (MPS), 640x640, first 20 images, 2 trials
python -m src.benchmark --config C --demo 20

# Same workload on the CPU, for a live CPU-vs-GPU comparison
python -m src.benchmark --config A --demo 20

# Optional: the lower resolution on either device
python -m src.benchmark --config D --demo 20
python -m src.benchmark --config B --demo 20
```

Configuration keys: `A` = CPU 640×640, `B` = CPU 320×320, `C` = GPU 640×640,
`D` = GPU 320×320. Names such as `--config GPU-640` also work.

Each demo command runs: load model → device check → one untimed warm-up pass
over the demo images → 2 timed trials. 20 images is a reasonable class
demonstration. Any value from 1 to 100 is accepted.

## 5. What the terminal shows

The layout is shown below. The values in `< >` are whatever the run measures;
they are deliberately not filled in here.

```text
ECE 4300 — QUICK DEMONSTRATION: configuration C (GPU-640)
-----------------------------------------
Demo workload: first 20 of the 100 benchmark images, 2 trial(s).
NOT a formal result: the formal experiment uses all 100 images and 5 trials.
Saved separately in results/demo/ (formal results are untouched).
-----------------------------------------
Model:       YOLO11n (yolo11n.pt)
Device:      Apple M2 GPU (MPS) (4 PyTorch CPU threads)
Resolution:  640 x 640
Dataset:     COCO 2017 val (20 images)
Batch Size:  1
Trials:      2

Preloaded 20 images into memory (<n> MB decoded).
Device check: model weights and outputs are on 'mps'.
Warm-up complete (20 untimed inferences).

Elapsed workload processing time per trial (preprocess + inference + postprocess):
Trial 1: <t1> s
Trial 2: <t2> s

How trial 2 was calculated:
  Execution time   = timer stop - timer start          = <t2> s
  Time per image   = <t2> s / 20 images x 1000   = <ms> ms/image
  Throughput       = 20 images / <t2> s          = <ips> images/s

Average execution time: <mean> s
Average latency:        <ms> ms/image
Average throughput:     <ips> images/s
...
Run ID:                 demo-<date>-<time>-<id>
Results saved to:       results/demo/gpu-640_demo-<...>.csv
(Quick demonstration only — compare formal results in results/summaries/.)
```

Points to make during the demo:
- **"Device check"** proves the GPU run really uses `mps`. The program stops
  with an error instead of quietly falling back to the CPU.
- **Warm-up is untimed**, so one-time setup is not measured.
- The **trial times** are the measured execution times. The "How ... was
  calculated" lines apply the course formulas to those numbers.

## 6. How execution time is recorded

- `time.perf_counter()` starts immediately before the first image and stops
  immediately after the last one (`run_trial()` in `src/benchmark.py`).
- **Included:** YOLO preprocessing (resize/letterbox, normalization), the
  neural-network forward pass, and postprocessing (NMS), for every image in
  turn with batch size 1. On the GPU, `torch.mps.synchronize()` after each
  image makes the timer wait until the GPU has finished.
- **Excluded:** model loading, reading and decoding the JPEGs (preloaded),
  the device check, warm-up, printing, and CSV writing.

This is **elapsed workload processing time**, not pure GPU time or pure
neural-network time.

## 7. How the metrics are calculated

```text
Execution time T (s)          = timer stop − timer start              (one trial, N images)
Time per image (ms/image)     = T / N × 1000
Throughput (images/s)         = N / T
Mean execution time (s)       = (T1 + T2 + … + Tk) / k                (k trials)
Speedup                       = T_reference / T_comparison             (mean times)
```

Example of the speedup idea for the class: run configuration C and A with the
same `--demo` size, then divide A's mean time by C's. A value above 1 means
the GPU was faster on that demo workload; below 1 means it was slower.

## 8. Where the CSV outputs are stored

| Output | Location |
|--------|----------|
| Quick demonstration | `results/demo/<config>_demo-<run id>.csv` (one new file per run; never overwritten) |
| Failed demo runs | `results/demo/failed_runs.csv` |
| Formal measurements | `results/raw/` (not touched by the demo) |
| Formal summaries / graphs | `results/summaries/`, `results/graphs/` |

The CSV columns are the same as in the formal runs: one row per trial with
`total_time_s`, `avg_time_ms_per_image`, `throughput_img_per_s`, device,
resolution, image count, model hash, and image-list hash.

## 9. Common errors and how to recover

| Message | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` (or `ultralytics`, `cv2`) | Environment not active | `source .venv/bin/activate` |
| `The Apple GPU (MPS backend) is not available` (exit code 2) | Not on Apple Silicon / macOS too old / non-arm64 Python | Use configuration A or B, or run on the M2 Mac with the project's `.venv` |
| `PYTORCH_ENABLE_MPS_FALLBACK=1 is set` | That variable would let GPU work silently run on the CPU | `unset PYTORCH_ENABLE_MPS_FALLBACK` |
| `Could not load or download model weights` | No internet on first run | Connect once, or copy `yolo11n.pt` into `models/` |
| `Missing or unreadable benchmark image` / `Manifest not found` | Dataset incomplete | `python src/prepare_benchmark_dataset.py`, then `python tests/validate_benchmark_dataset.py` |
| `--demo must be between 1 and 100 images` | Bad demo size | Use 1–100 |
| `Unknown configuration` | Typo in `--config` | Use A, B, C, D (or CPU-640, CPU-320, GPU-640, GPU-320) |
| Trial times much longer than usual | Other apps, low-power mode, heat | Close apps, plug in, wait a minute, run again |

## 10. Showing the formal results after the demo

```bash
python -m src.analyze_results --session S20261112-121717-4092   # prints the formal table and speedups
open results/graphs/                                              # the three report graphs
```

`analyze_results` only reads the formal raw CSVs. It rewrites the same summary
and graph files for that session from the same data, and never reads
`results/demo/`.
