# Object Detection Performance Benchmarking on Apple Silicon

**ECE 4300 – Computer Architecture Project**

## Objective

Compare the inference performance of a pretrained YOLO object-detection model
on the **CPU** and the **GPU (Apple Metal Performance Shaders, MPS)** of an
Apple Silicon Mac, at two input resolutions.

## Computing Hardware

| Component | Specification              |
|-----------|----------------------------|
| Machine   | Mac with Apple M2          |
| Memory    | 8 GB unified memory        |
| OS        | macOS 26.1 (arm64)         |
| Python    | 3.13.7                     |
| PyTorch   | 2.15.0 (MPS backend)       |

## Current Development Stage

**Stage 1 – Environment setup (complete)**

- Project structure created
- Python virtual environment (`.venv`) configured
- PyTorch, Ultralytics, and OpenCV installed
- CPU and MPS support verified

**Stage 2 – Initial object detection (complete)**

- Pretrained YOLO11n model loaded with Ultralytics
- Single-image detection on the CPU (640×640, batch size 1)
- Detected classes and confidence scores printed to the console
- Annotated image with bounding boxes saved to `results/`

**Stage 3 – Benchmark dataset preparation (complete)**

- Fixed set of 100 COCO 2017 validation images downloaded
- Matching COCO annotations retained
- Image manifest (`image_manifest.csv`) defines the exact workload
- Validation script confirms dataset integrity

**Stage 4 – CPU baseline benchmark (complete)**

- YOLO11n timed on the Apple M2 CPU at 640×640 over the 100 fixed COCO images
- 5 measured trials; execution time, latency, and throughput recorded
- Raw per-trial data saved to `results/raw/`

**Stage 5 – CPU and GPU benchmark comparisons (implemented)**

- Four selectable configurations (CPU/GPU × 640/320) defined in `config/benchmark_config.json`
- Apple GPU execution through PyTorch MPS, with availability checks and `torch.mps.synchronize()`
- Sequential runner for all four configurations, summary table, and speedup calculations
- Measured results for all four configurations still need to be recorded on the Mac (see Step 5)

**Stage 6 – Performance analysis and graphs (implemented)**

- `src/analyze_results.py` checks the Step 5 measurements and calculates mean time, ms/image, throughput, range, and speedups
- Summary CSV, speedup CSV, written report, and three graphs
- Run on the Mac after Step 5 has been measured (see Step 6)

No detection accuracy evaluation has been added yet.

## Project Structure

```text
ECE4300_Object_Detection/
├── config/
│   └── benchmark_config.json    # Step 5: settings for configurations A–D
├── data/
│   ├── images/                  # Custom test images (bus.jpg, your own)
│   └── benchmark/               # Fixed COCO workload (Step 3)
│       ├── images/              # 100 COCO 2017 val images
│       ├── annotations/         # instances_val2017_subset.json
│       └── image_manifest.csv   # Fixed, ordered image list
├── models/        # YOLO model weights (yolo11n.pt)
├── src/
│   ├── detector.py  # Loads YOLO11n and runs inference
│   ├── main.py      # Step 2 single-image detection test
│   ├── prepare_benchmark_dataset.py  # Step 3 COCO subset download
│   ├── benchmark.py                  # Steps 4–5: benchmark one configuration
│   ├── benchmark_config.py           # Step 5: loads config/benchmark_config.json
│   ├── run_all_benchmarks.py         # Step 5: runs A–D sequentially
│   ├── compare_results.py            # Step 5: summary table and speedups
│   └── analyze_results.py            # Step 6: analysis, report, and graphs
├── results/       # Output images
│   ├── raw/       # Benchmark CSVs (one file per run) + failed_runs.csv
│   ├── summary/   # Step 5 summary and speedup CSVs
│   ├── summaries/ # Step 6 summary CSV, speedup CSV, and analysis report
│   └── graphs/    # Step 6 PNG graphs
└── tests/         # Environment and unit checks
```

## Setup

From the `ECE4300_Object_Detection` folder:

```bash
# Activate the virtual environment
source .venv/bin/activate

# (Only if recreating the environment from scratch)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In VS Code, open the `ECE4300_Object_Detection` folder. The interpreter is set
to `.venv/bin/python` in `.vscode/settings.json`, so new terminals activate the
environment automatically. Run `deactivate` to leave the environment.

## Verify the Environment

```bash
python tests/check_environment.py
```

This prints the Python, PyTorch, and processor details, and runs a small tensor
operation on the CPU and on MPS. If all checks pass, the script exits with code 0.

## Step 2 – Object Detection Test

### How YOLO11n is used

`src/detector.py` loads the official pretrained **YOLO11n** model
(`models/yolo11n.pt`) with Ultralytics. If the file is missing, Ultralytics
downloads it into `models/` automatically on first run (~5.4 MB). The model
is used for inference only; it is not trained or modified.

| Setting          | Value     |
|------------------|-----------|
| Model            | YOLO11n   |
| Device           | CPU       |
| Input resolution | 640 × 640 |
| Batch size       | 1         |

### Test image

`data/images/bus.jpg`, the official Ultralytics sample image (810×1080,
one bus and several people). The original file is never modified. To
re-download it:

```bash
curl -L -o data/images/bus.jpg https://ultralytics.com/images/bus.jpg
```

### Run the detection test

From the `ECE4300_Object_Detection` folder:

```bash
source .venv/bin/activate
python src/main.py
```

Example output (actual run):

```text
ECE 4300 — Object Detection Test
--------------------------------
Model: YOLO11n
Device: CPU
Input resolution: 640x640

Image: data/images/bus.jpg (810x1080)
Detection completed.

Objects detected (5):
1. Bus — Confidence: 0.94
2. Person — Confidence: 0.89
3. Person — Confidence: 0.88
4. Person — Confidence: 0.86
5. Person — Confidence: 0.62

Output saved: results/detection_test.jpg
```

### Test your own image

Copy the image into `data/images/` and pass its file name:

```bash
python src/main.py my_photo.jpg
```

### Output

The annotated image with bounding boxes and labels is saved to `results/`:

- Default test (`bus.jpg`): `results/detection_test.jpg`
- Your own image: `results/detection_<image name>.jpg` (e.g. `detection_my_photo.jpg`)

Re-running on the same image overwrites its previous output.

## Step 3 – Benchmark Dataset

### What is COCO 2017?

COCO (Common Objects in Context) is the standard public dataset for object
detection, with 80 everyday object classes (people, cars, animals, furniture,
etc.). The 2017 **validation** split has 5,000 images. YOLO11n was trained on
COCO's training split, so the validation images are a fair, standard test
set for it.

### Why a fixed dataset?

A benchmark is only a fair comparison if every configuration does exactly
the same work. Image size and the number of objects in a scene affect
processing time, so if the CPU run and the MPS run (or the 640 and 320 runs)
used different images, differences in timing could come from the images
rather than the hardware. Using one fixed, ordered list of images makes the
workload identical across all configurations and makes every run repeatable.

### Selection and source

- **Source:** official COCO site, <https://cocodataset.org>
  - Images: `http://images.cocodataset.org/val2017/<file_name>`
  - Annotations: `instances_val2017.json` from
    `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
- **Selection rule:** sort all 5,000 val2017 images by image ID and take the
  first 100 (IDs 139 to 9891). There is no randomness, so the result is always the same.
- **Download size:** about 22 MB in total. Only `instances_val2017.json` is read out
  of the 241 MB annotations zip (using HTTP range requests, ~6.5 MB), and
  only the 100 selected images are downloaded (~15 MB). The full dataset is
  not downloaded.
- Images are saved byte-for-byte as downloaded: no resizing or re-encoding.
- **License:** COCO annotations are CC BY 4.0. Images are from Flickr under
  their individual Creative Commons licenses (listed in the annotation file).

### Where things are stored

| Path | Contents |
|------|----------|
| `data/benchmark/images/` | The 100 COCO images (original file names, e.g. `000000000139.jpg`) |
| `data/benchmark/annotations/instances_val2017_subset.json` | COCO-format annotations for only these 100 images (714 objects) |
| `data/benchmark/image_manifest.csv` | The fixed image list |

Your own images in `data/images/` are separate and unaffected.

### The image manifest

`image_manifest.csv` has one row per image, always sorted by `image_id`:

| Column | Meaning |
|--------|---------|
| `image_id` | COCO image ID (links to the annotations) |
| `file_name` | Original COCO file name |
| `width`, `height` | Original image size in pixels |
| `source` | `COCO 2017 val` |
| `relative_path` | Path from the project root |
| `sha256` | Checksum, used to confirm the file content is unchanged |

Future benchmark scripts will read images **from this manifest, in this
order**, never by listing or randomly picking files from the folder.

### Reproduce the dataset

```bash
source .venv/bin/activate
python src/prepare_benchmark_dataset.py
```

Re-running is safe: images already present are not downloaded again.

### Validate the dataset

```bash
python tests/validate_benchmark_dataset.py
```

This runs without YOLO. It checks that:
- exactly 100 images are listed
- every image exists and opens
- sizes and checksums match the manifest
- no extra files are in the folder
- every image has an annotation record

It prints `RESULT: PASS` (exit code 0) or lists the missing or corrupted files.

## Step 4 – CPU Baseline Benchmark

### Purpose

This step measures how fast the M2 **CPU** runs YOLO11n on the fixed
100-image workload. It is the **baseline**: later configurations (MPS GPU,
320×320) will be compared against it using speedup.

The metrics follow the performance concepts from **ECE 4300 Lecture 1**:
- **execution time:** how long a fixed task takes
- **throughput:** work completed per unit time
- **relative performance / speedup:** the ratio of two configurations'
  execution times, calculated in a later step once comparison runs exist

### Configuration

| Setting | Value |
|---------|-------|
| Model | YOLO11n (`models/yolo11n.pt`, pretrained, unmodified) |
| Device | CPU (Apple M2) |
| PyTorch CPU threads | 4 (= M2 performance cores; set explicitly, see below) |
| Input resolution | 640 × 640 |
| Batch size | 1 |
| Confidence / NMS IoU threshold | 0.25 / 0.7 |
| Dataset | 100 COCO 2017 val images, in `image_manifest.csv` order |
| Warm-up | 3 untimed inferences (Step 4 runs; changed to 100 in Step 5, see below) |
| Measured trials | 5 (the model is loaded once and reused) |

### Run the benchmark

From the `ECE4300_Object_Detection` folder:

```bash
source .venv/bin/activate
python -m src.benchmark
```

It takes about 20 seconds. Run the unit tests (no real inference, under 1 second) with:

```bash
python -m unittest tests.test_benchmark -v
```

### What the timer measures

`time.perf_counter()` starts immediately before the first image and stops
immediately after the 100th. Each image is processed sequentially by one
`model.predict()` call.

**Included:**
- YOLO preprocessing (letterbox resize to 640, normalization, conversion to a tensor)
- the neural-network forward pass
- postprocessing (non-maximum suppression and building the Results object)

**Excluded:**
- model loading
- dataset download
- reading the manifest
- reading and decoding the JPEGs from disk (all 100 are preloaded into memory first, ~82 MB)
- warm-up
- printing
- CSV writing
- saving any images

These are **elapsed (wall-clock) processing times for this workload**. They
are not pure CPU time or neural-network-only inference time.

### Formulas

```text
Total execution time (s)       = perf_counter() at end − perf_counter() at start
Average time per image (ms)    = Total execution time / 100 images × 1000
Throughput (images/s)          = 100 images / Total execution time
```

After 5 trials the script reports the mean of each metric and the minimum,
maximum, and standard deviation of the execution time.

### Baseline result (4 Nov 2026, run `20261104-131630-930709`)

| Trial | Time (s) |
|-------|----------|
| 1 | 3.239 |
| 2 | 3.034 |
| 3 | 3.099 |
| 4 | 3.069 |
| 5 | 3.491 |
| **Mean** | **3.186 s → 31.86 ms/image, 31.47 images/s** |

The second run, `20261104-131649-f6e1c8`, gave a mean of 3.187 s (31.40 images/s).

### Raw data

Each run creates a **new** file, `results/raw/cpu-640_<run_id>.csv`, with one
row per trial. The run ID is the date and time plus a random suffix, and the
file is opened in exclusive-create mode, so earlier results are never
overwritten. Columns:

- `run_id`, `timestamp`, `configuration`, `processor`
- `resolution`, `model`, `dataset`, `num_images`, `cpu_threads`
- `trial`, `total_time_s`, `avg_time_ms_per_image`, `throughput_img_per_s`

### Reproducing

1. `python tests/validate_benchmark_dataset.py` must print `RESULT: PASS`.
2. Plug the Mac into power and close other apps, then run `python -m src.benchmark`.
3. Compare the means with the table above. Expect a few percent of variation.

### Known limitations

- **Thread count:** Ultralytics resets PyTorch to `cpu_count − 1` (7)
  threads during its first prediction. The benchmark sets 4 threads
  explicitly **after** warm-up and records the actual value in the CSV.
  In a quick check (best of 2 trials each), 4 threads took 3.00 s, 8 threads 3.13 s and 7 threads 3.41 s, so 4 was the fastest.
- **Mixed cores:** the M2 has 4 performance and 4 efficiency cores. macOS
  decides which cores the threads run on, and this cannot be controlled from Python.
- **Wall-clock timing:** results include OS scheduling and Python overhead,
  and are affected by other running apps, battery/power mode, and
  temperature. A fanless MacBook Air may slow down under sustained load.
- **Trial-to-trial variation:** individual trials varied by about ±7% (for
  example 3.03–3.49 s). That is why 5 trials are averaged.
- **Ultralytics overhead:** `predict()` includes Python-side pre- and
  postprocessing, so times are higher than the raw model forward pass.
- **No accuracy check:** speed is measured without checking detection quality.

## Step 5 – CPU and GPU Benchmark Comparisons

### The four configurations

| Config | Name | Device | Resolution | Role |
|--------|------|--------|------------|------|
| A | `CPU-640` | Apple M2 CPU | 640 × 640 | Baseline |
| B | `CPU-320` | Apple M2 CPU | 320 × 320 | Resolution change on CPU |
| C | `GPU-640` | Apple M2 GPU (MPS) | 640 × 640 | Same workload as A, on the GPU |
| D | `GPU-320` | Apple M2 GPU (MPS) | 320 × 320 | Same workload as B, on the GPU |

Everything else is identical for all four: the same `models/yolo11n.pt`
weights, the same 100 images in manifest order, batch size 1, confidence
0.25, NMS IoU 0.7, 4 PyTorch CPU threads, 100 warm-up images, and 5 measured
trials.

### Configuration file

All settings are in **`config/benchmark_config.json`** (not hardcoded in
Python). The `common` block holds the shared settings (model file, dataset
manifest, number of trials, warm-up, threads, thresholds, CSV output
folders), and each configuration sets only `device` (`cpu` or `mps`) and
`image_size` (`640` or `320`). `src/benchmark_config.py` loads and checks the
file; it rejects unknown devices, resolutions other than 640/320, and batch
sizes other than 1.

### Commands

From the `ECE4300_Object_Detection` folder, with `.venv` activated:

```bash
# One configuration at a time
python -m src.benchmark                  # A: CPU 640x640 (baseline, same as Step 4)
python -m src.benchmark --config B       # B: CPU 320x320   (or --config CPU-320)
python -m src.benchmark --config C       # C: GPU 640x640   (or --config GPU-640)
python -m src.benchmark --config D       # D: GPU 320x320   (or --config GPU-320)

# All four, one after another, then the summary and speedup table
python -m src.run_all_benchmarks
python -m src.run_all_benchmarks --configs C D     # a subset

# Rebuild the summary from CSVs already in results/raw/
python -m src.compare_results                      # latest run of each configuration
python -m src.compare_results --session <id>       # the runs from one run_all session

# Unit tests (no real inference, no GPU needed, about 1 second)
python -m unittest tests.test_benchmark tests.test_configurations -v
```

`run_all_benchmarks` starts each configuration in its own Python process,
**strictly one at a time** (never in parallel), so memory is fully released
between runs on the 8 GB Mac. If one configuration fails, the error is
recorded and the others still run. Each configuration processes the
100 images 6 times (1 warm-up pass + 5 trials), so expect a few minutes in total.

### GPU (MPS) execution

Before any GPU work, `src/benchmark.py`:

1. Checks `torch.backends.mps.is_built()` and `torch.backends.mps.is_available()`.
   If either is false, it prints an explanation and exits with code 2.
   **It never falls back to the CPU.** This check matters because Ultralytics
   itself silently switches to the CPU when MPS is requested but unavailable.
2. Refuses to run if `PYTORCH_ENABLE_MPS_FALLBACK=1` is set, because that
   setting lets PyTorch run unsupported operations on the CPU without warning.
3. Runs one untimed inference and confirms that the predictor, the model
   weights, and the output boxes are all on `mps`. It checks this again after
   the trials. The same check is applied to the CPU configurations.

**Synchronization:** MPS runs GPU kernels asynchronously, so `predict()` can
return before the GPU is finished. Inside the timed loop,
`torch.mps.synchronize()` is called after **every** image, so each image is
completely processed before the next one starts (exactly as on the CPU) and
the timer never stops while GPU work is still queued. On the CPU the call is
skipped because CPU work is already finished when `predict()` returns.

### What is timed (the same for all four configurations)

The timer measures **elapsed workload processing time**: the complete
sequential loop of 100 `model.predict()` calls on images that are already
in memory.

- **Included:** YOLO preprocessing (letterbox resize, normalization, copy
  to the device), the neural-network forward pass, postprocessing (NMS,
  Results objects), and on the GPU, the wait for queued GPU work to finish.
- **Excluded:** model loading, dataset download, manifest reading, JPEG
  decoding, the device check, warm-up, console output, CSV writing, and
  saving images.

This is not neural-network-only time, and no configuration is timed
differently from another. The CSV `measurement` column is
`elapsed_workload_time` for every run.

### Changes to the Step 4 baseline methodology

The timing boundary for CPU-640 is **unchanged**: the same loop, the same
`perf_counter()` positions, and the same `predict()` settings. The added
synchronize call does nothing on the CPU. Two things changed, for all four
configurations equally:

1. **Warm-up: 3 → 100 images (one full untimed pass).** With
   `predict()`, Ultralytics letterboxes each image to the smallest
   stride-32 rectangle (`rect=True`), so the model input shape depends on
   the image's aspect ratio. The 100 images produce **14 different input
   shapes at 640 and 9 at 320**. On MPS each new shape triggers one-time GPU
   kernel/graph setup. With only 3 warm-up images, most of that one-time
   setup would fall inside measured trial 1. A full pass covers every shape
   before timing starts, and it is applied to the CPU too so all four
   configurations are prepared the same way. Warm-up is outside the timer, so
   the measured region is unchanged. To return to the Step 4 setting, set
   `"num_warmup": 3` in the config file.
2. **More CSV columns** (listed below). The Step 4 columns are all still present.

Because of change 1, compare the four configurations from the **same Step 5
session**, not against the Step 4 CPU-640 run (3.186 s).
`compare_results` warns when the selected runs used different warm-up
settings, model weights, or image lists.

### Formulas

```text
Total execution time (s)       = perf_counter() at end − perf_counter() at start   (one trial)
Average time per image (ms)    = Total execution time / 100 images × 1000
Throughput (images/s)          = 100 images / Total execution time
Mean execution time (s)        = average of the 5 trial execution times

Speedup = Baseline execution time / Comparison execution time   (using the mean times)
```

The summary reports the mean per-image time and mean throughput as the
average of the five per-trial values, the same as Step 4.

### Speedup comparisons

| Comparison | Formula | What it shows |
|------------|---------|---------------|
| Resolution effect on CPU | CPU-640 time / CPU-320 time | Effect of a smaller input on the same hardware |
| Resolution effect on GPU | GPU-640 time / GPU-320 time | Effect of a smaller input on the same hardware |
| CPU vs. GPU at 640×640 | CPU-640 time / GPU-640 time | Same workload, different processor |
| CPU vs. GPU at 320×320 | CPU-320 time / GPU-320 time | Same workload, different processor |

- **Speedup > 1:** the comparison configuration was faster.
- **Speedup < 1:** the comparison configuration was **slower** (for example,
  0.80 means it took 1/0.80 = 1.25× as long). The GPU is not assumed to be
  faster: with batch size 1 and a small model, GPU launch and
  synchronization overhead, and CPU-side pre/postprocessing, can cancel
  out the GPU's advantage.
- Only the **CPU vs. GPU** rows are same-workload hardware speedups. The
  **resolution** rows compare different workloads (320×320 has about 4× fewer
  input pixels), so they show the effect of reducing resolution, not
  faster hardware.

### Where results are stored

| File | Contents |
|------|----------|
| `results/raw/<config>_<run_id>.csv` | One row per measured trial (e.g. `gpu-320_20261112-101500-a1b2c3.csv`) |
| `results/raw/failed_runs.csv` | One row per failed run: configuration, device, and the error message (no timing data) |
| `results/summary/summary_<time>_<session>.csv` | One row per configuration: status, mean/std/min/max time, ms/image, images/s |
| `results/summary/speedup_<time>_<session>.csv` | The four speedups, the times used, and a plain-language interpretation |

Raw CSV columns: `run_id`, `session_id`, `timestamp`, `configuration`,
`config_id`, `device`, `processor`, `resolution`, `model`, `model_sha256`,
`dataset`, `image_list_sha256`, `num_images`, `cpu_threads`,
`warmup_images`, `measurement`, `trial`, `total_time_s`,
`avg_time_ms_per_image`, `throughput_img_per_s`.

`model_sha256` and `image_list_sha256` (a hash of the ordered image names
and checksums) show that every configuration used the same weights and the same
100 images in the same order. All result files are created in
exclusive-create mode, so earlier sessions are never overwritten. Step 4
CSVs (without the new columns) are still read correctly.

### Results

**Not measured yet.** Step 5 was implemented and tested on a different
computer that has no Apple GPU. The unit tests pass, and CPU-640 and CPU-320
were run end to end there, but those numbers come from a different processor
and are not project results. Run `python -m src.run_all_benchmarks` on the
Mac and record the summary and speedup tables here.

### Known limitations

- **macOS background activity:** Spotlight indexing, iCloud/OneDrive sync,
  Time Machine, browser tabs, and other apps compete for the CPU, the GPU,
  and memory bandwidth. Close other apps, plug in power, and run all four
  configurations in one session so they share the same conditions.
- **Unified memory:** the M2 CPU and GPU share the same 8 GB of memory and
  its bandwidth. There is no separate GPU memory and no PCIe copy, so
  "moving data to the GPU" is cheap. However, memory pressure from other
  apps (swapping) slows both devices.
- **CPU work inside GPU runs:** in the GPU configurations, Ultralytics
  preprocessing and parts of postprocessing still run on the CPU (4
  threads), so GPU times are not pure GPU compute time.
- **Per-image synchronization:** calling `torch.mps.synchronize()` after
  every image (batch size 1) adds a small fixed cost per image on the GPU.
  This is intentional: each image is fully finished, as on the CPU.
- **Thermal throttling:** a fanless MacBook Air can slow down during
  long runs. Later configurations in a session may run slightly hotter.
- **Variable input shapes:** with `rect=True`, the actual model input
  is at most 640×640 (or 320×320), not always exactly that size. The workload is still
  identical across devices because the images and their order are fixed.
- **Wall-clock timing and trial-to-trial variation:** as in Step 4.
- **No accuracy check:** 320×320 is faster but may detect fewer objects.
  Detection accuracy is not evaluated in this step.

## Step 6 – Performance Analysis and Graphs

### Purpose

`src/analyze_results.py` turns the Step 5 measurements into a summary table,
speedups, a short written interpretation, and three graphs. It only reads
measured data. It does not run YOLO, and it never changes the raw CSVs.

### Run the analysis

On the Mac, after Step 5 has been measured:

```bash
source .venv/bin/activate
python -m src.run_all_benchmarks      # Step 5: collect A–D in one session (if not done yet)
python -m src.analyze_results         # Step 6: analyze the latest complete session
python -m src.analyze_results --session <session id>   # or a specific session

# Calculation tests (known math examples, no real data, a few seconds)
python -m unittest tests.test_analysis -v
```

### Which data is used

The analysis uses **one complete benchmark session**: the most recent
`run_all_benchmarks` session that has all four configurations (or the one
given with `--session`). Runs from different sessions are never combined,
because they may have been measured under different conditions.

Before calculating anything, the script checks that:

- all four configurations are present, each with at least 5 trials
- every trial time is a positive number and every trial processed 100 images
- the stored ms/image value of each trial matches its measured time
- every run used the same model weights, image list and order, dataset,
  warm-up, and CPU thread count (from the hashes and settings in the CSVs)
- every trial was measured with the same timing boundary (`measurement` =
  `elapsed_workload_time`) and the expected device and resolution
- batch size is 1 (enforced by the configuration file)

If any check fails, the **whole run is rejected**: no trial is silently
skipped, and nothing is written. The script then prints what is missing and
the command to collect it (`python -m src.run_all_benchmarks`).

### Formulas

```text
Mean execution time (s)          = sum of trial execution times / number of trials
Average processing time (ms/img) = mean execution time / number of images × 1000
Trial throughput (images/s)      = number of images / trial execution time
Mean throughput (images/s)       = average of the trial throughputs
Minimum / maximum time (s)       = shortest / longest trial
Range (s)                        = maximum time − minimum time
Speedup                          = reference execution time / comparison execution time
```

Speedups use the mean execution times. Mean throughput is the average of
the five trial throughputs, so it can differ slightly from
100 / mean time.

| Comparison | Reference → Comparison |
|------------|------------------------|
| Resolution effect on CPU | CPU 640×640 → CPU 320×320 |
| Resolution effect on GPU | GPU 640×640 → GPU 320×320 |
| CPU vs. GPU at 640×640 | CPU 640×640 → GPU 640×640 |
| CPU vs. GPU at 320×320 | CPU 320×320 → GPU 320×320 |

A speedup **above 1** means the comparison configuration was faster. A
speedup **below 1** means it was **slower**, and it is reported that way,
not as an improvement.

### Output files

Files are named after the session ID, so analyses of different sessions
never overwrite each other. Re-running the analysis on the same session
recreates identical files from the same data.

| File | Contents |
|------|----------|
| `results/summaries/performance_summary_<session>.csv` | Per configuration: name, processor, resolution, trials, images per trial, mean time, ms/image, mean throughput, min, max, range |
| `results/summaries/speedup_<session>.csv` | The four speedups, the two mean times used for each, which configuration was faster, and an interpretation |
| `results/summaries/analysis_report_<session>.md` | Short written interpretation (see below) |
| `results/graphs/execution_time_<session>.png` | Graph 1 |
| `results/graphs/throughput_<session>.png` | Graph 2 |
| `results/graphs/cpu_gpu_speedup_<session>.png` | Graph 3 |

Graphs are 300 dpi PNGs on a white background, suitable for the report.
CPU bars are blue and GPU bars are orange in both bar charts.

### How to read the graphs

- **Graph 1 – Average Object Detection Execution Time:** mean time (seconds)
  to process the 100-image workload. **Lower is faster.** The shortest bar is
  the configuration that finishes the workload first. Whiskers show the
  fastest and slowest of the 5 trials; short whiskers mean consistent trials.
- **Graph 2 – Object Detection Throughput Comparison:** images processed per
  second (mean of the trial throughputs). **Higher is better.** Throughput is
  the reciprocal view of Graph 1: the shortest bar there should be the
  tallest bar here.
- **Graph 3 – CPU-to-GPU Speedup by Resolution:** CPU mean time / GPU mean
  time at each resolution. The dashed line is 1.0× (equal performance).
  **Above the line, the GPU was faster; below it, the CPU was faster.** It
  compares the same workload on two processors, so it shows only the
  hardware difference, not the resolution effect.

### Written report

`analysis_report_<session>.md` states, from the measured data, which
configuration had the shortest mean time and the highest throughput, how
reducing resolution changed the time on each processor, how CPU and GPU
compared, and whether trials were consistent. A configuration counts as
consistent when its range is at most 10% of its mean (set by
`consistency_threshold_pct` in the config file). Unexpected results, such as
the GPU being slower or trial 1 being much slower than the others, are
listed separately. Possible reasons are in their own section, clearly
labelled **hypotheses (not measured)**, because this project measures only
total elapsed time.

### Results depend on the conditions

All numbers apply only to this Mac, these software versions, and the
conditions during the session (power source, temperature, background apps,
memory pressure). Another run can give slightly different values, and
other hardware, model sizes, or batch sizes may give different conclusions.
Report the session ID together with the results.
