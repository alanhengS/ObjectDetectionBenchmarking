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

No MPS benchmarking, 320×320 runs, or speedup calculations have been added yet.

## Project Structure

```text
ECE4300_Object_Detection/
├── config/        # Configuration files (future)
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
│   └── benchmark.py                  # Step 4 CPU baseline benchmark
├── results/       # Output images
│   └── raw/       # Benchmark CSVs (one file per run)
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
| Warm-up | 3 untimed inferences |
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

### Next: Step 5

- The same benchmark on the GPU (MPS), and at 320×320
- Speedup calculations relative to this CPU baseline

## Planned Benchmark Configurations

| # | Device      | Input Resolution |
|---|-------------|------------------|
| 1 | CPU         | 640 × 640        |
| 2 | CPU         | 320 × 320        |
| 3 | GPU (MPS)   | 640 × 640        |
| 4 | GPU (MPS)   | 320 × 320        |
