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

No datasets, timing measurements, or MPS inference have been added yet.

## Project Structure

```text
ECE4300_Object_Detection/
├── config/        # Configuration files (future)
├── data/images/   # Test image (bus.jpg)
├── models/        # YOLO model weights (yolo11n.pt)
├── src/
│   ├── detector.py  # Loads YOLO11n and runs inference
│   └── main.py      # Step 2 single-image detection test
├── results/       # Output images
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

### Next: Step 3

- Run the same detection on the GPU using Apple MPS (`device="mps"`)
- Prepare a small set of test images
- Add 320×320 as a second input resolution
- Then move on to timing measurements and benchmarking

## Planned Benchmark Configurations

| # | Device      | Input Resolution |
|---|-------------|------------------|
| 1 | CPU         | 640 × 640        |
| 2 | CPU         | 320 × 320        |
| 3 | GPU (MPS)   | 640 × 640        |
| 4 | GPU (MPS)   | 320 × 320        |
