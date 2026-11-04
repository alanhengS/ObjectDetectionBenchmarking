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

No model weights, datasets, detection code, or benchmarks have been added yet.

## Project Structure

```text
ECE4300_Object_Detection/
├── config/        # Configuration files (future)
├── data/images/   # Test images (future)
├── models/        # YOLO model weights (future)
├── src/           # Source code
├── results/       # Benchmark output (future)
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

## Planned Benchmark Configurations

| # | Device      | Input Resolution |
|---|-------------|------------------|
| 1 | CPU         | 640 × 640        |
| 2 | CPU         | 320 × 320        |
| 3 | GPU (MPS)   | 640 × 640        |
| 4 | GPU (MPS)   | 320 × 320        |
