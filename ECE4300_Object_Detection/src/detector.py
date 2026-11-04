"""
YOLO11n detector wrapper for the ECE 4300 Object Detection project.

Responsibilities:
  - Load the official pretrained YOLO11n model (downloaded into models/ on first use).
  - Run inference on a single image with a fixed configuration.

Step 2 configuration: CPU only, 640x640 input, batch size 1.
"""

from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "Ultralytics is not installed. Activate the virtual environment "
        "(source .venv/bin/activate) and run: pip install -r requirements.txt"
    ) from e

# Project paths (resolved from this file so the script works from any directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Fixed configuration for this step
MODEL_NAME = "yolo11n.pt"
DEVICE = "cpu"
IMAGE_SIZE = 640  # model input is 640x640 (Ultralytics letterboxes the image)
BATCH_SIZE = 1    # one image per inference call


def load_model(model_name=MODEL_NAME):
    """
    Load a pretrained YOLO model from the models/ directory.

    If the weights file is not present, Ultralytics downloads the official
    release into models/ automatically. The model is only used for
    inference; it is never trained or modified.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    weights_path = MODELS_DIR / model_name

    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        raise RuntimeError(
            f"Could not load or download model weights '{weights_path}'. "
            f"Check your internet connection, or place {model_name} in "
            f"{MODELS_DIR} manually.\nDetails: {e}"
        ) from e

    return model


def detect(model, image):
    """
    Run object detection on a single image (a NumPy BGR array from OpenCV).

    Returns the Ultralytics Results object for that image.
    """
    results = model.predict(
        source=image,
        device=DEVICE,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        verbose=False,  # we print our own summary in main.py
    )
    return results[0]  # one image in -> one Results object out


def extract_detections(result):
    """
    Convert a Results object into a simple list of (class_name, confidence)
    tuples, sorted from highest to lowest confidence.
    """
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        confidence = float(box.conf.item())
        detections.append((class_name, confidence))

    detections.sort(key=lambda d: d[1], reverse=True)
    return detections
