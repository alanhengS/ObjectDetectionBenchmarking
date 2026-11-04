"""
ECE 4300 - Step 2: Single-image object detection test.

Loads the pretrained YOLO11n model, runs detection on one test image using
the CPU, prints the detected objects, and saves an annotated image with
bounding boxes to results/.

Run from the project root:
    python src/main.py                     # uses data/images/bus.jpg
    python src/main.py my_photo.jpg        # any image in data/images/
"""

import sys
from pathlib import Path

# Check required libraries up front so the error message is clear
try:
    import cv2
    import torch  # noqa: F401  (imported to confirm PyTorch is available)
    from detector import (
        DEVICE, IMAGE_SIZE, PROJECT_ROOT,
        load_model, detect, extract_detections,
    )
except ImportError as e:
    print(f"ERROR: A required library is missing: {e}")
    print("Activate the environment with 'source .venv/bin/activate' and "
          "run 'pip install -r requirements.txt'.")
    sys.exit(1)

# Input and output locations
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_IMAGE = "bus.jpg"


def get_paths():
    """
    Choose the input image and output file.

    With no argument, the default bus.jpg test is used and the result is saved
    as results/detection_test.jpg. With an argument (e.g. 'cat.jpg'), that file
    from data/images/ is used and saved as results/detection_<name>.jpg.
    """
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if not image_path.exists():           # allow just the file name
            image_path = IMAGES_DIR / image_path.name
        output_path = RESULTS_DIR / f"detection_{image_path.stem}.jpg"
    else:
        image_path = IMAGES_DIR / DEFAULT_IMAGE
        output_path = RESULTS_DIR / "detection_test.jpg"
    return image_path.resolve(), output_path


def read_image(path):
    """Read an image with OpenCV, with clear errors for common problems."""
    if not path.exists():
        raise FileNotFoundError(
            f"Test image not found: {path}\n"
            "Download it with:\n"
            "  curl -L -o data/images/bus.jpg https://ultralytics.com/images/bus.jpg"
        )

    image = cv2.imread(str(path))  # returns None if the format is unsupported
    if image is None:
        raise ValueError(
            f"Could not read '{path.name}'. The file may be corrupted or in an "
            "unsupported format (use JPG or PNG)."
        )
    return image


def main():
    print("ECE 4300 — Object Detection Test")
    print("--------------------------------")
    print("Model: YOLO11n")
    print(f"Device: {DEVICE.upper()}")
    print(f"Input resolution: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print()

    image_path, output_path = get_paths()

    # 1. Load the model (downloads weights into models/ if needed)
    try:
        model = load_model()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    # 2. Read the test image (the original file is never modified)
    try:
        image = read_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    height, width = image.shape[:2]
    print(f"Image: {image_path.name} ({width}x{height})")

    # 3. Run detection on the CPU
    try:
        result = detect(model, image)
    except Exception as e:
        print(f"ERROR: Detection failed: {e}")
        return 1
    print("Detection completed.")
    print()

    # 4. Print each detected object with its confidence score
    detections = extract_detections(result)
    if detections:
        print(f"Objects detected ({len(detections)}):")
        for i, (name, conf) in enumerate(detections, start=1):
            print(f"{i}. {name.capitalize()} — Confidence: {conf:.2f}")
    else:
        print("No objects were detected in this image.")
    print()

    # 5. Draw bounding boxes and save the annotated copy to results/
    annotated = result.plot()  # BGR image with boxes and labels drawn
    output_path.parent.mkdir(exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        print(f"ERROR: Could not save output image to {output_path}")
        return 1
    print(f"Output saved: {output_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
