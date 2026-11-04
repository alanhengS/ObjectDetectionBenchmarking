"""
ECE 4300 - Step 3: Validate the fixed COCO 2017 benchmark dataset.

Checks (no YOLO inference is performed):
  - The manifest lists exactly 100 images, in image-ID order.
  - Every manifest file exists in data/benchmark/images/ and opens with OpenCV.
  - Each image's size and SHA-256 checksum match the manifest (content unchanged).
  - No extra, unlisted images are in the folder.
  - Every image has a matching record in the annotation file.

Run from the project root:
    python tests/validate_benchmark_dataset.py
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
IMAGES_DIR = BENCHMARK_DIR / "images"
MANIFEST_PATH = BENCHMARK_DIR / "image_manifest.csv"
ANNOTATIONS_PATH = BENCHMARK_DIR / "annotations" / "instances_val2017_subset.json"
EXPECTED_COUNT = 100
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    print("ECE 4300 — Benchmark Dataset Validation")
    print("---------------------------------------")

    for path in (MANIFEST_PATH, ANNOTATIONS_PATH):
        if not path.exists():
            print(f"ERROR: Missing {path.relative_to(PROJECT_ROOT)}")
            print("Run: python src/prepare_benchmark_dataset.py")
            return 1

    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(ANNOTATIONS_PATH) as f:
        coco = json.load(f)

    annotated_image_ids = {img["id"] for img in coco["images"]}
    object_counts = {}
    for ann in coco["annotations"]:
        object_counts[ann["image_id"]] = object_counts.get(ann["image_id"], 0) + 1

    problems = []

    # Manifest-level checks
    if len(rows) != EXPECTED_COUNT:
        problems.append(f"Manifest has {len(rows)} rows, expected {EXPECTED_COUNT}")
    ids = [int(r["image_id"]) for r in rows]
    if ids != sorted(ids):
        problems.append("Manifest is not sorted by image_id")
    if len(set(ids)) != len(ids):
        problems.append("Manifest contains duplicate image IDs")

    # Per-image checks
    missing, corrupted, changed, no_annotation = [], [], [], []
    valid = 0
    for row in rows:
        path = PROJECT_ROOT / row["relative_path"]
        name = row["file_name"]

        if path.name != name:
            problems.append(f"{name}: relative_path does not match file_name")
        if not path.exists():
            missing.append(name)
            continue

        image = cv2.imread(str(path))
        if image is None:
            corrupted.append(name)
            continue

        height, width = image.shape[:2]
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if (width, height) != (int(row["width"]), int(row["height"])) or checksum != row["sha256"]:
            changed.append(name)
            continue

        if int(row["image_id"]) not in annotated_image_ids:
            no_annotation.append(name)
            continue

        valid += 1

    # Files in the folder that are not in the manifest
    listed = {r["file_name"] for r in rows}
    extra = sorted(
        p.name for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.name not in listed
    )

    # Report
    images_with_objects = sum(1 for i in ids if object_counts.get(i, 0) > 0)
    print(f"Manifest entries         : {len(rows)}")
    print(f"Annotation records       : {len(coco['annotations'])} objects, "
          f"{len(coco['categories'])} categories")
    print(f"Images with >=1 object   : {images_with_objects}")
    print(f"Images with 0 objects    : {len(ids) - images_with_objects}")
    print(f"Missing images           : {len(missing)} {missing or ''}")
    print(f"Corrupted/unreadable     : {len(corrupted)} {corrupted or ''}")
    print(f"Size/checksum mismatch   : {len(changed)} {changed or ''}")
    print(f"No annotation record     : {len(no_annotation)} {no_annotation or ''}")
    print(f"Unlisted extra images    : {len(extra)} {extra or ''}")
    for p in problems:
        print(f"Problem                  : {p}")
    print("---------------------------------------")
    print(f"Valid images: {valid}/{EXPECTED_COUNT}")

    ok = (valid == EXPECTED_COUNT and not problems and not extra)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
