"""
ECE 4300 - Step 7: Detection-quality check at 640x640 and 320x320.

This is an ACCURACY check, kept separate from the computer-architecture
performance measurements: no timing is recorded here, and these results are
saved apart from results/raw/.

It answers one question: does reducing the input resolution from 640x640 to
320x320 change how well the same pretrained YOLO11n detects objects in the
same fixed 100 COCO 2017 validation images?

Method (Ultralytics' built-in validation, the standard YOLO procedure):
  1. Convert the existing COCO ground-truth annotations
     (data/benchmark/annotations/instances_val2017_subset.json) into YOLO
     label files in data/benchmark/labels/ (derived files, rewritten on every
     run). Crowd boxes (iscrowd = 1) are skipped, as in Ultralytics' own COCO
     converter.
  2. Write a dataset file, results/accuracy/coco_benchmark100.yaml.
  3. Run model.val() once at imgsz=640 and once at imgsz=320 with the same
     weights and settings (batch 1, IoU 0.7, conf 0.001).
  4. Save precision, recall, mAP50, and mAP50-95 to a NEW file,
     results/accuracy/accuracy_<time>.csv.

conf = 0.001 is Ultralytics' validation default: mAP needs the full
precision-recall curve, so low-confidence boxes must be kept. (The benchmark
itself uses conf = 0.25, which is fine for timing.) The model is not trained
or changed. No data is downloaded. mAP here is Ultralytics' implementation,
which can differ slightly from the official pycocotools COCO evaluator.

Run from the project root (on the Mac, with .venv active):
    python -m src.evaluate_accuracy                # CPU (accuracy does not depend on the device)
    python -m src.evaluate_accuracy --device mps
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from src.benchmark_config import CONFIG_PATH, PROJECT_ROOT, ConfigError, load_all_configs

ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "benchmark" / "annotations" / "instances_val2017_subset.json"
OUT_DIR = PROJECT_ROOT / "results" / "accuracy"
VAL_CONF = 0.001          # Ultralytics validation default (full precision-recall curve)

CSV_COLUMNS = [
    "timestamp", "image_size", "device", "model", "model_sha256", "images",
    "ground_truth_boxes", "crowd_boxes_skipped", "batch_size", "conf_threshold",
    "iou_threshold", "precision", "recall", "mAP50", "mAP50_95",
    "ultralytics_version", "torch_version",
]


# ----------------------------------------------------------------------------
# COCO -> YOLO labels (standard library only, unit-tested)
# ----------------------------------------------------------------------------
def coco_to_yolo(coco):
    """
    Return (labels, class_names, crowd_skipped).

    labels: {file_name: ["class cx cy w h", ...]} with coordinates normalised
    to 0-1. Class index = position of the COCO category when sorted by ID,
    which is the 80-class order YOLO11n was trained with.
    """
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    class_index = {c["id"]: i for i, c in enumerate(categories)}
    images = {img["id"]: img for img in coco["images"]}
    labels = {img["file_name"]: [] for img in coco["images"]}
    crowd = 0
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            crowd += 1
            continue
        x, y, w, h = ann["bbox"]                       # COCO: top-left x, y, width, height
        if w <= 0 or h <= 0:
            continue
        img = images[ann["image_id"]]
        iw, ih = img["width"], img["height"]
        labels[img["file_name"]].append(
            f"{class_index[ann['category_id']]} {(x + w / 2) / iw:.6f} {(y + h / 2) / ih:.6f} "
            f"{w / iw:.6f} {h / ih:.6f}")
    return labels, [c["name"] for c in categories], crowd


def write_labels(labels, file_names, labels_dir):
    """One .txt per benchmark image (empty file = no objects). Returns the box count."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    boxes = 0
    for name in file_names:
        lines = labels.get(name, [])
        (labels_dir / f"{Path(name).stem}.txt").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8")
        boxes += len(lines)
    return boxes


def write_dataset_yaml(path, dataset_root, class_names):
    """Ultralytics dataset file. Names are JSON-quoted (valid YAML)."""
    lines = [f"path: {json.dumps(str(dataset_root))}", "train: images", "val: images", "names:"]
    lines += [f"  {i}: {json.dumps(name)}" for i, name in enumerate(class_names)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="mAP50 of YOLO11n at 640 and 320 on the 100 images.")
    p.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    p.add_argument("--config-file", type=Path, default=CONFIG_PATH)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        configs = load_all_configs(args.config_file)
    except ConfigError as e:
        print(f"ERROR: {e}")
        return 1
    cfg = next(iter(configs.values()))
    sizes = sorted({c.image_size for c in configs.values()}, reverse=True)      # 640, 320

    with open(cfg.manifest_path, newline="") as f:
        manifest = list(csv.DictReader(f))
    dataset_root = cfg.manifest_path.parent                  # data/benchmark
    images_dir = dataset_root / "images"
    on_disk = sorted(p.name for p in images_dir.glob("*.jpg"))
    if on_disk != sorted(r["file_name"] for r in manifest):
        print(f"ERROR: {images_dir} does not hold exactly the {len(manifest)} manifest images.")
        print("Run: python tests/validate_benchmark_dataset.py")
        return 1

    coco = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    labels, names, crowd = coco_to_yolo(coco)
    boxes = write_labels(labels, [r["file_name"] for r in manifest], dataset_root / "labels")
    (dataset_root / "labels.cache").unlink(missing_ok=True)   # force Ultralytics to re-read labels
    yaml_path = write_dataset_yaml(OUT_DIR / "coco_benchmark100.yaml", dataset_root, names)

    # Heavy imports only now, so the functions above work without them (tests).
    import torch
    import ultralytics
    from src.detector import load_model

    model = load_model(cfg.model_path.name)
    model_names = [model.names[i] for i in range(len(model.names))]
    if model_names != names:
        print("ERROR: The model's class names do not match the COCO categories; "
              "labels would be misaligned. Nothing was evaluated.")
        return 1

    print("ECE 4300 — Detection Quality Check (accuracy only, not a performance measurement)")
    print(f"Model YOLO11n ({cfg.model_path.name}), device {args.device}, batch {cfg.batch_size}, "
          f"conf {VAL_CONF}, IoU {cfg.iou_threshold}")
    print(f"{len(manifest)} images, {boxes} ground-truth boxes ({crowd} crowd boxes skipped)\n")

    model_sha = hashlib.sha256(cfg.model_path.read_bytes()).hexdigest()[:16]
    now = datetime.now()
    rows = []
    for size in sizes:
        model = load_model(cfg.model_path.name)               # fresh, identical model per size
        metrics = model.val(
            data=str(yaml_path), imgsz=size, batch=cfg.batch_size, device=args.device,
            conf=VAL_CONF, iou=cfg.iou_threshold, plots=False, save_json=False,
            verbose=False, workers=0, project=str(OUT_DIR / "runs"), name=f"val_{size}",
            exist_ok=True,
        )
        box = metrics.box
        rows.append({
            "timestamp": now.isoformat(timespec="seconds"), "image_size": size,
            "device": args.device, "model": cfg.model_path.name, "model_sha256": model_sha,
            "images": len(manifest), "ground_truth_boxes": boxes, "crowd_boxes_skipped": crowd,
            "batch_size": cfg.batch_size, "conf_threshold": VAL_CONF,
            "iou_threshold": cfg.iou_threshold,
            "precision": f"{box.mp:.4f}", "recall": f"{box.mr:.4f}",
            "mAP50": f"{box.map50:.4f}", "mAP50_95": f"{box.map:.4f}",
            "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"accuracy_{now.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(out, "x", newline="") as f:                     # never overwrite earlier results
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'Input':<10}{'Precision':>10}{'Recall':>10}{'mAP50':>10}{'mAP50-95':>10}")
    for r in rows:
        print(f"{r['image_size']}x{r['image_size']:<6}{r['precision']:>10}{r['recall']:>10}"
              f"{r['mAP50']:>10}{r['mAP50_95']:>10}")
    print(f"\nSaved: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
