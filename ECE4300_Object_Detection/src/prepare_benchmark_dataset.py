"""
ECE 4300 - Step 3: Prepare the fixed COCO 2017 benchmark dataset.

Selects the first 100 COCO 2017 validation images (sorted by image ID),
downloads only those images from the official COCO server, saves their
annotations, and writes a manifest describing the fixed workload.

Download size is kept small:
  - Annotations: only instances_val2017.json is read out of the official
    241 MB annotations zip, using HTTP range requests (~8 MB transferred).
  - Images: 100 individual JPEGs (~16 MB total).

Run from the project root:
    python src/prepare_benchmark_dataset.py

Re-running is safe: images that already exist are not downloaded again.
"""

import csv
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
IMAGES_DIR = BENCHMARK_DIR / "images"
ANNOTATIONS_DIR = BENCHMARK_DIR / "annotations"
MANIFEST_PATH = BENCHMARK_DIR / "image_manifest.csv"
ANNOTATIONS_PATH = ANNOTATIONS_DIR / "instances_val2017_subset.json"

NUM_IMAGES = 100
SOURCE_NAME = "COCO 2017 val"

# Official COCO 2017 sources
ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{file_name}"

MANIFEST_COLUMNS = [
    "image_id", "file_name", "width", "height", "source", "relative_path", "sha256",
]


# ----------------------------------------------------------------------------
# Reading one file out of a remote zip without downloading the whole zip
# ----------------------------------------------------------------------------
class HttpRangeFile(io.RawIOBase):
    """
    A read-only, seekable file object backed by HTTP range requests.

    Python's zipfile only needs seek/tell/read, so wrapping this lets us
    open the remote zip and fetch just the bytes of the one member we need.
    """

    def __init__(self, url):
        self.url = url
        self.pos = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError("Server does not support partial downloads.")
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size or len(buffer) == 0:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def load_coco_val_annotations():
    """Read instances_val2017.json from the official zip via range requests."""
    print(f"Reading {ANNOTATIONS_MEMBER} from the official COCO annotations zip...")
    remote = io.BufferedReader(HttpRangeFile(ANNOTATIONS_ZIP_URL), buffer_size=4 * 1024 * 1024)
    with zipfile.ZipFile(remote) as archive:
        info = archive.getinfo(ANNOTATIONS_MEMBER)
        print(f"  compressed: {info.compress_size / 1e6:.1f} MB, "
              f"uncompressed: {info.file_size / 1e6:.1f} MB")
        with archive.open(info) as member:
            return json.load(member)


# ----------------------------------------------------------------------------
# Selection, download and output
# ----------------------------------------------------------------------------
def select_images(coco):
    """Deterministic selection: sort all val2017 images by ID, take the first N."""
    images = sorted(coco["images"], key=lambda img: img["id"])
    return images[:NUM_IMAGES]


def build_annotation_subset(coco, selected):
    """Keep only the annotation records for the selected images (COCO format)."""
    selected_ids = {img["id"] for img in selected}
    annotations = [a for a in coco["annotations"] if a["image_id"] in selected_ids]
    annotations.sort(key=lambda a: (a["image_id"], a["id"]))
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": selected,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_image(file_name, retries=3):
    """Download one image unchanged (raw bytes are written exactly as served)."""
    destination = IMAGES_DIR / file_name
    if destination.exists() and destination.stat().st_size > 0:
        return False  # already present
    url = IMAGE_URL_TEMPLATE.format(file_name=file_name)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            destination.write_bytes(data)
            return True
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {e}") from e
            time.sleep(2 * attempt)


def write_manifest(selected):
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for img in selected:  # already sorted by image_id
            path = IMAGES_DIR / img["file_name"]
            writer.writerow({
                "image_id": img["id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "source": SOURCE_NAME,
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_of(path),
            })


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        coco = load_coco_val_annotations()
    except Exception as e:
        print(f"ERROR: Could not read COCO annotations from {ANNOTATIONS_ZIP_URL}\n{e}")
        print("The official source may be unavailable. No substitute dataset was used.")
        return 1

    selected = select_images(coco)
    print(f"Selected {len(selected)} images "
          f"(IDs {selected[0]['id']} to {selected[-1]['id']}, sorted by image ID).")

    subset = build_annotation_subset(coco, selected)
    del coco  # free the full annotation file from memory
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(subset, f)
    print(f"Saved {len(subset['annotations'])} annotation records to "
          f"{ANNOTATIONS_PATH.relative_to(PROJECT_ROOT)}")

    print("Downloading images from images.cocodataset.org/val2017 ...")
    downloaded = 0
    for i, img in enumerate(selected, start=1):
        try:
            if download_image(img["file_name"]):
                downloaded += 1
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 1
        if i % 20 == 0:
            print(f"  {i}/{len(selected)}")
    print(f"Downloaded {downloaded} new images "
          f"({len(selected) - downloaded} already present).")

    write_manifest(selected)
    print(f"Manifest written: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print("Next: python tests/validate_benchmark_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
