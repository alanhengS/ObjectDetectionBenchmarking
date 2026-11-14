"""
Unit tests for Step 7 (src/validate_experiment.py, src/evaluate_accuracy.py)
and the two Step 7 fixes in compare_results / analyze_results.

Standard library only: no YOLO, PyTorch, OpenCV, or matplotlib needed, so
these run on any computer. All test data is synthetic and written to
temporary folders; nothing touches results/.

Run from the project root:
    python -m unittest tests.test_validation -v
"""

import csv
import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from src import analyze_results as ar
from src import compare_results
from src import evaluate_accuracy as ea
from src import validate_experiment as ve
from src.benchmark_config import CONFIG_PATH, load_all_configs, load_comparisons


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def temp_project(tmp):
    """Config copy whose outputs and model file live in `tmp`. Returns (config path, model hash)."""
    tmp = Path(tmp)
    model = tmp / "yolo11n.pt"
    model.write_bytes(b"fake weights for tests")
    data = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    data["common"].update(model=str(model), raw_results_dir=str(tmp / "raw"),
                          summary_dir=str(tmp / "summary"))
    data["analysis"].update(summaries_dir=str(tmp / "summaries"), graphs_dir=str(tmp / "graphs"))
    path = tmp / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, hashlib.sha256(model.read_bytes()).hexdigest()[:16]


def manifest_fingerprint(cfg):
    with open(cfg.manifest_path, newline="") as f:
        return ve.image_list_fingerprint(list(csv.DictReader(f)))


def trial_rows(cfg, times, session, model_sha, image_sha, trials=None, n=100, **override):
    rows = []
    for i, t in enumerate(times):
        row = {
            "run_id": f"{session}-{cfg.key}", "session_id": session,
            "timestamp": f"2026-11-12T12:00:0{ord(cfg.key) - 65}", "configuration": cfg.name,
            "config_id": cfg.key, "device": cfg.device,
            "processor": f"Apple M2 {cfg.device_label}", "resolution": cfg.resolution,
            "model": "yolo11n.pt", "model_sha256": model_sha,
            "dataset": cfg.dataset_id, "image_list_sha256": image_sha, "num_images": n,
            "cpu_threads": cfg.cpu_threads, "warmup_images": cfg.num_warmup,
            "measurement": "elapsed_workload_time",
            "trial": (trials or range(1, len(times) + 1))[i], "total_time_s": f"{t:.6f}",
            "avg_time_ms_per_image": f"{t / n * 1000:.4f}" if t > 0 else "0",
            "throughput_img_per_s": f"{n / t:.4f}" if t > 0 else "0",
        }
        row.update(override)
        rows.append(row)
    return rows


def write_run(raw_dir, rows):
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{rows[0]['configuration'].lower()}_{rows[0]['run_id']}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ve.REQUIRED_COLUMNS)
        w.writeheader()
        w.writerows(rows)


TIMES = {"A": [3.1, 2.9, 3.0, 3.2, 2.8], "B": [1.0] * 5,
         "C": [1.4, 1.5, 1.6, 1.5, 1.5], "D": [1.25] * 5}       # means 3.0, 1.0, 1.5, 1.25


def statuses(checks, section_start, status=None):
    return [(name, st, d) for sec, name, st, d in checks.items
            if sec.startswith(section_start) and (status is None or st == status)]


# ----------------------------------------------------------------------------
# Formulas
# ----------------------------------------------------------------------------
class TestIndependentFormulas(unittest.TestCase):
    def test_recalculate_known_values(self):
        r = ve.recalculate([2.0, 4.0, 6.0], 100)
        self.assertAlmostEqual(r["mean_time_s"], 4.0)            # (2 + 4 + 6) / 3
        self.assertAlmostEqual(r["ms_per_image"], 40.0)          # 4 s / 100 x 1000
        self.assertAlmostEqual(r["mean_throughput"], (50 + 25 + 100 / 6) / 3)
        self.assertEqual((r["min_time_s"], r["max_time_s"], r["range_s"]), (2.0, 6.0, 4.0))

    def test_speedup_direction(self):
        self.assertEqual(ve.speedup_direction(3.0 / 1.5), "comparison faster")
        self.assertEqual(ve.speedup_direction(1.0 / 1.25), "comparison SLOWER")
        self.assertEqual(ve.speedup_direction(1.001), "same")

    def test_letterbox_shapes(self):
        self.assertEqual(ve.letterbox_shape(640, 426, 640), (448, 640))   # 426 -> 448 (x32)
        self.assertEqual(ve.letterbox_shape(586, 640, 640), (640, 608))
        self.assertEqual(ve.letterbox_shape(640, 426, 320), (224, 320))   # 213 -> 224
        self.assertEqual(ve.letterbox_shape(500, 375, 640), (480, 640))   # upscaled to 640

    def test_png_info_reads_dpi(self):
        def chunk(kind, data):
            return (struct.pack(">I", len(data)) + kind + data +
                    struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2100, 1380, 8, 2, 0, 0, 0))
               + chunk(b"pHYs", struct.pack(">IIB", 11811, 11811, 1)) + chunk(b"IEND", b""))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.png"
            path.write_bytes(png)
            self.assertEqual(ve.png_info(path), (2100, 1380, 300))
            path.write_bytes(b"not a png")
            self.assertIsNone(ve.png_info(path))

    def test_fingerprint_matches_benchmark_module(self):
        try:
            from src import benchmark
        except ImportError:
            self.skipTest("benchmark.py needs OpenCV/PyTorch (run on the Mac)")
        rows = [{"file_name": "a.jpg", "sha256": "1"}, {"file_name": "b.jpg", "sha256": "2"}]
        self.assertEqual(ve.image_list_fingerprint(rows), benchmark.image_list_fingerprint(rows))


# ----------------------------------------------------------------------------
# Raw-data validation
# ----------------------------------------------------------------------------
class TestSessionValidation(unittest.TestCase):
    def run_validation(self, edit=None, session=None):
        """Write a valid session S1 (optionally edited), validate, return Checks."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path, model_sha = temp_project(tmp)
            configs = load_all_configs(cfg_path)
            image_sha = manifest_fingerprint(configs["A"])
            runs = {k: trial_rows(configs[k], TIMES[k], "S1", model_sha, image_sha)
                    for k in configs}
            if edit:
                edit(runs, configs, model_sha, image_sha)
            for rows in runs.values():
                write_run(Path(tmp) / "raw", rows)
            checks, sid = ve.validate(session, cfg_path)
        return checks, sid

    def test_valid_session_passes_raw_checks(self):
        checks, sid = self.run_validation()
        self.assertEqual(sid, "S1")
        for section in ("Configuration file", "Benchmark session", "Raw trials",
                        "Same workload", "Input resolution"):
            self.assertEqual(statuses(checks, section, ve.FAIL), [], section)
        speedups = {n: d for n, st, d in statuses(checks, "Recalculation")}
        self.assertIn("= 3.000 (comparison faster", speedups["Speedup: Resolution effect on CPU"])
        self.assertIn("= 0.800 (comparison SLOWER", speedups["Speedup: CPU vs. GPU at 320x320"])
        self.assertIn("same workload", speedups["Speedup: CPU vs. GPU at 320x320"])
        self.assertIn("different workload", speedups["Speedup: Resolution effect on GPU"])

    def failed(self, edit):
        checks, _ = self.run_validation(edit)
        return " | ".join(f"{n}: {d}" for _, n, st, d in
                          [(s, n, st, d) for s, n, st, d in checks.items if st == ve.FAIL])

    def test_missing_trial(self):
        def edit(runs, *_):
            runs["D"] = runs["D"][:4]
        self.assertIn("At least 5 trials", self.failed(edit))

    def test_duplicate_trial_id(self):
        def edit(runs, *_):
            runs["B"][4]["trial"] = 4
        self.assertIn("no duplicates or gaps", self.failed(edit))

    def test_gap_in_trial_ids(self):
        def edit(runs, configs, m, i):
            runs["B"] = trial_rows(configs["B"], TIMES["B"], "S1", m, i, trials=[1, 2, 3, 4, 6])
        self.assertIn("no duplicates or gaps", self.failed(edit))

    def test_invalid_time_stops_recalculation(self):
        def edit(runs, configs, m, i):
            runs["C"] = trial_rows(configs["C"], [1.5, 1.5, 0.0, 1.5, 1.5], "S1", m, i)
        checks, _ = self.run_validation(edit)
        fails = [n for _, n, st, _ in checks.items if st == ve.FAIL]
        self.assertIn("Every execution time is a positive number (seconds)", fails)
        self.assertIn("All four configurations usable", fails)
        self.assertEqual([n for s, n, st, d in checks.items if s == "Recalculation" and st == ve.INFO], [])

    def test_wrong_image_count(self):
        def edit(runs, configs, m, i):
            runs["A"] = trial_rows(configs["A"], TIMES["A"], "S1", m, i, n=99)
        self.assertIn("processed 100 images", self.failed(edit))

    def test_different_model_weights(self):
        def edit(runs, configs, m, i):
            runs["C"] = trial_rows(configs["C"], TIMES["C"], "S1", "differenthash00", i)
        self.assertIn("Same model weights", self.failed(edit))

    def test_wrong_timing_boundary(self):
        def edit(runs, configs, m, i):
            runs["D"] = trial_rows(configs["D"], TIMES["D"], "S1", m, i, measurement="inference_only")
        failed = self.failed(edit)
        self.assertIn("Timing boundary", failed)
        self.assertIn("Same timing boundary", failed)

    def test_gpu_run_recorded_on_cpu_rejected(self):
        def edit(runs, configs, m, i):
            runs["C"] = trial_rows(configs["C"], TIMES["C"], "S1", m, i, device="cpu")
        self.assertIn("Device recorded", self.failed(edit))

    def test_incomplete_newer_session_not_combined(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path, m = temp_project(tmp)
            configs = load_all_configs(cfg_path)
            i = manifest_fingerprint(configs["A"])
            raw = Path(tmp) / "raw"
            for k in configs:
                write_run(raw, trial_rows(configs[k], TIMES[k], "S1", m, i))
            for k in "AB":                     # newer session S2 with only A and B
                write_run(raw, trial_rows(configs[k], [9.9] * 5, "S2", m, i,
                                          timestamp="2026-12-01T00:00:00"))
            checks, sid = ve.validate(None, cfg_path)
            checks_s2, _ = ve.validate("S2", cfg_path)
        self.assertEqual(sid, "S1")
        info = {n: d for s, n, st, d in checks.items if st == ve.INFO}
        self.assertIn("S2", info["Other sessions in results/raw (not combined)"])
        self.assertIn("Exactly one run of GPU-640 in this session",
                      [n for s, n, st, d in checks_s2.items if st == ve.FAIL])

    def test_no_data_reports_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path, _ = temp_project(tmp)
            checks, sid = ve.validate(None, cfg_path)
        self.assertIsNone(sid)
        self.assertIn("python -m src.run_all_benchmarks", checks.items[-1][3])


class TestStep6OutputsCheck(unittest.TestCase):
    """The Step 6 summary, speedup CSV and report must equal the recalculation."""

    def make(self, tamper=None):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path, m = temp_project(tmp)
            configs = load_all_configs(cfg_path)
            i = manifest_fingerprint(configs["A"])
            runs = {k: trial_rows(configs[k], TIMES[k], "S1", m, i) for k in configs}
            for rows in runs.values():
                write_run(Path(tmp) / "raw", rows)
            # Step 6 outputs from the real analyze_results code (stdlib parts only)
            str_runs = {k: [{**r, "trial": str(r["trial"]), "num_images": str(r["num_images"])}
                            for r in rows] for k, rows in runs.items()}
            results = {k: ar.analyze_run(str_runs[k], configs[k]) for k in configs}
            speedups = ar.compute_speedups(results, configs, load_comparisons(cfg_path))
            out = Path(tmp) / "summaries"
            ar.write_summary_csv(results, out / "performance_summary_S1.csv")
            ar.write_speedup_csv(speedups, out / "speedup_S1.csv")
            (out / "analysis_report_S1.md").write_text(
                ar.build_report(results, speedups, "S1", 10), encoding="utf-8")
            if tamper:
                tamper(out)
            checks, _ = ve.validate("S1", cfg_path)
        return statuses(checks, "Step 6 summary")

    def test_matching_outputs_pass(self):
        rows = self.make()
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(st == ve.PASS for _, st, _ in rows), rows)

    def test_tampered_summary_value_detected(self):
        def tamper(out):
            p = out / "performance_summary_S1.csv"
            p.write_text(p.read_text().replace("3.000000", "2.500000", 1))
        rows = self.make(tamper)
        self.assertIn(("Summary values = independent recalculation", ve.FAIL),
                      [(n, st) for n, st, _ in rows])

    def test_swapped_speedup_detected(self):
        def tamper(out):
            p = out / "speedup_S1.csv"
            p.write_text(p.read_text().replace("2.0000", "0.5000", 1))   # CPU/GPU 640 inverted
        rows = self.make(tamper)
        self.assertIn(("Speedups and direction = independent recalculation", ve.FAIL),
                      [(n, st) for n, st, _ in rows])


# ----------------------------------------------------------------------------
# Accuracy-evaluation helpers
# ----------------------------------------------------------------------------
class TestCocoToYolo(unittest.TestCase):
    COCO = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100},
                   {"id": 2, "file_name": "b.jpg", "width": 50, "height": 50}],
        "categories": [{"id": 3, "name": "car"}, {"id": 1, "name": "person"}],
        "annotations": [
            {"image_id": 1, "category_id": 3, "bbox": [50, 25, 100, 50], "iscrowd": 0},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 10], "iscrowd": 1},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 10], "iscrowd": 0},
        ],
    }

    def test_conversion(self):
        labels, names, crowd = ea.coco_to_yolo(self.COCO)
        self.assertEqual(names, ["person", "car"])               # sorted by COCO id
        # centre (100, 50) of 200x100 -> 0.5 0.5; size 100x50 -> 0.5 0.5; car = class 1
        self.assertEqual(labels["a.jpg"], ["1 0.500000 0.500000 0.500000 0.500000"])
        self.assertEqual(labels["b.jpg"], [])                    # image without objects
        self.assertEqual(crowd, 1)

    def test_real_annotations_match_coco_80_classes(self):
        coco = json.loads(ea.ANNOTATIONS_PATH.read_text(encoding="utf-8"))
        labels, names, crowd = ea.coco_to_yolo(coco)
        self.assertEqual(len(names), 80)
        self.assertEqual((names[0], names[-1]), ("person", "toothbrush"))
        self.assertEqual(len(labels), 100)
        self.assertEqual(sum(len(v) for v in labels.values()) + crowd, len(coco["annotations"]))

    def test_label_files_and_yaml(self):
        labels, names, _ = ea.coco_to_yolo(self.COCO)
        with tempfile.TemporaryDirectory() as tmp:
            boxes = ea.write_labels(labels, ["a.jpg", "b.jpg"], Path(tmp) / "labels")
            self.assertEqual(boxes, 1)
            self.assertEqual((Path(tmp) / "labels" / "b.txt").read_text(), "")
            y = ea.write_dataset_yaml(Path(tmp) / "d.yaml", Path(tmp), names).read_text()
        self.assertIn('val: images', y)
        self.assertIn('  1: "car"', y)


# ----------------------------------------------------------------------------
# Step 7 fixes in existing modules
# ----------------------------------------------------------------------------
class TestStep7Fixes(unittest.TestCase):
    def test_compare_results_warns_on_mixed_sessions(self):
        base = {"num_images": 100, "dataset": "d", "model": "m", "model_sha256": "x",
                "image_list_sha256": "y", "warmup_images": "100"}
        same = {"A": {**base, "session_id": "S1"}, "B": {**base, "session_id": "S1"}}
        mixed = {"A": {**base, "session_id": "S1"}, "B": {**base, "session_id": "S2"}}
        self.assertEqual(compare_results.check_consistency(same), [])
        self.assertTrue(any("different benchmark sessions" in w
                            for w in compare_results.check_consistency(mixed)))

    def test_analyze_results_rejects_trial_gaps(self):
        cfg = load_all_configs()["A"]
        rows = trial_rows(cfg, [3.0] * 5, "S1", "m", "i", trials=[1, 2, 3, 4, 6])
        rows = [{k: str(v) for k, v in r.items()} for r in rows]
        self.assertTrue(any("not 1..5" in p for p in ar.validate_run(rows, cfg)))
        ok = [{k: str(v) for k, v in r.items()} for r in trial_rows(cfg, [3.0] * 5, "S1", "m", "i")]
        self.assertEqual(ar.validate_run(ok, cfg), [])


if __name__ == "__main__":
    unittest.main()
