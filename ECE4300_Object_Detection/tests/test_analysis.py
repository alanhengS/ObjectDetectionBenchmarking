"""
Unit tests for Step 6 (src/analyze_results.py), using known mathematical
examples. Test data is written only to temporary folders; nothing here
touches results/ or produces the real project graphs.

Run from the project root:
    python -m unittest tests.test_analysis -v
"""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import analyze_results as ar
from src import benchmark
from src.benchmark_config import CONFIG_PATH, load_all_configs, load_config


def temp_config(tmp):
    """Copy of the real config with every output folder inside `tmp`."""
    data = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    data["common"]["raw_results_dir"] = str(Path(tmp) / "raw")
    data["common"]["summary_dir"] = str(Path(tmp) / "summary")
    data["analysis"]["summaries_dir"] = str(Path(tmp) / "summaries")
    data["analysis"]["graphs_dir"] = str(Path(tmp) / "graphs")
    data["analysis"]["graph_dpi"] = 60          # small, fast test images
    path = Path(tmp) / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def make_trials(times, cfg, num_images=100, session="S1", run_id="R1", **overrides):
    """Trial rows shaped like the Step 5 raw CSV."""
    rows = []
    for i, t in enumerate(times, start=1):
        row = {
            "run_id": run_id, "session_id": session, "timestamp": "2026-11-12T10:00:00",
            "configuration": cfg.name, "config_id": cfg.key, "device": cfg.device,
            "processor": f"Apple M2 {cfg.device_label}", "resolution": cfg.resolution,
            "model": "yolo11n.pt", "model_sha256": "m1", "dataset": "COCO",
            "image_list_sha256": "i1", "num_images": str(num_images), "cpu_threads": "4",
            "warmup_images": "100", "measurement": "elapsed_workload_time", "trial": str(i),
            "total_time_s": str(t),
            "avg_time_ms_per_image": f"{t / num_images * 1000:.4f}" if t > 0 else "0",
            "throughput_img_per_s": f"{num_images / t:.4f}" if t > 0 else "0",
        }
        row.update(overrides)
        rows.append(row)
    return rows


class TestFormulas(unittest.TestCase):
    def test_mean_execution_time(self):
        # (2 + 4 + 6) / 3 = 4
        self.assertAlmostEqual(ar.mean_execution_time([2.0, 4.0, 6.0]), 4.0)
        self.assertAlmostEqual(ar.mean_execution_time([3.1, 2.9, 3.0, 3.2, 2.8]), 3.0)

    def test_avg_processing_time_in_ms(self):
        # 4 s / 100 images = 0.04 s = 40 ms per image
        self.assertAlmostEqual(ar.avg_processing_time_ms(4.0, 100), 40.0)
        self.assertAlmostEqual(ar.avg_processing_time_ms(3.186, 100), 31.86)

    def test_trial_throughput(self):
        # 100 images in 2 s = 50 images/s
        self.assertAlmostEqual(ar.trial_throughput(2.0, 100), 50.0)

    def test_mean_throughput_averages_trial_throughputs(self):
        # Trials of 2 s and 4 s: 50 and 25 images/s -> mean 37.5.
        # (Not 100 / mean time = 100 / 3 = 33.3; the task asks for the mean of trials.)
        self.assertAlmostEqual(ar.mean_throughput([2.0, 4.0], 100), 37.5)

    def test_range(self):
        lo, hi, rng = ar.time_range([3.1, 2.9, 3.5, 3.0])
        self.assertEqual((lo, hi), (2.9, 3.5))
        self.assertAlmostEqual(rng, 0.6)

    def test_speedup_numerator_is_reference(self):
        # Reference 3 s, comparison 1.5 s -> comparison is 2x faster
        self.assertAlmostEqual(ar.speedup(3.0, 1.5), 2.0)
        # Reference 1 s, comparison 1.25 s -> 0.8 (comparison slower)
        self.assertAlmostEqual(ar.speedup(1.0, 1.25), 0.8)

    def test_speedup_direction_wording(self):
        text, faster = ar.describe_speedup(2.0, "CPU-640", "GPU-640")
        self.assertIn("GPU-640 was faster", text)
        self.assertEqual(faster, "GPU-640")
        text, faster = ar.describe_speedup(0.8, "CPU-320", "GPU-320")
        self.assertIn("SLOWER", text)
        self.assertIn("1.25x as long", text)
        self.assertEqual(faster, "CPU-320")
        _, faster = ar.describe_speedup(1.001, "A", "B")
        self.assertEqual(faster, "neither")

    def test_invalid_values_rejected(self):
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                ar.mean_execution_time([1.0, bad])
            with self.assertRaises(ValueError):
                ar.speedup(1.0, bad)
        with self.assertRaises(ValueError):
            ar.mean_execution_time([])


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config("C")

    def test_valid_run(self):
        self.assertEqual(ar.validate_run(make_trials([1.0] * 5, self.cfg), self.cfg), [])

    def test_too_few_trials(self):
        problems = ar.validate_run(make_trials([1.0] * 4, self.cfg), self.cfg)
        self.assertTrue(any("only 4 trials" in p for p in problems))

    def test_non_positive_time_rejected_not_skipped(self):
        problems = ar.validate_run(make_trials([1.0, 1.0, 0.0, 1.0, 1.0], self.cfg), self.cfg)
        self.assertTrue(any("not positive" in p for p in problems))

    def test_wrong_image_count(self):
        problems = ar.validate_run(make_trials([1.0] * 5, self.cfg, num_images=99), self.cfg)
        self.assertTrue(any("99 images" in p for p in problems))

    def test_different_timing_boundary_rejected(self):
        trials = make_trials([1.0] * 5, self.cfg, measurement="inference_only")
        self.assertTrue(ar.validate_run(trials, self.cfg))

    def test_stored_value_mismatch_detected(self):
        trials = make_trials([1.0] * 5, self.cfg)
        trials[2]["avg_time_ms_per_image"] = "12.0"     # should be 10.0
        self.assertTrue(any("does not match" in p for p in ar.validate_run(trials, self.cfg)))

    def test_wrong_device_rejected(self):
        trials = make_trials([1.0] * 5, self.cfg, device="cpu")
        self.assertTrue(any("device" in p for p in ar.validate_run(trials, self.cfg)))

    def test_incompatible_runs_detected(self):
        a, c = load_config("A"), load_config("C")
        same = {"A": make_trials([1.0] * 5, a), "C": make_trials([1.0] * 5, c)}
        self.assertEqual(ar.check_compatible(same), [])
        diff_model = {"A": make_trials([1.0] * 5, a),
                      "C": make_trials([1.0] * 5, c, model_sha256="other")}
        self.assertTrue(any("model weights" in p for p in ar.check_compatible(diff_model)))
        diff_session = {"A": make_trials([1.0] * 5, a),
                        "C": make_trials([1.0] * 5, c, session="S2")}
        self.assertTrue(any("session" in p for p in ar.check_compatible(diff_session)))
        old_run = {"A": make_trials([1.0] * 5, a, model_sha256=""),
                   "C": make_trials([1.0] * 5, c)}
        self.assertTrue(any("not recorded" in p for p in ar.check_compatible(old_run)))

    def test_only_complete_sessions_selected(self):
        configs = load_all_configs()
        runs = {}
        for key in "ABCD":
            runs[f"old-{key}"] = make_trials([1.0] * 5, configs[key], session="OLD",
                                             run_id=f"old-{key}")
        for key in "AB":    # newer session, but incomplete
            t = make_trials([1.0] * 5, configs[key], session="NEW", run_id=f"new-{key}")
            for r in t:
                r["timestamp"] = "2026-12-01T00:00:00"
            runs[f"new-{key}"] = t
        sessions = ar.find_sessions(runs, configs)
        sid, chosen = ar.select_session(sessions, configs)
        self.assertEqual(sid, "OLD")
        self.assertEqual(set(chosen), set("ABCD"))
        self.assertEqual(ar.select_session(sessions, configs, "MISSING"), (None, None))


class TestAnalysisEndToEnd(unittest.TestCase):
    """Known trial times -> known summary, speedups, report, and graph files."""

    # Mean times: A 3.0, B 1.0, C 1.5, D 1.25 seconds
    TIMES = {
        "A": [3.1, 2.9, 3.0, 3.2, 2.8],
        "B": [1.0, 1.0, 1.0, 1.0, 1.0],
        "C": [1.5, 1.5, 1.5, 1.5, 1.5],
        "D": [1.25, 1.25, 1.25, 1.25, 1.25],
    }

    def write_raw(self, cfg_path, session="S1", keys="ABCD", times=None):
        configs = load_all_configs(cfg_path)
        for key in keys:
            cfg = configs[key]
            t_list = (times or self.TIMES)[key]
            recs = [{"trial": i, "total_time_s": t,
                     "avg_time_ms_per_image": t / 100 * 1000,
                     "throughput_img_per_s": 100 / t} for i, t in enumerate(t_list, 1)]
            benchmark.save_results(recs, f"{session}-{key}", "2026-11-12T10:00:00", "Apple M2",
                                   100, cfg=cfg, session_id=session, model_sha256="m1",
                                   image_list_sha256="i1", cpu_threads=4)

    def run_main(self, cfg_path, *extra):
        with mock.patch("builtins.print"):
            return ar.main(["--config-file", str(cfg_path), *extra])

    def test_full_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = temp_config(tmp)
            self.write_raw(cfg_path)
            self.assertEqual(self.run_main(cfg_path), 0)

            summaries = Path(tmp) / "summaries"
            with open(summaries / "performance_summary_S1.csv", newline="") as f:
                summary = {r["config_id"]: r for r in csv.DictReader(f)}
            with open(summaries / "speedup_S1.csv", newline="") as f:
                speedups = {r["comparison"]: r for r in csv.DictReader(f)}
            report = (summaries / "analysis_report_S1.md").read_text(encoding="utf-8")
            graphs = sorted(p.name for p in (Path(tmp) / "graphs").glob("*.png"))
            sizes = [p.stat().st_size for p in (Path(tmp) / "graphs").glob("*.png")]

        a = summary["A"]
        self.assertEqual(list(a), ar.SUMMARY_COLUMNS)
        self.assertAlmostEqual(float(a["mean_execution_time_s"]), 3.0)
        self.assertAlmostEqual(float(a["avg_processing_time_ms_per_image"]), 30.0)
        self.assertAlmostEqual(float(a["min_execution_time_s"]), 2.8)
        self.assertAlmostEqual(float(a["max_execution_time_s"]), 3.2)
        self.assertAlmostEqual(float(a["execution_time_range_s"]), 0.4)
        expected_tp = sum(100 / t for t in self.TIMES["A"]) / 5
        self.assertAlmostEqual(float(a["mean_throughput_img_per_s"]), expected_tp, places=3)
        self.assertEqual((a["trials"], a["images_per_trial"]), ("5", "100"))
        self.assertEqual(summary["D"]["processor"], "Apple M2 GPU (MPS)")

        self.assertAlmostEqual(float(speedups["Resolution effect on CPU"]["speedup"]), 3.0)
        self.assertAlmostEqual(float(speedups["Resolution effect on GPU"]["speedup"]), 1.2)
        self.assertAlmostEqual(float(speedups["CPU vs. GPU at 640x640"]["speedup"]), 2.0)
        s320 = speedups["CPU vs. GPU at 320x320"]
        self.assertAlmostEqual(float(s320["speedup"]), 0.8)        # 1.0 / 1.25: GPU slower
        self.assertEqual(s320["faster_configuration"], "CPU-320")
        self.assertIn("SLOWER", s320["interpretation"])

        self.assertIn("**Shortest mean execution time:** CPU 320×320, 1.000 s", report)
        self.assertIn("the GPU was slower than the CPU", report)
        self.assertIn("hypotheses — not measured", report)

        self.assertEqual(graphs, ["cpu_gpu_speedup_S1.png", "execution_time_S1.png",
                                  "throughput_S1.png"])
        self.assertTrue(all(s > 1000 for s in sizes))

    def test_missing_data_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = temp_config(tmp)
            self.write_raw(cfg_path, keys="AB")             # GPU runs missing
            self.assertEqual(self.run_main(cfg_path), 1)
            self.assertFalse((Path(tmp) / "graphs").exists())
            self.assertFalse((Path(tmp) / "summaries").exists())

    def test_invalid_trial_blocks_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = temp_config(tmp)
            bad = dict(self.TIMES, C=[1.5, 1.5, 1.5, 1.5])  # only 4 trials
            self.write_raw(cfg_path, times=bad)
            self.assertEqual(self.run_main(cfg_path, "--session", "S1"), 1)
            self.assertFalse((Path(tmp) / "graphs").exists())

    def test_raw_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = temp_config(tmp)
            self.write_raw(cfg_path)
            raw = Path(tmp) / "raw"
            before = {p.name: p.read_bytes() for p in raw.glob("*.csv")}
            self.run_main(cfg_path)
            after = {p.name: p.read_bytes() for p in raw.glob("*.csv")}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
