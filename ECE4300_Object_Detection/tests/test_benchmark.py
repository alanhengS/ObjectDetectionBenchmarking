"""
Unit tests for src/benchmark.py (no real YOLO inference; runs in ~1 second).

Run from the project root:
    python -m unittest tests.test_benchmark -v
"""

import csv
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from src import benchmark


class FakeModel:
    """Stand-in for YOLO: records each predict() call and sleeps briefly."""

    def __init__(self, delay_s=0.001):
        self.delay_s = delay_s
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append((source, kwargs))
        time.sleep(self.delay_s)
        return []


def fake_images(n):
    """n small images; pixel value = index, so order can be checked."""
    return [(f"img_{i:03d}.jpg", np.full((4, 4, 3), i, dtype=np.uint8)) for i in range(n)]


class TestManifest(unittest.TestCase):
    def test_manifest_has_100_images_in_fixed_order(self):
        rows = benchmark.read_manifest()
        self.assertEqual(len(rows), 100)
        ids = [int(r["image_id"]) for r in rows]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), 100)

    def test_images_load_in_manifest_order(self):
        rows = benchmark.read_manifest()[:5]
        images = benchmark.load_images(rows)
        self.assertEqual([name for name, _ in images], [r["file_name"] for r in rows])

    def test_missing_image_raises(self):
        rows = [{"file_name": "nope.jpg", "relative_path": "data/benchmark/images/nope.jpg"}]
        with self.assertRaises(FileNotFoundError):
            benchmark.load_images(rows)


class TestConfiguration(unittest.TestCase):
    def test_settings(self):
        self.assertEqual(benchmark.DEVICE, "cpu")
        self.assertEqual(benchmark.IMAGE_SIZE, 640)
        self.assertEqual(benchmark.BATCH_SIZE, 1)
        self.assertEqual(benchmark.NUM_TRIALS, 5)
        self.assertEqual(benchmark.NUM_THREADS, 4)

    def test_predict_called_with_cpu_and_640(self):
        model = FakeModel(delay_s=0)
        benchmark.predict_one(model, np.zeros((4, 4, 3), dtype=np.uint8))
        kwargs = model.calls[0][1]
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["imgsz"], 640)
        self.assertEqual(kwargs["batch"], 1)
        self.assertEqual(kwargs["conf"], benchmark.CONF_THRESHOLD)


class TestFormulas(unittest.TestCase):
    def test_average_time(self):
        # 10 s for 100 images = 0.1 s = 100 ms per image
        self.assertAlmostEqual(benchmark.average_time_ms(10.0, 100), 100.0)
        self.assertAlmostEqual(benchmark.average_time_ms(2.5, 100), 25.0)

    def test_throughput(self):
        # 100 images in 10 s = 10 images/s
        self.assertAlmostEqual(benchmark.throughput(10.0, 100), 10.0)
        self.assertAlmostEqual(benchmark.throughput(4.0, 100), 25.0)

    def test_latency_and_throughput_are_reciprocal(self):
        ms = benchmark.average_time_ms(7.3, 100)
        fps = benchmark.throughput(7.3, 100)
        self.assertAlmostEqual(ms * fps, 1000.0)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.throughput(0.0, 100)
        with self.assertRaises(ValueError):
            benchmark.average_time_ms(-1.0, 100)


class TestTiming(unittest.TestCase):
    def test_trial_time_positive_and_order_fixed(self):
        model = FakeModel()
        images = fake_images(10)
        elapsed = benchmark.run_trial(model, images)
        self.assertGreater(elapsed, 0)
        seen = [int(src[0, 0, 0]) for src, _ in model.calls]
        self.assertEqual(seen, list(range(10)))

    def test_model_loading_excluded_from_timing(self):
        # Simulate a slow model load (0.5 s) before the trial.
        # The trial only receives the already-loaded model, so its time
        # must reflect just the 5 x 1 ms predict calls, not the load.
        time.sleep(0.5)
        model = FakeModel(delay_s=0.001)
        elapsed = benchmark.run_trial(model, fake_images(5))
        self.assertLess(elapsed, 0.25)

    def test_five_trial_records(self):
        model = FakeModel()
        images = fake_images(4)
        records = benchmark.run_benchmark(model, images, num_trials=5)
        self.assertEqual(len(records), 5)
        self.assertEqual([r["trial"] for r in records], [1, 2, 3, 4, 5])
        self.assertEqual(len(model.calls), 5 * 4)  # every image, every trial
        for r in records:
            self.assertGreater(r["total_time_s"], 0)


class TestCsvOutput(unittest.TestCase):
    def make_records(self):
        return [{"trial": t, "total_time_s": 10.0, "avg_time_ms_per_image": 100.0,
                 "throughput_img_per_s": 10.0} for t in range(1, 6)]

    def test_csv_written_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            p1 = benchmark.save_results(self.make_records(), "run-A", "2026-01-01T00:00:00",
                                        "Apple M2", 100, out_dir=out)
            p2 = benchmark.save_results(self.make_records(), "run-B", "2026-01-01T00:00:01",
                                        "Apple M2", 100, out_dir=out)
            self.assertNotEqual(p1, p2)
            self.assertEqual(len(list(out.glob("*.csv"))), 2)

            with open(p1, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)
            self.assertEqual(list(rows[0].keys()), benchmark.CSV_COLUMNS)
            self.assertEqual(rows[0]["configuration"], "CPU-640")
            self.assertEqual(rows[0]["resolution"], "640x640")
            self.assertEqual(rows[0]["num_images"], "100")

            # Writing the same run ID again must fail rather than overwrite
            with self.assertRaises(FileExistsError):
                benchmark.save_results(self.make_records(), "run-A", "x", "Apple M2", 100,
                                       out_dir=out)

    def test_run_ids_unique(self):
        from datetime import datetime
        now = datetime.now()
        self.assertNotEqual(benchmark.new_run_id(now), benchmark.new_run_id(now))


if __name__ == "__main__":
    unittest.main()
