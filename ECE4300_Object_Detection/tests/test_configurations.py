"""
Unit tests for Step 5: configurations A-D, GPU synchronization, device
checks, CSV output, and speedup calculations. No real YOLO inference is
run, and no GPU is needed.

Run from the project root:
    python -m unittest tests.test_configurations -v
"""

import csv
import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from src import benchmark, compare_results
from src.benchmark_config import (
    CONFIG_PATH, ConfigError, find_config, load_all_configs, load_comparisons, load_config,
)


class FakeModel:
    """Stand-in for YOLO: records each predict() call."""

    def __init__(self, delay_s=0.0, log=None):
        self.delay_s = delay_s
        self.calls = []
        self.log = log

    def predict(self, source, **kwargs):
        self.calls.append((source, kwargs))
        if self.log is not None:
            self.log.append("predict")
        time.sleep(self.delay_s)
        return []


def fake_images(n):
    return [(f"img_{i:03d}.jpg", np.full((4, 4, 3), i, dtype=np.uint8)) for i in range(n)]


def temp_config(tmp, **overrides):
    """Copy of the real config file with results redirected into `tmp`."""
    data = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    data["common"]["raw_results_dir"] = str(Path(tmp) / "raw")
    data["common"]["summary_dir"] = str(Path(tmp) / "summary")
    data["common"].update(overrides)
    path = Path(tmp) / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestConfigFile(unittest.TestCase):
    def setUp(self):
        self.configs = load_all_configs()

    def test_four_configurations(self):
        got = {k: (c.name, c.device, c.image_size) for k, c in self.configs.items()}
        self.assertEqual(got, {
            "A": ("CPU-640", "cpu", 640),
            "B": ("CPU-320", "cpu", 320),
            "C": ("GPU-640", "mps", 640),
            "D": ("GPU-320", "mps", 320),
        })

    def test_baseline_is_cpu_640(self):
        base = load_config()
        self.assertEqual((base.key, base.name), ("A", "CPU-640"))
        self.assertTrue(base.is_baseline)
        self.assertEqual(sum(c.is_baseline for c in self.configs.values()), 1)

    def test_only_device_and_resolution_differ(self):
        a = self.configs["A"]
        for c in self.configs.values():
            same = replace(c, key=a.key, name=a.name, device=a.device,
                           image_size=a.image_size, is_baseline=a.is_baseline)
            self.assertEqual(same, a, f"{c.name} differs in a shared setting")

    def test_shared_settings(self):
        for c in self.configs.values():
            self.assertEqual(c.batch_size, 1)
            self.assertEqual(c.num_trials, 5)
            self.assertEqual(c.expected_images, 100)
            self.assertEqual(c.model_path.name, "yolo11n.pt")

    def test_lookup_by_key_or_name(self):
        self.assertEqual(find_config("b", self.configs).name, "CPU-320")
        self.assertEqual(find_config("gpu-640", self.configs).key, "C")
        with self.assertRaises(ConfigError):
            find_config("TPU-640", self.configs)

    def test_invalid_device_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = temp_config(tmp)
            data = json.loads(path.read_text())
            data["configurations"]["C"]["device"] = "cuda"
            path.write_text(json.dumps(data))
            with self.assertRaises(ConfigError):
                load_all_configs(path)

    def test_comparisons_match_assignment(self):
        pairs = [(c["baseline"], c["comparison"], c["kind"]) for c in load_comparisons()]
        self.assertEqual(pairs, [("A", "B", "resolution"), ("C", "D", "resolution"),
                                 ("A", "C", "device"), ("B", "D", "device")])

    def test_step4_constants_still_baseline(self):
        self.assertEqual(benchmark.DEVICE, "cpu")
        self.assertEqual(benchmark.IMAGE_SIZE, 640)
        self.assertEqual(benchmark.CONFIG_NAME, "CPU-640")


class TestPredictSettings(unittest.TestCase):
    def test_each_config_passes_its_device_and_size(self):
        for cfg in load_all_configs().values():
            model = FakeModel()
            benchmark.predict_one(model, np.zeros((4, 4, 3), dtype=np.uint8), cfg)
            kwargs = model.calls[0][1]
            self.assertEqual(kwargs["device"], cfg.device)
            self.assertEqual(kwargs["imgsz"], cfg.image_size)
            self.assertEqual(kwargs["batch"], 1)
            self.assertEqual(kwargs["conf"], 0.25)
            self.assertEqual(kwargs["iou"], 0.7)


class TestGpuSynchronization(unittest.TestCase):
    def test_synchronize_calls_mps_only_for_gpu(self):
        with mock.patch("torch.mps.synchronize") as sync:
            benchmark.synchronize("cpu")
            sync.assert_not_called()
            benchmark.synchronize("mps")
            sync.assert_called_once()

    def test_gpu_trial_syncs_after_every_image(self):
        log = []
        model = FakeModel(log=log)
        cfg = load_config("C")
        with mock.patch("torch.mps.synchronize", side_effect=lambda: log.append("sync")):
            benchmark.run_trial(model, fake_images(4), cfg)
        self.assertEqual(log, ["predict", "sync"] * 4)

    def test_timer_includes_pending_gpu_work(self):
        # If the GPU is still busy (simulated by a 50 ms synchronize),
        # the measured time must include that wait.
        cfg = load_config("D")
        with mock.patch("torch.mps.synchronize", side_effect=lambda: time.sleep(0.05)):
            elapsed = benchmark.run_trial(FakeModel(), fake_images(3), cfg)
        self.assertGreaterEqual(elapsed, 0.15)

    def test_cpu_trial_never_calls_mps(self):
        with mock.patch("torch.mps.synchronize") as sync:
            benchmark.run_trial(FakeModel(), fake_images(3), load_config("B"))
        sync.assert_not_called()


class TestDeviceChecks(unittest.TestCase):
    def test_cpu_always_allowed(self):
        benchmark.check_device_available("cpu")

    def test_mps_unavailable_is_an_error_not_a_fallback(self):
        with mock.patch("torch.backends.mps.is_built", return_value=True), \
             mock.patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaises(benchmark.DeviceError) as ctx:
                benchmark.check_device_available("mps")
        self.assertIn("not fall back", str(ctx.exception))

    def test_mps_not_built_is_an_error(self):
        with mock.patch("torch.backends.mps.is_built", return_value=False):
            with self.assertRaises(benchmark.DeviceError):
                benchmark.check_device_available("mps")

    def test_cpu_fallback_env_var_rejected(self):
        with mock.patch("torch.backends.mps.is_built", return_value=True), \
             mock.patch("torch.backends.mps.is_available", return_value=True), \
             mock.patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaises(benchmark.DeviceError):
                benchmark.check_device_available("mps")

    @staticmethod
    def fake_model_and_result(weights_device, output_device):
        net = torch.nn.Linear(1, 1).to(weights_device)
        model = SimpleNamespace(predictor=SimpleNamespace(device=torch.device(weights_device),
                                                          model=net))
        result = SimpleNamespace(boxes=SimpleNamespace(data=torch.zeros(1, device=output_device)))
        return model, result

    def test_weights_found_inside_ultralytics_backend(self):
        # Ultralytics 8.4: AutoBackend has no parameters of its own; the
        # network is at AutoBackend.backend.model.
        class Wrapper(torch.nn.Module):
            pass
        wrapper = Wrapper()
        wrapper.__dict__["backend"] = SimpleNamespace(model=torch.nn.Linear(1, 1))
        self.assertEqual(benchmark.weights_device(wrapper), "cpu")
        with self.assertRaises(benchmark.DeviceError):
            benchmark.weights_device(Wrapper())   # no weights anywhere

    def test_verify_model_device_accepts_match(self):
        model, result = self.fake_model_and_result("cpu", "cpu")
        found = benchmark.verify_model_device(model, result, "cpu")
        self.assertEqual(set(found.values()), {"cpu"})

    def test_verify_model_device_rejects_silent_cpu_fallback(self):
        # Asked for MPS, but the model ended up on the CPU.
        model, result = self.fake_model_and_result("cpu", "cpu")
        with self.assertRaises(benchmark.DeviceError):
            benchmark.verify_model_device(model, result, "mps")


class TestCsvOutput(unittest.TestCase):
    records = [{"trial": t, "total_time_s": 2.0, "avg_time_ms_per_image": 20.0,
                "throughput_img_per_s": 50.0} for t in range(1, 6)]

    def test_gpu_csv_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config("D")
            path = benchmark.save_results(self.records, "run-X", "2026-01-01T00:00:00",
                                          "Apple M2", 100, out_dir=Path(tmp), cfg=cfg,
                                          session_id="S1", cpu_threads=4)
            self.assertTrue(path.name.startswith("gpu-320_"))
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)
        r = rows[0]
        self.assertEqual((r["configuration"], r["config_id"], r["device"], r["resolution"]),
                         ("GPU-320", "D", "mps", "320x320"))
        self.assertEqual(r["processor"], "Apple M2 GPU (MPS)")
        self.assertEqual(r["session_id"], "S1")
        self.assertEqual(r["measurement"], "elapsed_workload_time")
        self.assertEqual(r["num_images"], "100")

    def test_failure_recorded_and_appended(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config("C")
            benchmark.record_failure(cfg, "S1", "MPS not available", out_dir=Path(tmp))
            path = benchmark.record_failure(cfg, "S2", "second error", out_dir=Path(tmp))
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual([r["error"] for r in rows], ["MPS not available", "second error"])
        self.assertEqual(rows[0]["configuration"], "GPU-640")

    def test_image_fingerprint_depends_on_order(self):
        rows = [{"file_name": "a.jpg", "sha256": "1"}, {"file_name": "b.jpg", "sha256": "2"}]
        self.assertNotEqual(benchmark.image_list_fingerprint(rows),
                            benchmark.image_list_fingerprint(rows[::-1]))


class TestSpeedup(unittest.TestCase):
    def test_formula(self):
        # Course formula: baseline time / comparison time
        self.assertAlmostEqual(compare_results.speedup(3.0, 1.5), 2.0)
        self.assertAlmostEqual(compare_results.speedup(2.0, 4.0), 0.5)
        with self.assertRaises(ValueError):
            compare_results.speedup(0.0, 1.0)

    def test_slower_comparison_explained(self):
        text = compare_results.interpret(0.8, "CPU-640", "GPU-640", "device")
        self.assertIn("SLOWER", text)
        self.assertIn("1.25x as long", text)
        self.assertIn("same-workload", text)

    def test_resolution_is_not_called_a_hardware_speedup(self):
        text = compare_results.interpret(2.5, "CPU-640", "CPU-320", "resolution")
        self.assertIn("2.50x faster", text)
        self.assertIn("NOT a same-workload", text)


class TestCompareEndToEnd(unittest.TestCase):
    """Write fake raw CSVs for A-D and check the summary and speedups."""

    TIMES = {"A": 3.0, "B": 1.0, "C": 1.5, "D": 1.2}   # seconds per trial

    def write_session(self, configs, session, keys):
        for key in keys:
            cfg = configs[key]
            t = self.TIMES[key]
            recs = [{"trial": i, "total_time_s": t,
                     "avg_time_ms_per_image": benchmark.average_time_ms(t, 100),
                     "throughput_img_per_s": benchmark.throughput(t, 100)} for i in range(1, 6)]
            benchmark.save_results(recs, f"{session}-{key}", f"2026-01-01T00:00:0{ord(key) - 65}",
                                   "Apple M2", 100, cfg=cfg, session_id=session,
                                   model_sha256="abc", image_list_sha256="def", cpu_threads=4)

    def test_summary_and_speedups(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = temp_config(tmp)
            configs = load_all_configs(cfg_path)
            self.write_session(configs, "S1", "ABCD")
            with mock.patch("builtins.print"):
                code = compare_results.main(["--session", "S1", "--config-file", str(cfg_path)])
            self.assertEqual(code, 0)
            summary_dir = Path(tmp) / "summary"
            speed_file = next(summary_dir.glob("speedup_*.csv"))
            with open(speed_file, newline="") as f:
                speedups = {r["comparison"]: float(r["speedup"]) for r in csv.DictReader(f)}
            with open(next(summary_dir.glob("summary_*.csv")), newline="") as f:
                summary = list(csv.DictReader(f))
        self.assertAlmostEqual(speedups["Resolution effect on CPU"], 3.0 / 1.0, places=3)
        self.assertAlmostEqual(speedups["Resolution effect on GPU"], 1.5 / 1.2, places=3)
        self.assertAlmostEqual(speedups["CPU vs. GPU at 640x640"], 3.0 / 1.5, places=3)
        self.assertAlmostEqual(speedups["CPU vs. GPU at 320x320"], 1.0 / 1.2, places=3)  # GPU slower
        self.assertEqual([r["configuration"] for r in summary],
                         ["CPU-640", "CPU-320", "GPU-640", "GPU-320"])
        self.assertTrue(all(r["status"] == "OK" for r in summary))

    def test_failed_configuration_reported_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = temp_config(tmp)
            configs = load_all_configs(cfg_path)
            self.write_session(configs, "S2", "AB")
            for key in "CD":
                benchmark.record_failure(configs[key], "S2", "MPS not available")
            with mock.patch("builtins.print"):
                code = compare_results.main(["--session", "S2", "--config-file", str(cfg_path)])
            self.assertEqual(code, 1)
            with open(next((Path(tmp) / "summary").glob("summary_*.csv")), newline="") as f:
                summary = {r["config_id"]: r for r in csv.DictReader(f)}
            with open(next((Path(tmp) / "summary").glob("speedup_*.csv")), newline="") as f:
                speedups = list(csv.DictReader(f))
        self.assertEqual(summary["C"]["status"], "FAILED")
        self.assertEqual(summary["C"]["mean_total_time_s"], "")
        self.assertIn("MPS not available", summary["D"]["error"])
        calculated = [r for r in speedups if r["speedup"]]
        self.assertEqual([r["comparison"] for r in calculated], ["Resolution effect on CPU"])

    def test_step4_baseline_csv_still_readable(self):
        old_columns = ["run_id", "timestamp", "configuration", "processor", "resolution",
                       "model", "dataset", "num_images", "cpu_threads", "trial",
                       "total_time_s", "avg_time_ms_per_image", "throughput_img_per_s"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = temp_config(tmp)
            configs = load_all_configs(cfg_path)
            raw = Path(tmp) / "raw"
            raw.mkdir()
            with open(raw / "cpu-640_old.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=old_columns)
                w.writeheader()
                for t in range(1, 6):
                    w.writerow({"run_id": "old", "timestamp": "2026-11-04T13:16:30",
                                "configuration": "CPU-640", "processor": "Apple M2 CPU",
                                "resolution": "640x640", "model": "yolo11n.pt",
                                "dataset": "COCO2017-val-first100-by-id", "num_images": 100,
                                "cpu_threads": 4, "trial": t, "total_time_s": 3.2,
                                "avg_time_ms_per_image": 32.0, "throughput_img_per_s": 31.25})
            rows = compare_results.read_raw_rows(raw, configs)
            chosen = compare_results.select_runs(compare_results.group_runs(rows), configs)
        self.assertEqual(list(chosen), ["A"])
        self.assertEqual(chosen["A"][0]["device"], "cpu")
        self.assertEqual(len(chosen["A"]), 5)


if __name__ == "__main__":
    unittest.main()
