"""
Load benchmark configurations A-D from config/benchmark_config.json.

All benchmark settings live in that one JSON file; Python code reads them
from here instead of hardcoding values in several places.
"""

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "benchmark_config.json"

VALID_DEVICES = ("cpu", "mps")
VALID_IMAGE_SIZES = (320, 640)


@dataclass(frozen=True)
class BenchmarkConfig:
    key: str               # "A".."D"
    name: str              # e.g. "CPU-640"
    device: str            # "cpu" or "mps"
    image_size: int        # 640 or 320 (square model input)
    batch_size: int
    conf_threshold: float
    iou_threshold: float
    num_trials: int
    num_warmup: int
    cpu_threads: int
    expected_images: int
    dataset_id: str
    model_path: Path
    manifest_path: Path
    raw_results_dir: Path
    summary_dir: Path
    is_baseline: bool

    @property
    def device_label(self):
        """'CPU' or 'GPU (MPS)', for reports."""
        return "GPU (MPS)" if self.device == "mps" else "CPU"

    @property
    def resolution(self):
        return f"{self.image_size}x{self.image_size}"


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_config_file(path=CONFIG_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Configuration file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Configuration file {path} is not valid JSON: {e}") from e


def _build(key, entry, common, baseline_key, root):
    merged = {**common, **entry}
    try:
        cfg = BenchmarkConfig(
            key=key,
            name=merged["name"],
            device=merged["device"],
            image_size=int(merged["image_size"]),
            batch_size=int(merged["batch_size"]),
            conf_threshold=float(merged["conf_threshold"]),
            iou_threshold=float(merged["iou_threshold"]),
            num_trials=int(merged["num_trials"]),
            num_warmup=int(merged["num_warmup"]),
            cpu_threads=int(merged["cpu_threads"]),
            expected_images=int(merged["expected_images"]),
            dataset_id=merged["dataset_id"],
            model_path=_resolve(root, merged["model"]),
            manifest_path=_resolve(root, merged["manifest"]),
            raw_results_dir=_resolve(root, merged["raw_results_dir"]),
            summary_dir=_resolve(root, merged["summary_dir"]),
            is_baseline=(key == baseline_key),
        )
    except KeyError as e:
        raise ConfigError(f"Configuration {key} is missing setting {e}") from e

    if cfg.device not in VALID_DEVICES:
        raise ConfigError(f"Configuration {key}: device must be one of {VALID_DEVICES}, "
                          f"got {cfg.device!r}")
    if cfg.image_size not in VALID_IMAGE_SIZES:
        raise ConfigError(f"Configuration {key}: image_size must be one of "
                          f"{VALID_IMAGE_SIZES}, got {cfg.image_size}")
    if cfg.batch_size != 1:
        raise ConfigError(f"Configuration {key}: batch_size must be 1 (sequential benchmark)")
    if cfg.num_trials < 1 or cfg.num_warmup < 0 or cfg.cpu_threads < 1:
        raise ConfigError(f"Configuration {key}: num_trials and cpu_threads must be >= 1 "
                          "and num_warmup >= 0")
    return cfg


def load_all_configs(path=CONFIG_PATH, root=PROJECT_ROOT):
    """Return {key: BenchmarkConfig} for every configuration, in file order."""
    data = read_config_file(path)
    common = data.get("common", {})
    baseline = data.get("baseline", "A")
    configs = {key: _build(key, entry, common, baseline, root)
               for key, entry in data.get("configurations", {}).items()}
    if baseline not in configs:
        raise ConfigError(f"Baseline configuration {baseline!r} is not defined")
    names = [c.name for c in configs.values()]
    if len(set(names)) != len(names):
        raise ConfigError("Configuration names must be unique")
    return configs


def load_comparisons(path=CONFIG_PATH):
    """Return the list of speedup comparisons defined in the config file."""
    return read_config_file(path).get("comparisons", [])


def find_config(selector, configs):
    """Look up a configuration by key ('B') or name ('CPU-320'), case-insensitive."""
    wanted = str(selector).strip().upper()
    for cfg in configs.values():
        if wanted in (cfg.key.upper(), cfg.name.upper()):
            return cfg
    choices = ", ".join(f"{c.key} ({c.name})" for c in configs.values())
    raise ConfigError(f"Unknown configuration {selector!r}. Choose one of: {choices}")


def load_config(selector=None, path=CONFIG_PATH, root=PROJECT_ROOT):
    """Return one configuration; with no selector, the baseline (A, CPU-640)."""
    configs = load_all_configs(path, root)
    if selector is None:
        return next(c for c in configs.values() if c.is_baseline)
    return find_config(selector, configs)
