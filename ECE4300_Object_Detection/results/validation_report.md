# ECE 4300 — Step 7 Validation Report

**Project:** Object Detection Performance Benchmarking on Apple Silicon
**Validated session:** `S20261112-121717-4092` (Step 5 measurements, Step 6 analysis)
**Report date:** 13 Nov 2026
**Overall status:** **Not yet ready for final reporting.** The code, methodology,
configuration, and dataset checks have passed. The checks on the measured data,
the graphs, and detection quality are **PENDING** a run of the Step 7 tools on
the Mac (Section 7.4).

### How to read this report

| Status | Meaning |
|--------|---------|
| **PASS** | Checked, with the evidence named in the same row |
| **PENDING** | Not yet checked: needs the Mac, its measurement files, or its Python packages |
| **FAIL** | Checked and found wrong |

**Where this validation was done.** The checks in this version were run on a
Windows copy of the repository, which has only the standard-library Python. The
measurement files (`results/raw/`, `results/summary/`, `results/summaries/`,
`results/graphs/`), the weights (`models/yolo11n.pt`), and the PyTorch,
Ultralytics, OpenCV, and matplotlib packages exist only on the Mac, because
`results/` and `*.pt` are excluded from git. Nothing below is marked PASS on
the strength of the README alone. Values quoted from the README are labelled
as *reported*.

---

## 1. Project Configuration

| Item | Value | Source |
|------|-------|--------|
| Hardware | Apple M2 Mac, 8 GB unified memory, macOS 26.1 (arm64) | README (reported). `validate_experiment` prints the environment on the Mac |
| Software | Python 3.13.7, PyTorch 2.15.0 (MPS), Ultralytics 8.4.202, OpenCV 5.0.0.93, matplotlib | `requirements.txt` (pinned) |
| Model | Pretrained YOLO11n (`models/yolo11n.pt`), inference only, not trained or modified | `config/benchmark_config.json`, `src/detector.py` |
| Dataset | First 100 COCO 2017 val images by ID (139–9891), 714 annotated objects, 80 categories, in manifest order | `data/benchmark/` (in git) |
| Prediction settings | batch 1, conf 0.25, NMS IoU 0.7, 4 PyTorch CPU threads, 100 untimed warm-up images, 5 measured trials | `config/benchmark_config.json` |

| Config | Name | Processor | Input resolution | Role |
|--------|------|-----------|------------------|------|
| A | CPU-640 | Apple M2 CPU | 640×640 | Baseline |
| B | CPU-320 | Apple M2 CPU | 320×320 | Resolution change on CPU |
| C | GPU-640 | Apple M2 GPU (MPS) | 640×640 | Same workload as A, on the GPU |
| D | GPU-320 | Apple M2 GPU (MPS) | 320×320 | Same workload as B, on the GPU |

---

## 2. Benchmark Verification

### 2.1 Configuration settings

| Check | Status | Evidence |
|-------|--------|----------|
| Exactly the four configurations CPU/GPU × 640/320 | **PASS** | `validate_experiment` on the config file: A=CPU-640, B=CPU-320, C=GPU-640, D=GPU-320 |
| Only device and resolution differ | **PASS** | `validate_experiment`: batch, conf, IoU, trials, warm-up, threads, image count, dataset, model, and manifest are identical for A–D |
| Batch size 1 | **PASS** | Config file. [benchmark_config.py:99](../src/benchmark_config.py#L99) rejects any other value |
| Same prediction call for every configuration | **PASS** | [benchmark.py:243-254](../src/benchmark.py#L243-L254): one `predict_one()` that changes only `device` and `imgsz` |
| Same weights file in every run | **PASS** (code) / **PENDING** (data) | Code: one `model` setting for all. The recorded `model_sha256` values still need comparing (Section 7.4) |
| Same 100 images, same order | **PASS** (files) / **PENDING** (data) | Here: 100 images, all SHA-256 checksums match the manifest, which is sorted by ID. The recorded `image_list_sha256` still needs comparing to the manifest |
| GPU runs really use MPS, with no silent CPU fallback | **PASS** (code) / **PENDING** (tests) | [benchmark.py:161-190](../src/benchmark.py#L161-L190) stops if MPS is not built or not available, or if `PYTORCH_ENABLE_MPS_FALLBACK=1`. [benchmark.py:222-237](../src/benchmark.py#L222-L237) checks that the predictor, weights, and output boxes are on `mps` before ([line 490](../src/benchmark.py#L490)) and after ([line 508](../src/benchmark.py#L508)) the trials. If any check fails, no timing CSV is written. Unit tests: `tests/test_configurations.py` (`TestDeviceChecks`) need PyTorch, so they are pending |
| Requested resolution actually applied | **PASS** (calculation) / **PENDING** (Ultralytics cross-check) | `predict()` letterboxes each image to the smallest multiple-of-32 rectangle (`rect=True`). Calculated from the 100 image sizes: **14 input shapes at 640 and 9 at 320**, with the longest side always 640 or 320 (e.g. 640×426 → 448×640 at 640 and 224×320 at 320). This matches the README. The comparison with Ultralytics' own `LetterBox` runs on the Mac |

### 2.2 Timing methodology

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Model loaded before timing | **PASS** | [benchmark.py:479](../src/benchmark.py#L479) runs before the trials at [line 503](../src/benchmark.py#L503) |
| Dataset preparation outside the timed section | **PASS** | All 100 JPEGs are decoded into memory first ([line 471](../src/benchmark.py#L471)). The timer covers only the `predict()` loop |
| Warm-up excluded | **PASS** | [line 493](../src/benchmark.py#L493): one full untimed pass of 100 images, before timing |
| Same stages timed in every configuration | **PASS** | One `run_trial()` for all four ([benchmark.py:266-281](../src/benchmark.py#L266-L281)): preprocessing, forward pass, and postprocessing inside `predict()` |
| GPU synchronized before the timer stops | **PASS** | `torch.mps.synchronize()` after every image ([line 279](../src/benchmark.py#L279)), before `perf_counter()` stops ([line 280](../src/benchmark.py#L280)). It does nothing on the CPU |
| CSV writing and printing excluded | **PASS** | `print` is between trials ([line 297](../src/benchmark.py#L297)), outside the timed loop. The CSV is written after all trials ([line 517](../src/benchmark.py#L517)) |
| Consistent units | **PASS** | Seconds (`total_time_s`), ms/image, images/s, in every CSV and summary |
| Measured quantity described correctly | **PASS** | Recorded as `measurement = elapsed_workload_time` ([line 76](../src/benchmark.py#L76)). The README describes it as elapsed workload time, not CPU time or neural-network-only time |

**No inconsistency between configurations was found, so the timing method
gives no reason to repeat any measurement.**

### 2.3 Repeated trials

| Check | Status | Evidence |
|-------|--------|----------|
| ≥ 5 trials each for CPU-640, CPU-320, GPU-640, GPU-320 | **PENDING** | README reports 5 trials each for session `S20261112-121717-4092`; not yet checked against the raw CSVs |
| Trial numbers 1..N, no duplicates or gaps | **PENDING** | Checked by `validate_experiment` |
| Positive, valid execution times | **PENDING** | Checked by `validate_experiment` |
| 100 images in every trial | **PENDING** | Checked by `validate_experiment` |
| Same settings, model, and dataset version in all runs | **PENDING** | Hash and settings comparison in `validate_experiment` |
| Incompatible sessions never combined | **PASS** (code) | `analyze_results` and `validate_experiment` use one session only. `compare_results` now warns when its default mode mixes sessions (Section 7.3) |

The validator rejects a whole run if any trial is invalid. It never drops
trials silently, and this is covered by tests with deliberately broken data
in `tests/test_validation.py` (missing, duplicate, and gapped trials, zero time,
wrong image count, different weights, timing boundary, and device).

### 2.4 Custom-image detection is independent of the benchmark

| Check | Status | Evidence |
|-------|--------|----------|
| Uses its own images and settings | **PASS** (code) | `src/main.py` reads `data/images/` ([main.py:31](../src/main.py#L31)) with its own fixed CPU / 640 settings ([detector.py:27-28](../src/detector.py#L27-L28)). It does not read the benchmark config, manifest, or results. The benchmark reuses only `load_model()` from `detector.py` |
| Still runs | **PENDING** | Needs Ultralytics and `data/images/bus.jpg` (not in git). Run `python src/main.py` on the Mac |

---

## 3. Performance Calculation Verification

| Metric | Formula | Code | Status |
|--------|---------|------|--------|
| Execution time | `perf_counter()` end − start, one trial (s) | [benchmark.py:276-281](../src/benchmark.py#L276-L281) | **PASS** (code) |
| Mean execution time | sum of trial times / number of trials | `analyze_results.mean_execution_time` | **PASS** (tests) |
| Average processing time | mean time / 100 images × 1000 (ms/image) | `analyze_results.avg_processing_time_ms` | **PASS** (tests) |
| Throughput | 100 images / trial time, then averaged over trials (images/s) | `analyze_results.mean_throughput` | **PASS** (tests) |
| Min, max, range | max trial time − min trial time | `analyze_results.time_range` | **PASS** (tests) |
| Speedup | reference mean time / comparison mean time | `analyze_results.speedup` | **PASS** (tests) |
| Recalculated from the real raw CSVs | the same formulas, implemented a second time in `validate_experiment.recalculate` | — | **PENDING** |

**Test evidence (run on 13 Nov 2026 on the Windows copy):**
- `tests/test_validation.py`: **24 run, 23 passed, 1 skipped.** The skipped test
  compares the image-list hash with `benchmark.py`, which needs OpenCV.
- `tests/test_analysis.py` (Step 6): run with placeholders for cv2/torch, which
  the analysis code never calls. **19 of 21 passed.** The 2 errors are the
  graph-rendering tests, which need matplotlib, so they are pending on the Mac.

**Speedup comparisons.** Each pair below is checked by `validate_experiment`
(PASS from the config file):

| Comparison | Reference / Comparison | Type |
|------------|------------------------|------|
| Resolution effect on CPU | CPU-640 / CPU-320 | Different workload (resolution); not a hardware speedup |
| Resolution effect on GPU | GPU-640 / GPU-320 | Different workload (resolution); not a hardware speedup |
| CPU vs. GPU at 640×640 | CPU-640 / GPU-640 | Same workload, different processor |
| CPU vs. GPU at 320×320 | CPU-320 / GPU-320 | Same workload, different processor |

**Reported values (README, session `S20261112-121717-4092`): consistency check
only.** These have not yet been checked against the raw CSVs.

| Item | Reported | Arithmetic check | Status |
|------|----------|------------------|--------|
| ms/image = mean time × 10 | 29.91, 9.68, 17.47, 11.40 | 2.991, 0.968, 1.747, 1.140 s × 10 | consistent |
| images/s ≈ 100 / mean time | 33.44, 103.29, 57.25, 87.77 | 33.43, 103.31, 57.24, 87.72 | consistent within rounding (see note) |
| A/B, C/D, A/C, B/D | 3.09, 1.53, 1.71, 0.85 | 3.090, 1.532, 1.712, 0.849 | consistent |

*Note:* the mean of the trial throughputs is always slightly above
100 / mean time when trial times vary, which explains the 0.05 difference for
D (std 0.021 s).

The README correctly reports **B/D = 0.85 as the GPU being slower** at
320×320 (it took 1/0.849 = 1.18× as long), not as an improvement.

---

## 4. Detection Quality

| Item | Status | Detail |
|------|--------|--------|
| Ground-truth annotations available | **PASS** | `instances_val2017_subset.json`: 100 images, 714 objects, 80 categories in the same order as YOLO11n's classes |
| Compatible with the existing software | **PASS** | No new dependency is needed. The COCO boxes are converted to YOLO labels (709 boxes; the 5 crowd boxes are skipped, as in Ultralytics' COCO converter). The conversion is unit-tested |
| Method | — | Ultralytics' built-in validation (`model.val()`) at 640 and 320. Same weights, batch 1, IoU 0.7, and conf 0.001 (the Ultralytics validation default, which mAP needs). `src/evaluate_accuracy.py` |
| **Results** | **PENDING — not measured** | Needs Ultralytics and the weights on the Mac: `python -m src.evaluate_accuracy`. No accuracy values are reported here |

**Limitations:** 100 images is a small sample, so mAP differences of a few
points may not generalize. Ultralytics' mAP can differ slightly from the
official pycocotools evaluator. Accuracy is measured with conf 0.001, while
the benchmark detects with conf 0.25. These results are **separate from the
performance measurements**: they show whether the faster 320×320 input costs
detection quality, not anything about the hardware.

---

## 5. Graph Verification

The graph code was reviewed. The PNG files themselves are not on this computer.

| Check | Status | Evidence |
|-------|--------|----------|
| Graphs use actual measured results | **PASS** (code) / **PENDING** (files) | Graphs are drawn only from validated raw CSVs. `validate_experiment` re-renders them from the raw data and compares the files byte for byte |
| All four configurations labelled correctly | **PASS** (code) | Labels "CPU 640×640", "CPU 320×320", "GPU 640×640", "GPU 320×320", checked by `validate_experiment` |
| Execution time in seconds | **PASS** (code) | Y axis "Average execution time (seconds)" |
| Throughput in images/s | **PASS** (code) | Y axis "Throughput (images processed per second)" |
| Speedup dimensionless (×) | **PASS** (code) | Y axis "Speedup factor (×)", with a reference line at 1.0× |
| CPU/GPU speedup numerator and denominator | **PASS** (code) | CPU mean time / GPU mean time (A/C at 640, B/D at 320), [analyze_results.py:684](../src/analyze_results.py#L684) |
| Graphs match the CSV summaries | **PENDING** | `validate_experiment` |
| No incomplete or invalid trials included | **PASS** (code) / **PENDING** (data) | The analysis rejects the whole session on any invalid trial |
| Axes and units readable, 300 dpi | **PASS** (code) / **PENDING** (files) | 10–14 pt text (8.5 pt footnotes), white background. The dpi is read from the PNG header by `validate_experiment` |
| Measurements kept separate from assumptions | **PASS** (code) | Graphs show only measured values, plus the formula in a footnote. Hypotheses appear only in the analysis report, under "Possible explanations (hypotheses — not measured)" |
| ≥ 2 meaningful graphs or tables for the final report | **PASS** (code) / **PENDING** (files) | 3 graphs, a summary CSV, and a speedup CSV are produced |

**Errors found and corrected in the graphs:** none in the graph code. File
checks are pending.

---

## 6. Experimental Limitations

These are possible influences, not measured causes. No measured result is
attributed to any of them.

**Environmental factors**
- macOS background activity (Spotlight, iCloud/OneDrive sync, Time Machine,
  other apps) shares the CPU, the GPU, and memory bandwidth during trials.
- Power source and temperature: a fanless Mac can throttle during sustained
  load, and configurations run later in a session may run warmer.

**Resource constraints**
- A single computer: one Apple M2 with 8 GB of unified memory. Results may
  differ on other Apple chips, discrete GPUs, or x86 CPUs.
- The CPU and GPU share the same memory, so memory pressure from other apps
  slows both.
- Only the small YOLO11n model at batch size 1 was tested. Larger models or
  batches may change the CPU/GPU comparison.

**Measurement limitations**
- Only total elapsed time per trial is measured. GPU times include CPU-side
  pre/postprocessing, GPU launch, and per-image synchronization, so they are
  not GPU compute time. Any reason given for a difference is a hypothesis.
- Wall-clock timing includes operating-system scheduling. Trial-to-trial
  variation is reported through the minimum, maximum, and range.
- 640 → 320 compares different workloads (about 4× fewer input pixels), not
  different hardware.
- 100 images do not represent every object-detection scenario. The input
  shape varies with the aspect ratio (rect letterboxing), but identically
  for every configuration.
- Detection quality was not part of the timing experiment (Section 4).

---

## 7. Final Validation Summary

### 7.1 Passed checks (with evidence)
- The configuration file defines exactly A–D, with only device and resolution differing, batch 1, and ≥ 5 trials.
- The speedup pairs use the correct reference / comparison in all four comparisons.
- Timing boundary: all eight requirements in Section 2.2, verified by code inspection.
- The GPU no-fallback design: availability check, fallback-variable refusal, and device verification before and after the trials.
- Input resolution: 14 (640) and 9 (320) letterbox shapes, with the longest side equal to the requested size.
- The benchmark dataset: 100 images, checksums match, manifest sorted by ID, and annotations compatible with mAP evaluation.
- Custom-image detection is independent of the benchmark (code).
- Formula tests: 23/24 Step 7 tests and 19/21 Step 6 tests passed on this computer. The rest need the Mac.
- The reported Step 5 values are internally consistent (Section 3).

### 7.2 Failed checks
- None found.

### 7.3 Issues corrected in Step 7
1. **`compare_results` could mix sessions without warning.** In its default
   mode (no `--session`), it takes the latest run of each configuration, which
   may come from different sessions, and warned only if the hashes differed.
   It now prints a warning when the runs come from different sessions.
   *Impact:* the reported Step 5 and Step 6 results came from one session, so
   **no measurement needs to be repeated.**
2. **`analyze_results` did not reject gaps in trial numbers** (e.g. 1, 2, 3, 4, 6).
   It caught duplicates but not gaps. It now rejects them. *Impact:* none on
   valid data.
3. `results/validation_report.md` is now tracked in git (`.gitignore`), and the
   derived accuracy labels are excluded.
4. The README now has a Step 7 section: validation commands, accuracy method,
   reproducibility checklist, and limitations.

### 7.4 Outstanding issues: run on the Mac
From the project folder with `.venv` active:

```bash
git pull                                         # after the Step 7 files are committed
python -m unittest discover -s tests -p "test_*.py" -v
python tests/validate_benchmark_dataset.py
python src/main.py                               # needs data/images/bus.jpg
python -m src.validate_experiment --session S20261112-121717-4092
python -m src.evaluate_accuracy
```

Then update this report with:
- the trial, data, hash, summary, and graph results from
  `results/validation/data_validation_S20261112-121717-4092_<time>.md`
- the mAP50 values from `results/accuracy/accuracy_<time>.csv`
- the results of the PyTorch-dependent unit tests and the custom-image detection test.

**Measurements to repeat:** none identified. Repeat the benchmark only if
`validate_experiment` reports a FAIL in the raw-trial or same-workload
sections, using `python -m src.run_all_benchmarks` and then
`python -m src.analyze_results`.

### 7.5 Ready for Step 8?

**Not yet.** The methodology and calculations are verified. The project can be
considered ready for final reporting once the Mac run in 7.4 shows **no FAIL**
in `validate_experiment` and the accuracy values have been recorded in
Section 4.
