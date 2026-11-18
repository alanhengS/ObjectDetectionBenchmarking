# ECE 4300 — Step 7 Validation Report

**Project:** Object Detection Performance Benchmarking on Apple Silicon
**Validated session:** `S20261112-121717-4092` (Step 5 measurements, Step 6 analysis)
**Report date:** 13 Nov 2026 (methodology); updated 17 Nov 2026 (Mac run)
**Overall status:** **Ready for final reporting.** All checks pass. On the Mac,
`python -m src.validate_experiment --session S20261112-121717-4092` reported
**114 passed, 0 failed, 0 skipped**, all 92 unit tests passed, and detection
quality was measured (Section 4).

### How to read this report

| Status | Meaning |
|--------|---------|
| **PASS** | Checked, with the evidence named in the same row |
| **FAIL** | Checked and found wrong |

**Where this validation was done.** The code and methodology checks were first
run on 13 Nov 2026 on a Windows copy of the repository (standard-library Python
only). The data, graph, unit-test, and accuracy checks were run on 17 Nov 2026
on the measurement Mac (Apple M2, macOS 26.1, Python 3.13.7, PyTorch 2.15.0,
Ultralytics 8.4.202), where the raw measurement files and weights are stored.
Full validator output:
`results/validation/data_validation_S20261112-121717-4092_20261117-191329.md`.

---

## 1. Project Configuration

| Item | Value | Source |
|------|-------|--------|
| Hardware | Apple M2 Mac, 8 GB unified memory, macOS 26.1 (arm64) | `tests/check_environment.py` and `validate_experiment` on the Mac |
| Software | Python 3.13.7, PyTorch 2.15.0 (MPS), Ultralytics 8.4.202, OpenCV 5.0.0.93, matplotlib | `requirements.txt`; confirmed on the Mac |
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
| Same weights file in every run | **PASS** | One `model` setting for all. `model_sha256` = `0ebbc80d4a7680d1` in all four runs, matching `models/yolo11n.pt` on disk |
| Same 100 images, same order | **PASS** | 100 images, all SHA-256 checksums match the manifest, which is sorted by ID. `image_list_sha256` = `a32a96557777e56a` in all four runs, matching the current manifest |
| GPU runs really use MPS, with no silent CPU fallback | **PASS** | [benchmark.py:161-190](../src/benchmark.py#L161-L190) stops if MPS is not built or not available, or if `PYTORCH_ENABLE_MPS_FALLBACK=1`. [benchmark.py:222-237](../src/benchmark.py#L222-L237) checks that the predictor, weights, and output boxes are on `mps` before ([line 490](../src/benchmark.py#L490)) and after ([line 508](../src/benchmark.py#L508)) the trials. If any check fails, no timing CSV is written. Unit tests `tests/test_configurations.py` (`TestDeviceChecks`) pass on the Mac. Raw CSVs record `device = mps` for C and D |
| Requested resolution actually applied | **PASS** | `predict()` letterboxes each image to the smallest multiple-of-32 rectangle (`rect=True`). Calculated from the 100 image sizes: **14 input shapes at 640 and 9 at 320**, with the longest side always 640 or 320 (e.g. 640×426 → 448×640 at 640 and 224×320 at 320). This matches the README and Ultralytics' own `LetterBox` for all image sizes at both resolutions (`validate_experiment`) |

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
| ≥ 5 trials each for CPU-640, CPU-320, GPU-640, GPU-320 | **PASS** | 5 trials each in the raw CSVs of session `S20261112-121717-4092` |
| Trial numbers 1..N, no duplicates or gaps | **PASS** | Trials 1–5 in every run |
| Positive, valid execution times | **PASS** | CPU-640 2.964–3.038 s, CPU-320 0.955–0.983 s, GPU-640 1.716–1.804 s, GPU-320 1.116–1.171 s |
| 100 images in every trial | **PASS** | 100 images × 5 trials in every run |
| Same settings, model, and dataset version in all runs | **PASS** | Identical model hash, image-list hash, dataset ID, warm-up (100), threads (4), and timing boundary |
| Incompatible sessions never combined | **PASS** (code) | `analyze_results` and `validate_experiment` use one session only. `compare_results` now warns when its default mode mixes sessions (Section 7.3) |

The validator rejects a whole run if any trial is invalid. It never drops
trials silently, and this is covered by tests with deliberately broken data
in `tests/test_validation.py` (missing, duplicate, and gapped trials, zero time,
wrong image count, different weights, timing boundary, and device).

### 2.4 Custom-image detection is independent of the benchmark

| Check | Status | Evidence |
|-------|--------|----------|
| Uses its own images and settings | **PASS** (code) | `src/main.py` reads `data/images/` ([main.py:31](../src/main.py#L31)) with its own fixed CPU / 640 settings ([detector.py:27-28](../src/detector.py#L27-L28)). It does not read the benchmark config, manifest, or results. The benchmark reuses only `load_model()` from `detector.py` |
| Still runs | **PASS** | `python src/main.py` on the Mac: bus.jpg → 5 objects (bus 0.94, people 0.89/0.88/0.86/0.62), as in the README |

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
| Recalculated from the real raw CSVs | the same formulas, implemented a second time in `validate_experiment.recalculate` | — | **PASS** |

**Test evidence (17 Nov 2026, on the Mac):**
`python -m unittest discover -s tests -p "test_*.py"`: **92 tests, all passed**,
including the image-list hash test and the graph-rendering tests that could not
run on the Windows copy.

**Speedup comparisons.** Each pair below is checked by `validate_experiment`
(PASS from the config file):

| Comparison | Reference / Comparison | Type |
|------------|------------------------|------|
| Resolution effect on CPU | CPU-640 / CPU-320 | Different workload (resolution); not a hardware speedup |
| Resolution effect on GPU | GPU-640 / GPU-320 | Different workload (resolution); not a hardware speedup |
| CPU vs. GPU at 640×640 | CPU-640 / GPU-640 | Same workload, different processor |
| CPU vs. GPU at 320×320 | CPU-320 / GPU-320 | Same workload, different processor |

**README values (session `S20261112-121717-4092`) against the independent
recalculation from the raw CSVs:** all match.

| Item | Reported | Arithmetic check | Status |
|------|----------|------------------|--------|
| ms/image = mean time × 10 | 29.91, 9.68, 17.47, 11.40 | 2.991, 0.968, 1.747, 1.140 s × 10 | consistent |
| images/s ≈ 100 / mean time | 33.44, 103.29, 57.25, 87.77 | 33.43, 103.31, 57.24, 87.72 | consistent within rounding (see note) |
| A/B, C/D, A/C, B/D | 3.09, 1.53, 1.71, 0.85 | 3.089, 1.533, 1.712, 0.850 (from raw CSVs) | **PASS** |

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
| **Results** | **PASS — measured** | `results/accuracy/accuracy_20261117-191334.csv` (table below) |

| Input | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 640×640 | 0.7543 | 0.5183 | 0.6215 | 0.4667 |
| 320×320 | 0.6038 | 0.4244 | 0.4869 | 0.3625 |

320×320 lowers mAP50 by 0.135 (−21.7 % relative) and recall by 0.094. The
faster resolution therefore costs detection quality.

**Limitations:** 100 images is a small sample, so mAP differences of a few
points may not generalize. Ultralytics' mAP can differ slightly from the
official pycocotools evaluator. Accuracy is measured with conf 0.001, while
the benchmark detects with conf 0.25. These results are **separate from the
performance measurements**: they show whether the faster 320×320 input costs
detection quality, not anything about the hardware.

---

## 5. Graph Verification

The graph code was reviewed, and the PNG files were checked on the Mac.

| Check | Status | Evidence |
|-------|--------|----------|
| Graphs use actual measured results | **PASS** | Graphs are drawn only from validated raw CSVs. All three PNGs are byte-identical to a fresh render of the raw data. `validate_experiment` re-renders them from the raw data and compares the files byte for byte |
| All four configurations labelled correctly | **PASS** (code) | Labels "CPU 640×640", "CPU 320×320", "GPU 640×640", "GPU 320×320", checked by `validate_experiment` |
| Execution time in seconds | **PASS** (code) | Y axis "Average execution time (seconds)" |
| Throughput in images/s | **PASS** (code) | Y axis "Throughput (images processed per second)" |
| Speedup dimensionless (×) | **PASS** (code) | Y axis "Speedup factor (×)", with a reference line at 1.0× |
| CPU/GPU speedup numerator and denominator | **PASS** (code) | CPU mean time / GPU mean time (A/C at 640, B/D at 320), [analyze_results.py:684](../src/analyze_results.py#L684) |
| Graphs match the CSV summaries | **PASS** | Plotted bar values equal the recalculation (`validate_experiment`) |
| No incomplete or invalid trials included | **PASS** | The analysis rejects the whole session on any invalid trial; all 20 trials are valid |
| Axes and units readable, 300 dpi | **PASS** | 10–14 pt text (8.5 pt footnotes), white background. All three PNGs are 2100×1380 px at 300 dpi |
| Measurements kept separate from assumptions | **PASS** (code) | Graphs show only measured values, plus the formula in a footnote. Hypotheses appear only in the analysis report, under "Possible explanations (hypotheses — not measured)" |
| ≥ 2 meaningful graphs or tables for the final report | **PASS** | 3 graphs (copied to `deliverables/performance_graphs/`), a summary CSV, and a speedup CSV |

**Errors found and corrected in the graphs:** none.

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
- Detection quality was measured separately from timing, on the same 100 images (Section 4).

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
- Unit tests: all 92 pass on the Mac.
- Raw data: 5 valid trials × 100 images for each of A–D, all from one session, identical model and image-list hashes (`validate_experiment`: 114 passed, 0 failed).
- Step 5 and Step 6 summaries, speedups, and the report's claims equal an independent recalculation from the raw CSVs.
- All three graphs are 300 dpi and byte-identical to a fresh render of the raw data.
- Detection quality measured: mAP50 0.622 (640) vs. 0.487 (320).
- Custom-image detection (`src/main.py`) still works.

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

### 7.4 Outstanding issues
None. The Mac run (17 Nov 2026) completed every check that was pending:

```bash
python -m unittest discover -s tests -p "test_*.py" -v   # 92 tests OK
python tests/validate_benchmark_dataset.py               # RESULT: PASS
python src/main.py                                       # 5 objects detected
python -m src.validate_experiment --session S20261112-121717-4092   # 114 passed, 0 failed
python -m src.evaluate_accuracy                          # mAP50 0.622 / 0.487
```

**Measurements to repeat:** none.

### 7.5 Ready for Step 8?

**Yes.** `validate_experiment` reported no FAIL, and the accuracy values are
recorded in Section 4.
