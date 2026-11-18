# ECE 4300 — Rubric Compliance Checklist

**Status on 13 Nov 2026: 4 of 7 requirements met with evidence; 3 pending.**
The pending items depend on files and runs on the measurement Mac (raw
CSVs, graphs, demo run). Paths are relative to `ECE4300_Object_Detection/`.

| Status | Meaning |
|---|---|
| **MET** | Evidence exists in the repository and was checked |
| **PENDING** | Implemented or reported, but the evidence (data file or run) has not been checked |

| # | Requirement | Status | Evidence | What is missing |
|---|---|---|---|---|
| 1 | Clearly defined computational workload | **MET** | 100 fixed COCO 2017 val images in a fixed order (`data/benchmark/image_manifest.csv`, all 100 SHA-256 checksums re-checked in Step 7). Pretrained YOLO11n, batch 1, fixed thresholds (`config/benchmark_config.json`). Timed region defined in `src/benchmark.py` (`run_trial`) and in Report §4 | — |
| 2 | Documented baseline configuration | **MET** | `"baseline": "A"` (CPU, 640×640) in `config/benchmark_config.json`. README "Step 4". Report §2, Table 1 | — |
| 3 | Meaningful alternative configurations | **MET** (design) | B, C, D change only the processor and/or resolution (verified by `validate_experiment` against the config file). Report Table 1. Measurements for all four are *reported* for session `S20261112-121717-4092` (README) | Raw-CSV confirmation is part of #5 |
| 4 | At least two quantitative performance metrics | **MET** (definition and calculation) | Execution time, time per image, throughput, speedup (`src/analyze_results.py`; formulas unit-tested in `tests/test_analysis.py` and `tests/test_validation.py`). Report §5 | The values themselves are still unverified (#5) |
| 5 | Repeatable experiments, ≥ 5 measured trials per configuration | **PENDING** | `num_trials: 5` in the config file. README reports 5 trials × 100 images for A–D | Run `python -m src.validate_experiment --session S20261112-121717-4092` on the Mac. It must report no FAIL in "Raw trials" and "Same workload" |
| 6 | ≥ 2 meaningful graphs or tables with technical analysis | **PENDING** | Tables: Report Tables 3 and 4 (reported values) with analysis in Report §7. Graphs: 3 PNGs generated in Step 6 (per README), in the Mac's `results/graphs/` | Verify the tables against the raw CSVs (#5). Copy the three PNGs into `deliverables/performance_graphs/` and the report |
| 7 | Working and reproducible benchmarking demonstration | **PENDING** | `python -m src.benchmark --config C --demo 20` (Step 8 option on the existing benchmark program; unit-tested in `tests/test_benchmark.py::TestQuickDemo`, 4/4 passed on Windows with placeholder modules). Guide: `deliverables/demo_guide.md` | Run the demo once on the Mac (configs C and A) and confirm a CSV appears in `results/demo/` |

## Related items that are not rubric requirements

| Item | Status |
|---|---|
| Detection quality (mAP50 at 640 vs. 320) | Not measured. `python -m src.evaluate_accuracy` on the Mac |
| Software versions in the report | Recorded/pinned values only. Confirm with `python tests/check_environment.py` |
| Word / PDF report, PowerPoint file | Not generated; Markdown drafts only (`ECE4300_Final_Report.md`, `ECE4300_Final_Presentation.md`) |
| Step 7 validation report | `results/validation_report.md`: methodology checks PASS; data checks PENDING |

## To reach 7/7

On the Mac, from `ECE4300_Object_Detection` with `.venv` active:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m src.validate_experiment --session S20261112-121717-4092
python -m src.evaluate_accuracy
python -m src.benchmark --config C --demo 20
python -m src.benchmark --config A --demo 20
cp results/graphs/*_S20261112-121717-4092.png deliverables/performance_graphs/
```

Then update the report and slides with the verified values, the min/max, the
graphs, and the mAP50 results, and change rows 5–7 to MET if those runs pass.
