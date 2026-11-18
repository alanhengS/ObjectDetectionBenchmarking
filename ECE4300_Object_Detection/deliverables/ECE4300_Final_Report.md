# Object Detection Performance Benchmarking on Apple Silicon

**ECE 4300 – Computer Architecture — Final Project Report**

**Team members:** Alan Heng, Kevin Davis

> **DRAFT — NOT FOR SUBMISSION.** The numbers in this draft are the
> results *reported* for benchmark session `S20261112-121717-4092` in the
> project README. They have **not yet been verified against the raw trial
> CSV files**, which are on the measurement Mac (Step 7 validation is
> pending). The minimum/maximum trial times, the figures, the detection-
> quality results, and the installed software versions are marked
> **[PENDING]**. Every pending item is listed in Section 12. The Word and
> PDF versions have not been produced yet.

---

## 1. Introduction

**Object detection** is a computer-vision task in which a program finds
objects in an image and, for each one, outputs a class label (for example
"person" or "bus"), a confidence score, and a bounding box. Modern detectors
are convolutional neural networks. Running a trained network on new images
is called **inference**, and it requires a large, regular amount of
arithmetic for every image.

**Why computational performance matters.** Applications such as video
analysis, robotics, and driver assistance must process images at a sustained
rate. How long one image takes and how many images per second a system can
process depend not only on the network but on the **hardware that executes
it** (CPU or GPU) and on the **size of the workload** (the input
resolution).

**Purpose.** This project measures the inference performance of a fixed,
pretrained detector (YOLO11n) on one computer, an Apple M2 Mac. It compares
two processors in the same chip, the CPU and the GPU, at two input
resolutions.

**Relationship to computer architecture.** The project applies the
performance concepts of ECE 4300: **execution time**, **throughput**, and
**relative performance (speedup)**. It treats the detector as a fixed
workload and observes how the execution behavior of the computing system
changes when (a) the processor executing the workload changes and (b) the
amount of work per image changes. No network is trained and no processor is
designed. The project is an experimental performance evaluation of an
existing system.

## 2. Project Objectives

The project answers three research questions:

1. **Resolution:** How does reducing the input resolution from 640×640 to
   320×320 affect execution time and throughput?
2. **Processor:** How does Apple M2 GPU execution (through PyTorch's Metal
   Performance Shaders backend, MPS) compare with CPU execution at the same
   resolution?
3. **Trade-offs:** What performance trade-offs result from the choice of
   resolution and processor?

**Baseline and alternatives.** Configuration A (CPU, 640×640) is the
baseline. The default YOLO input size is 640, and the CPU is available on
every computer. Configurations B, C, and D each change one or both factors
(Table 1).

**Table 1. Benchmark configurations**

| Config. | Processor | Input resolution | Role |
|---|---|---|---|
| A | Apple M2 CPU | 640×640 | Baseline |
| B | Apple M2 CPU | 320×320 | Resolution change on the CPU |
| C | Apple M2 GPU (MPS) | 640×640 | Processor change at 640×640 |
| D | Apple M2 GPU (MPS) | 320×320 | Processor change at 320×320 |

## 3. Experimental Setup

**Table 2. Hardware and software**

| Item | Value | Source |
|---|---|---|
| Computer | Apple M2 Mac | Project README |
| CPU | Apple M2, 4 performance + 4 efficiency cores | Project README |
| GPU | Apple M2 integrated GPU, used through PyTorch MPS | Project configuration |
| Memory | 8 GB unified memory (shared by CPU and GPU) | Project README |
| Operating system | macOS 26.1 (arm64) | Project README **[PENDING: confirm on Mac]** |
| Python | 3.13.7 | Project README **[PENDING: confirm on Mac]** |
| PyTorch | 2.15.0 (MPS backend) | `requirements.txt` pin **[PENDING: confirm on Mac]** |
| Ultralytics | 8.4.202 | `requirements.txt` pin **[PENDING: confirm on Mac]** |
| OpenCV | 5.0.0.93 | `requirements.txt` pin |
| Model | YOLO11n, official pretrained weights (`yolo11n.pt`), inference only | Ultralytics |
| Dataset | First 100 images by ID of the COCO 2017 validation set (IDs 139–9891; 714 annotated objects) | `data/benchmark/` |
| Input resolutions | 640×640 and 320×320 | Configuration file |
| Batch size | 1 (images processed one at a time) | Configuration file |

The versions marked PENDING are the recorded or pinned values. They must be
confirmed on the measurement Mac (`python tests/check_environment.py`) before
submission.

**Execution configurations.** All four configurations use one shared
configuration file (`config/benchmark_config.json`). Only the `device`
(`cpu` or `mps`) and `image_size` (640 or 320) settings differ. The other
settings are identical: confidence threshold 0.25, non-maximum-suppression
IoU threshold 0.7, 4 PyTorch CPU threads, 100 warm-up images, and 5 measured
trials.

## 4. Benchmarking Methodology

**Fixed workload.** Each trial processes the same 100 COCO images, in the
order fixed by a manifest file that records each image's SHA-256 checksum.
The images are decoded into memory before timing, so disk access is not
measured.

**Same model and settings.** All configurations load the same
`yolo11n.pt` file. Each run records the SHA-256 hash of the weights and a
hash of the ordered image list in its CSV, so it can be checked that all
four configurations did identical work. Every image is processed with one
`model.predict()` call with batch size 1 and the same thresholds.

**Input resolution.** YOLO resizes each image so that its longer side
equals the requested size (640 or 320) and pads the shorter side to a
multiple of 32 ("letterboxing"). The 100 images therefore produce 14
different input shapes at 640 and 9 at 320 (for example 448×640 and
224×320). This was calculated from the image sizes in Step 7, and the
behavior is identical for the CPU and the GPU.

**Warm-up.** Before timing, every configuration runs one complete untimed
pass over all 100 images. This removes one-time setup costs from the
measurements, such as the GPU preparing its computations for each new input
shape.

**Timing boundary.** The timer, Python's `time.perf_counter()`, starts
immediately before the first image and stops immediately after the 100th.
The measured **elapsed workload processing time** includes:
- YOLO preprocessing (resize, normalization, conversion to a tensor),
- the neural-network forward pass, and
- postprocessing (non-maximum suppression, building the result objects).

It excludes model loading, image decoding, the device check, warm-up,
printing, and file writing. It is wall-clock time for the whole workload, not
pure CPU time or neural-network-only time.

**Making CPU and GPU measurements comparable.** GPU work through MPS is
*asynchronous*: a Python call can return before the GPU has finished. The
benchmark therefore calls `torch.mps.synchronize()` after every image inside
the timed loop. Each image is completely processed before the next one
starts, as on the CPU, and the timer never stops while GPU work is still
queued. In addition, the program checks that MPS is available and refuses
to run if PyTorch's CPU fallback is enabled. It also confirms, before and
after the trials, that the model weights and outputs are really on the
requested device. A GPU configuration therefore cannot silently run on the
CPU.

**Repetition and raw data.** Each configuration was measured in **5 trials**
using the same loaded model. All four configurations were run one after
another in a single session (`python -m src.run_all_benchmarks`), each in a
separate process. Every trial is saved as one row of a CSV file in
`results/raw/`, and existing files are never overwritten. Summaries and
graphs are calculated afterwards from these raw files.

## 5. Performance Metrics and Equations

The metrics follow the performance definitions of ECE 4300 (Lecture 1).
Variables:

- *N* = number of images in the workload (*N* = 100)
- *T<sub>i</sub>* = measured execution time of trial *i*, in seconds
- *k* = number of measured trials (*k* = 5)

**Execution time (s).** The measured elapsed time to complete the
100-image workload in one trial, *T<sub>i</sub>*.

**Mean execution time (s).**

  T̄ = (T<sub>1</sub> + T<sub>2</sub> + … + T<sub>k</sub>) / k

**Average processing time per image (ms/image).**

  t<sub>image</sub> = T̄ / N × 1000

**Throughput (images/s).** Throughput is calculated for each trial as
*N / T<sub>i</sub>*, and the mean over the *k* trials is reported:

  X = (1/k) · Σ (N / T<sub>i</sub>)

Because it averages per-trial values, the mean throughput can be very slightly
higher than *N / T̄*.

**Speedup (dimensionless, ×).** Using mean execution times:

  Speedup = T̄<sub>reference</sub> / T̄<sub>comparison</sub>

A speedup above 1 means the comparison configuration is faster. A speedup
below 1 means it is **slower**: it took 1/Speedup times as long.

**Variability (s).** Minimum and maximum trial times, and their range
(maximum − minimum).

## 6. Experimental Results

All values below are from benchmark session `S20261112-121717-4092` (12 Nov
2026), as **reported** in the project README: 5 trials per configuration,
100 images per trial.

**Table 3. Measured performance by configuration (reported; verification
pending)**

| Config. | Processor | Resolution | Mean execution time (s) | Time per image (ms) | Mean throughput (images/s) | Min. time (s) | Max. time (s) |
|---|---|---|---|---|---|---|---|
| A | CPU | 640×640 | 2.991 | 29.91 | 33.44 | [PENDING] | [PENDING] |
| B | CPU | 320×320 | 0.968 | 9.68 | 103.29 | [PENDING] | [PENDING] |
| C | GPU (MPS) | 640×640 | 1.747 | 17.47 | 57.25 | [PENDING] | [PENDING] |
| D | GPU (MPS) | 320×320 | 1.140 | 11.40 | 87.77 | [PENDING] | [PENDING] |

*The minimum and maximum come from the raw trial CSVs, which are not
available for this draft. The Step 5 output reported standard deviations of
0.028, 0.011, 0.037, and 0.021 s for A–D, which is 0.9 %, 1.1 %, 2.1 %, and
1.8 % of the respective means.*

**Table 4. Speedups (mean execution times)**

| Comparison | Type | Calculation | Speedup | Result |
|---|---|---|---|---|
| CPU 640 → CPU 320 (A / B) | Resolution | 2.991 / 0.968 | 3.09× | 320×320 faster on the CPU |
| GPU 640 → GPU 320 (C / D) | Resolution | 1.747 / 1.140 | 1.53× | 320×320 faster on the GPU |
| CPU 640 → GPU 640 (A / C) | Processor | 2.991 / 1.747 | 1.71× | GPU faster at 640×640 |
| CPU 320 → GPU 320 (B / D) | Processor | 0.968 / 1.140 | 0.85× | **GPU slower** at 320×320 (took 1.18× as long) |

The resolution rows compare two different workloads on the same processor.
The processor rows compare the same workload on two processors, so only those
are hardware speedups.

**Figure 1. Average object-detection execution time (seconds) by
configuration.** [PENDING: insert
`results/graphs/execution_time_S20261112-121717-4092.png`. Bars show the
mean of 5 trials; whiskers show the fastest and slowest trial.]

**Figure 2. Object-detection throughput (images/s) by configuration.**
[PENDING: insert `results/graphs/throughput_S20261112-121717-4092.png`]

**Figure 3. CPU-to-GPU speedup by input resolution, with a 1.0× reference
line.** [PENDING: insert
`results/graphs/cpu_gpu_speedup_S20261112-121717-4092.png`. Expected values
from Table 4: 0.85× at 320×320 and 1.71× at 640×640.]

## 7. Technical Analysis

### 7.1 Measured observations

**Fastest configuration.** Configuration **B (CPU, 320×320)** had the
shortest mean execution time, 0.968 s for 100 images (9.68 ms/image). It
also had the highest throughput, 103.29 images/s. Relative to the baseline
A, it reduced the execution time by 67.6 % (speedup 3.09×).

**Resolution comparison.**
- *CPU:* 640×640 → 320×320 reduced the mean time from 2.991 s to 0.968 s
  (−67.6 %), and throughput rose from 33.44 to 103.29 images/s (+208.9 %).
- *GPU:* 640×640 → 320×320 reduced the mean time from 1.747 s to 1.140 s
  (−34.7 %), and throughput rose from 57.25 to 87.77 images/s (+53.3 %).
- Reducing resolution helped on both processors, but the benefit on the CPU
  (3.09×) was about twice the benefit on the GPU (1.53×).

**Processor comparison.**
- *At 640×640:* the GPU was faster than the CPU, with a speedup of 1.71×
  (2.991 s vs. 1.747 s; throughput 57.25 vs. 33.44 images/s).
- *At 320×320:* the GPU was **slower** than the CPU, with a speedup of 0.85×
  (0.968 s vs. 1.140 s). The GPU needed 1.18× as long, and its throughput was
  15.0 % lower (87.77 vs. 103.29 images/s).
- So GPU execution improved performance only at the larger resolution. The
  faster processor depended on the workload size.

**Unexpected result.** The GPU being slower than the CPU at 320×320 was not
assumed beforehand and is the main unexpected result. A second notable
observation is that the CPU gained much more than the GPU from the smaller
input.

**Consistency.** The reported standard deviations are at most 2.1 % of the
mean in every configuration, so trial-to-trial variation is small compared
with the differences between configurations. These differences range from
15 % to 209 % in throughput. **[PENDING: confirm with the min/max range from
the raw CSVs.]**

### 7.2 Workload and resolution

At 320×320, each image is fed to the network with about one quarter of the
pixels used at 640×640 (for example, 224×320 instead of 448×640, exactly
4× fewer). The convolutional layers process every input position, so the
amount of arithmetic per image is expected to fall roughly in proportion.
This is the workload change being tested. The measured CPU speedup (3.09×) is
below 4×, and the GPU speedup (1.53×) is well below 4×.

### 7.3 Possible system-level explanations (hypotheses — not measured)

The project measured only total elapsed time per trial. The following are
possible explanations consistent with the measurements, **not confirmed
causes**:

- **Fixed per-image overhead.** Each image carries costs that do not shrink
  with a smaller input or a faster processor: Python and Ultralytics
  bookkeeping, CPU-side preprocessing and postprocessing (which run on the
  CPU even in the GPU configurations), and, on the GPU, launching work and
  waiting for it to finish (`synchronize`) once per image. When the network
  computation shrinks at 320×320, this fixed part becomes a larger share of
  each image's time. That would reduce both the benefit of the lower
  resolution and the GPU's advantage. This is consistent with the sub-4×
  resolution speedups and with the GPU losing at 320×320.
- **Parallel work per launch.** A GPU performs best when it has a large
  amount of parallel work. At 640×640 each image provides about four times
  more work to spread across the GPU than at 320×320. This may explain why
  the GPU's advantage appears only at the larger input.
- **Batch size 1.** Processing one image at a time gives the GPU the smallest
  possible amount of work per step. Larger batches were not tested, so it is
  unknown whether they would change the comparison.

Confirming any of these would require measurements this project did not make,
such as timing preprocessing, the forward pass, and postprocessing
separately.

### 7.4 Trade-offs

- **Resolution vs. speed:** lowering the resolution gave the largest
  improvements (up to 3.09× on the CPU), but it reduces the image detail
  available to the detector. Its effect on detection quality was not
  measured in this draft (Section 8).
- **Processor choice depends on workload size:** for the 640×640 workload
  the GPU is the better choice (1.71×). For the 320×320 workload the CPU is
  (0.85× for the GPU). Neither processor is better for every workload.
- **Best configuration overall:** CPU at 320×320, if the lower resolution is
  acceptable for the application. If 640×640 is required, the GPU should be
  used.

## 8. Detection Quality

**Status: not measured.** A quantitative evaluation was prepared in Step 7
(`src/evaluate_accuracy.py`) but has not been run yet. It uses Ultralytics'
standard validation procedure on the same 100 images and their COCO ground
truth (709 non-crowd objects), with the same weights at 640×640 and 320×320,
and reports precision, recall, mAP50, and mAP50-95.

No qualitative (visual) comparison of the two resolutions has been recorded
either. This report therefore **makes no claim** about whether detection
quality is maintained at 320×320. The speed advantage of 320×320 in
Section 7 must be read with this limitation. **[PENDING: insert the mAP50
results from `results/accuracy/accuracy_<time>.csv`.]**

## 9. Experimental Limitations

- **One computer.** All results come from a single Apple M2 Mac with 8 GB.
  Other Apple chips, discrete GPUs, or x86 processors may give different
  speedups, including a different winner at either resolution.
- **Limited workload.** 100 COCO images, one small model (YOLO11n), and batch
  size 1. Larger models, batches, or other image sets were not tested.
- **Background operating-system activity.** macOS services and other
  applications share the processors during measurement. Their effect is not
  controlled, only reduced: other apps were closed and all configurations
  were run in one session.
- **Unified memory.** The CPU and GPU share the same 8 GB of memory and its
  bandwidth. This avoids copying data to a separate GPU memory, but memory
  used by other software can affect both processors.
- **Possible thermal effects.** Sustained load can raise the chip's
  temperature and reduce its clock speed, especially in a fanless Mac.
  Configurations run later in the session may have been affected. This was
  not measured.
- **Measurement variability.** Wall-clock times include operating-system
  scheduling. Five trials per configuration were averaged, and variation
  was small (Section 7.1).
- **Software overhead.** The measured times include Python and Ultralytics
  processing, not only the neural network, and the GPU times include
  CPU-side work. The results describe this complete software stack, not the
  peak capability of the hardware.
- **Generalizability.** The conclusions apply to this system, these
  software versions, and this workload.
- **Verification status of this draft.** See Section 12.

## 10. Conclusion

Based on the reported measurements of session `S20261112-121717-4092`:

1. **Resolution:** Reducing the input from 640×640 to 320×320 reduced the
   execution time on both processors: by 67.6 % on the CPU (speedup 3.09×,
   throughput 33.44 → 103.29 images/s) and by 34.7 % on the GPU (speedup
   1.53×, throughput 57.25 → 87.77 images/s). The CPU benefited much more.
2. **Processor:** At 640×640 the M2 GPU was 1.71× faster than the CPU. At
   320×320 it was slower (speedup 0.85×; 1.18× as long). GPU execution did
   not improve performance for every workload.
3. **Trade-offs:** The fastest configuration was the CPU at 320×320 (0.968 s
   per 100 images, 103.29 images/s, 3.09× faster than the baseline). The
   better processor depended on the size of the work per image. Lower
   resolution gave the largest gains, and its cost in detection quality
   still has to be measured.

The experiment shows that execution time and throughput depend on how the
workload and the processor interact, not on the processor alone. For small
per-image workloads, costs other than the main computation may limit what a
faster processor can achieve. This explanation is a hypothesis that this
project did not measure directly.

## 11. References

**Course material (performance concepts)**

[1] ECE 4300 Computer Architecture, Lecture 1: performance, execution time,
throughput, relative performance and speedup. [Instructor, term, and
textbook to be completed by the team.]

**Software, model, and data (implementation)**

[2] G. Jocher and J. Qiu, *Ultralytics YOLO11*, version 11.0.0, 2024.
https://github.com/ultralytics/ultralytics (documentation:
https://docs.ultralytics.com/models/yolo11/).

[3] Ultralytics, "Predict mode" and "Val mode" documentation.
https://docs.ultralytics.com/modes/predict/ and https://docs.ultralytics.com/modes/val/.

[4] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep
Learning Library," *Advances in Neural Information Processing Systems 32
(NeurIPS)*, 2019.

[5] PyTorch documentation, "MPS backend." https://pytorch.org/docs/stable/notes/mps.html.

[6] T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," *European
Conference on Computer Vision (ECCV)*, 2014. Dataset: https://cocodataset.org.

[7] Python documentation, `time.perf_counter()`.
https://docs.python.org/3/library/time.html#time.perf_counter.

[8] OpenCV, Open Source Computer Vision Library. https://opencv.org.

[9] J. D. Hunter, "Matplotlib: A 2D Graphics Environment," *Computing in
Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007.

## 12. Verification Status of This Draft (remove before submission)

| Item | Status | How to complete |
|---|---|---|
| Table 3 means, ms/image, throughput; Table 4 speedups | Reported in README; internally consistent; **not verified against raw CSVs** | `python -m src.validate_experiment --session S20261112-121717-4092` on the Mac |
| Table 3 min/max times | Missing | From the same validation run or `results/summaries/performance_summary_S20261112-121717-4092.csv` |
| Figures 1–3 | Exist on the Mac (per README); not inserted | Copy from `results/graphs/` |
| Software versions (Table 2) | Recorded/pinned values | `python tests/check_environment.py` on the Mac |
| Detection quality (Section 8) | Not measured | `python -m src.evaluate_accuracy` on the Mac |
| Word and PDF versions | Not generated | Convert after the data is final |
