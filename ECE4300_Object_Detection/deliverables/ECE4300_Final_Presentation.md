# ECE4300_Final_Presentation — slide content and speaker notes (DRAFT)

> **Draft source for `ECE4300_Final_Presentation.pptx`. The .pptx itself has not been created.**
> Values are the **reported** results of session `S20261112-121717-4092`
> (README), not yet verified against the raw CSVs. Graph slides need the
> PNGs from the Mac's `results/graphs/`. Remove every `[PENDING]` before
> presenting.
>
> Suggested style: white background, dark gray text, one accent color
> (e.g. dark blue) for headings and the GPU bars. At least 24 pt body text
> and at most 5 bullets per slide.
>
> Speaking order alternates: **Alan** speaks on slides 1, 3, 5, and 7;
> **Kevin** speaks on slides 2, 4, 6, and 8. The notes do not describe who did
> which part of the work.

---

## Slide 1 — Project Overview

**Object Detection Performance Benchmarking on Apple Silicon**
ECE 4300 Computer Architecture — Alan Heng, Kevin Davis

- **Workload:** pretrained YOLO11n object detection on 100 fixed COCO images
- **Q1:** How does lower input resolution change execution time and throughput?
- **Q2:** How does the M2 GPU compare with the M2 CPU at the same resolution?
- **Q3:** What trade-offs follow from resolution and processor choice?

**Speaker notes — Alan:**
"Our project measures how fast one computer runs one fixed object-detection
workload, and how that changes with the processor and the input size. Object
detection finds objects in an image and draws a labelled box around each one.
We used a pretrained network called YOLO11n and did not train anything. Our
job was to treat it as a workload, measure it the way we learned in Lecture 1
(execution time, throughput, and speedup), and answer three questions about
resolution, CPU versus GPU, and the trade-offs between them."

---

## Slide 2 — Experimental Setup

| Config. | Processor | Resolution | Role |
|---|---|---|---|
| A | M2 CPU | 640×640 | Baseline |
| B | M2 CPU | 320×320 | Resolution change |
| C | M2 GPU (MPS) | 640×640 | Processor change |
| D | M2 GPU (MPS) | 320×320 | Both changed |

- Apple M2 Mac, 8 GB unified memory, macOS
- YOLO11n pretrained, batch size 1; Python, PyTorch (MPS), Ultralytics
- 100 COCO 2017 validation images, fixed order

**Speaker notes — Kevin:**
"Everything ran on an Apple M2 Mac with 8 gigabytes of unified memory. The
CPU and the GPU share that memory. We used four configurations. A, the CPU at
640 by 640, is the baseline, because 640 is YOLO's standard input size. B
lowers the resolution on the CPU. C runs the same 640 workload on the M2 GPU
through Apple's Metal backend in PyTorch, called MPS. D is the GPU at 320.
Only the processor and the resolution change. The model file, the 100 images
and their order, and the settings are identical."

---

## Slide 3 — Benchmarking Methodology

- Images preloaded; model loaded; **1 untimed warm-up pass** of 100 images
- **Timer:** start before image 1, stop after image 100
  - includes preprocessing + network + postprocessing
- GPU: `torch.mps.synchronize()` after every image, no silent CPU fallback
- **5 measured trials** per configuration, all four in one session, raw CSV per trial
- Time/image = T / 100 · Throughput = 100 / T · Speedup = T_ref / T_comp

**Speaker notes — Alan:**
"To keep the comparison fair, everything outside the workload is outside the
timer: loading the model, reading the images from disk, and a full warm-up
pass. The timer measures elapsed time for all 100 images, including YOLO's
pre- and postprocessing, so it is the time to complete the workload, not just
the neural network. One detail matters for the GPU. It runs asynchronously,
so the program could return before the GPU has finished. We synchronize after
every image so the GPU timer is as honest as the CPU timer. The program also
checks that the GPU runs really used the GPU. Each configuration was measured
five times, and the formulas are the ones from Lecture 1."

---

## Slide 4 — Execution Time Results

[PENDING: insert Figure `results/graphs/execution_time_S20261112-121717-4092.png`]

| | CPU 640 | CPU 320 | GPU 640 | GPU 320 |
|---|---|---|---|---|
| Mean time (s, 100 images) | 2.991 | **0.968** | 1.747 | 1.140 |
| Time per image (ms) | 29.91 | **9.68** | 17.47 | 11.40 |

- Fastest: **CPU 320×320**, 0.968 s
- Trial-to-trial variation small (std ≤ 2.1 % of the mean) [PENDING: min/max]

**Speaker notes — Kevin:**
"This graph shows the average time to process all 100 images, so lower is
better. The baseline CPU at 640 took about 3 seconds, which is 30 milliseconds
per image. The fastest configuration was the CPU at 320, under one second, or
about 10 milliseconds per image. The GPU at 640 was in between at 1.75
seconds. The whiskers show the fastest and slowest of the five trials. They
are short, so the differences between the bars are much larger than the
run-to-run noise."

---

## Slide 5 — Throughput Results

[PENDING: insert Figure `results/graphs/throughput_S20261112-121717-4092.png`]

| | CPU 640 | CPU 320 | GPU 640 | GPU 320 |
|---|---|---|---|---|
| Throughput (images/s) | 33.4 | **103.3** | 57.3 | 87.8 |

- Highest throughput: **CPU 320×320**, about 103 images/s (3.1× the baseline)
- 640 → 320: CPU +209 %, GPU +53 %

**Speaker notes — Alan:**
"Throughput is the other side of execution time: images completed per
second, so higher is better. The CPU at 320 processed about 103 images per
second, roughly three times the baseline's 33. Lowering the resolution helped
both processors, but not equally. CPU throughput roughly tripled, while GPU
throughput rose by about half."

---

## Slide 6 — Speedup Comparison

[PENDING: insert Figure `results/graphs/cpu_gpu_speedup_S20261112-121717-4092.png`]

| Comparison | Speedup | Meaning |
|---|---|---|
| CPU 640 / GPU 640 | **1.71×** | GPU faster |
| CPU 320 / GPU 320 | **0.85×** | GPU **slower** (1.18× as long) |
| CPU 640 / CPU 320 | 3.09× | lower resolution, different workload |
| GPU 640 / GPU 320 | 1.53× | lower resolution, different workload |

**Speaker notes — Kevin:**
"Speedup is the reference time divided by the comparison time. Above one
means the comparison is faster. In the graph, the dashed line at 1.0 means
equal performance. At 640, the GPU was 1.71 times faster than the CPU. At 320
the ratio is 0.85, which is below one, so the GPU was slower. It took about
18 percent longer. The bottom two rows are not hardware speedups: they compare
a smaller workload with a larger one on the same processor. The CPU gained
3.09 times from the smaller input, and the GPU only 1.53 times."

---

## Slide 7 — Technical Analysis and Limitations

- **Measured:** GPU wins at 640×640, CPU wins at 320×320. Lower resolution gives the largest gains
- **Possible explanation (hypothesis, not measured):** fixed per-image costs
  (CPU-side pre/postprocessing, GPU launch + synchronization, batch 1) dominate
  when the network work shrinks
- **Detection quality:** not yet measured [PENDING: mAP50 at 640 vs. 320]. No claim that accuracy is maintained
- **Limitations:** one Mac, 100 images, batch 1, background OS activity,
  shared memory, possible thermal effects

**Speaker notes — Alan:**
"What we measured is clear: the better processor depends on the workload
size. Why that happens we can only suggest, because we measured total time,
not its parts. Our hypothesis is that each image has fixed costs, such as
Python and preprocessing on the CPU and starting and waiting for the GPU, that
do not get smaller with the input. At 320 the actual network work is about
four times smaller, so those fixed costs take up a bigger share and cancel
the GPU's advantage. We have not measured detection quality yet, so we do
not claim 320 is as accurate as 640. Finally, this is one computer and one
small workload, so other hardware could behave differently."

---

## Slide 8 — Conclusion and Demonstration

- **Q1:** 640 → 320 cut execution time by 68 % on the CPU and 35 % on the GPU
- **Q2:** GPU 1.71× faster at 640×640, but 0.85× (slower) at 320×320
- **Q3:** Best overall: CPU at 320×320 (3.09× the baseline). The processor choice depends on the workload size
- **Live demo:** `python -m src.benchmark --config C --demo 20`, then `--config A --demo 20`
  - 20 images, 2 trials, saved to `results/demo/`, separate from the formal results

**Speaker notes — Kevin:**
"To answer our questions. First, lowering the resolution made both
processors faster: 68 percent less time on the CPU and 35 percent on the GPU.
Second, the GPU was faster at 640 but slower at 320. Third, the fastest setup
overall was the CPU at 320, and the right processor depends on how much work
each image needs. Now we'll show how a measurement is made. We run the same
benchmark program on a quick 20-image subset, first on the GPU and then on
the CPU. The terminal shows the device check, the warm-up, each timed trial,
and how time per image and throughput are calculated. These quick-demo
numbers are a smaller workload than the formal experiment, so they are saved
separately and are not our reported results."

*Demo operator: whoever is at the Mac. Follow `demo_guide.md`, sections 2–5.
Check that the dataset validator passes before class.*
