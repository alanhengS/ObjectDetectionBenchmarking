# Performance graphs — not yet copied

The three Step 6 graphs are generated on the measurement Mac and are not in
this folder yet. They are not stored in git, because `results/` is excluded.

Copy them here on the Mac (from `ECE4300_Object_Detection`):

```bash
cp results/graphs/execution_time_S20261112-121717-4092.png  deliverables/performance_graphs/
cp results/graphs/throughput_S20261112-121717-4092.png      deliverables/performance_graphs/
cp results/graphs/cpu_gpu_speedup_S20261112-121717-4092.png deliverables/performance_graphs/
```

Run `python -m src.validate_experiment --session S20261112-121717-4092`
first. It confirms that each PNG is byte-identical to a fresh render of the raw
measurements.
