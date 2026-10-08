# CPU/GPU serving comparison

Package: `20261008_132449_a32ab1`. Status: **failed**.

All measured cohorts contain 300 accepted B episodes. Ten disjoint warmup episodes, model startup, worker startup and worker teardown are excluded. B=1, k=1, 64 attested timan107 workers, frozen R16 spatial/100 task levels; common allocation budget 46,000 MiB.

| Model / arm | Copies / fronts | Episodes/min | GPU % | Host CPU % | Serving CPU (cores) | Peak VRAM MiB | Decision p50 / p99 ms | Rollout failures / errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

No GPU benchmark has been run. No performance numbers have been inferred as results.

## Deadline

Absolute deadline: `2026-10-08T15:15:00-05:00`; effective end: `2026-10-08T15:14:59.999999-05:00`. Active work stops 120 seconds earlier for owned-process cleanup.

## Paired best current versus best split

- groot: pending.
- pi05: pending.

The first pair is the initial screening measurement and has selection bias; the two additional pairs run in alternating orders (split→current, then current→split). Spread is descriptive, not a confidence interval. Host CPU is whole-machine utilization; serving cores sum the owned model/front/router/relay process CPU percentages divided by 100. VRAM peak includes startup and warmup; utilization means use only the measurement window. Decision latency is relay ingress through response send, excluding the worker network. Rollout failures are completed episodes with success=false; errors are accepted error records and are never treated as rollout failures.

## Numerical gates

- pi05_k1_b1: PASS (bit exact); `/home/weiland/projects/openpi_decouple/exp/decouple/results/gpu_comparison/pi05_k1_b1.json`.
- pi05_k2_b1: PASS (bit exact); `/home/weiland/projects/openpi_decouple/exp/decouple/results/gpu_comparison/pi05_k2_b1.json`.
- groot_k1_b1: PASS (bit exact); `/home/weiland/projects/openpi_decouple/exp/decouple/results/gpu_comparison/groot_k1_b1.json`.
- groot_k2_b1: PASS (bit exact); `/home/weiland/projects/openpi_decouple/exp/decouple/results/gpu_comparison/groot_k2_b1.json`.

## Last failure

`groot_initial_current_1: benchmark has no complete episodes/latencies/utilization samples`

## Artifacts

- Incomplete attempt: `/home/weiland/trace_runs/os_closed_loop/decouple_bench_20261008_132449_a32ab1_groot_initial_current_1_try01` — benchmark has no complete episodes/latencies/utilization samples
