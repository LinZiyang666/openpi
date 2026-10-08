# CPU/GPU serving comparison

Package: `reanalysis_decouple_bench_20261008_132449_a32ab1_groot_initial_current_1_try01`. Status: **reanalysed partial**.

All measured cohorts contain 300 accepted B episodes. Ten disjoint warmup episodes, model startup, worker startup and worker teardown are excluded. B=1, k=1, 64 attested timan107 workers, frozen R16 spatial/100 task levels; common allocation budget 46,000 MiB.

| Model / arm | Copies / fronts | Episodes/min | GPU % | Host CPU % | Serving CPU (cores) | Peak VRAM MiB | Decision p50 / p99 ms | Rollout failures / errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| groot/initial/current_1 | 1 / — | 31.02 | unavailable | unavailable | unavailable | unavailable | 1829.33 / 7838.90 | 23 / 0 |

groot/initial/current_1 latency scope: **resident policy after_infer timing; legacy recovery; excludes relay transport and queue**.
- Legacy relay has no UID/attempt/timestamps; measured-cohort relay latency is unrecoverable. Policy timings are reported separately.
- Original sampler was memory-only and lost on collection failure; GPU/CPU means and full-run VRAM peak cannot be reconstructed.
- Startup VRAM lower bound: 6618.19 MiB; full-run peak unavailable.
- Unattributed relay p50/p99: 3893.24/9667.04 ms (all relay rows; includes warmup and rejected attempts; not cohort-attributed).

## Paired best current versus best split

- groot: pending.
- pi05: pending.

The first pair is the initial screening measurement and has selection bias; the two additional pairs run in alternating orders (split→current, then current→split). Spread is descriptive, not a confidence interval. Host CPU is whole-machine utilization; serving cores sum the owned model/front/router/relay process CPU percentages divided by 100. VRAM peak includes startup and warmup; utilization means use only the measurement window. Decision latency is relay ingress through response send, excluding the worker network. Rollout failures are completed episodes with success=false; errors are accepted error records and are never treated as rollout failures.

## Numerical gates


## Artifacts

- groot/initial/current_1: `/home/weiland/trace_runs/os_closed_loop/decouple_bench_20261008_132449_a32ab1_groot_initial_current_1_try01` (metrics, samples, journal, latencies, server/worker logs, worker census and sealed group).
