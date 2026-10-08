# CPU/GPU serving comparison

Package: `20261008_172020_ae91da`. Status: **failed**.

All measured cohorts contain 20 accepted B episodes. Ten disjoint warmup episodes, model startup, worker startup and worker teardown are excluded. B=1, k=1, 64 attested timan107 workers, frozen R16 spatial/100 task levels; common allocation budget 46,000 MiB.

| Model / arm | Copies / fronts | Episodes/min | GPU % | Host CPU % | Serving CPU (cores) | Peak VRAM MiB | Decision p50 / p99 ms | Rollout failures / errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| groot/initial/current_1 | 1 / — | 23.12 | 5.6 | 5.9 | 0.86 | 6876 | 341.16 / 1265.01 | 2 / 0 |

## Deadline

Absolute deadline: `2026-10-08T17:32:16-05:00`; effective end: `2026-10-08T17:30:20.521832-05:00`. Active work stops 120 seconds earlier for owned-process cleanup.

## Paired best current versus best split

- groot: pending.
- pi05: pending.

Smoke checks collection and launch wiring on 20 episodes per arm; it is not the full throughput experiment.

## Numerical gates

- pi05_k1_b1: PASS (bit exact); `/home/weiland/projects/openpi_decouple/exp/decouple/results/gate_cache/proofs/pi05_k1_b1_2bba67bc5c7d5db8be3db4b184de23eaac5470bbbbb75fb29e067cb0572f8e40_9e8d211b03ac.json`.
- pi05_k2_b1: PASS (bit exact); `/home/weiland/projects/openpi_decouple/exp/decouple/results/gate_cache/proofs/pi05_k2_b1_7e6805e6baec7a1f9ead18605e8f74625e1e73e3691e562f0558ffc86aac17b4_9e8d211b03ac.json`.
- groot_k1_b1: PASS (bit exact); `/home/weiland/projects/openpi_decouple/exp/decouple/results/gate_cache/proofs/groot_k1_b1_ad136b6ff437bc7d4411f56f3f21e37aba22f86013eb527f865bdb9904cac012_9e8d211b03ac.json`.
- groot_k2_b1: PASS (bit exact); `/home/weiland/projects/openpi_decouple/exp/decouple/results/gate_cache/proofs/groot_k2_b1_e3aee5b12e548456bd48d93ba410bc07d78891074b1b7241b36200d08674f1d7_9e8d211b03ac.json`.

## Last failure

`groot_initial_split_8: server exited during startup (1)`

## Artifacts

- groot/initial/current_1: `/home/weiland/trace_runs/os_closed_loop/decouple_bench_20261008_172020_ae91da_groot_initial_current_1_try01` (metrics, samples, journal, latencies, server/worker logs, worker census and sealed group).
- Incomplete attempt: `/home/weiland/trace_runs/os_closed_loop/decouple_bench_20261008_172020_ae91da_groot_initial_split_8_try01` — server exited during startup (1)
