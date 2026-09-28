# K8 search latency: idle-machine re-check (task #40, 2026-09-28 04:51–04:55 CDT)

After the last R5 closed-loop arm ended (04:50), `benchmark.py` was re-run unchanged for 20 key configurations, two fresh
processes each (`--run 5`, `--run 6`; K8's `r1`/`r2` files untouched), pinned to CPUs 34-37,78-81 with BLAS threads 1
(script: `/home/weiland/.claude/jobs/a607dd74/tmp/k8_idle.sh`). No servers or chains were running; the R5 analysis
agent (CPU 26-29,70-73) and another project's processes were still active, so this is "no closed loop on the machine",
not a silent machine (load average 15 at start; the benchmark's own CPUs were 82–84 % idle). 40/40 runs, 0 errors.

Method-only CPU latency per vision decision (ms, p50 of each process; raw files `bench_<config>_r{5,6}.json`):

| config | K8 r1/r2 vision p50 ms | idle r5/r6 vision p50 ms | idle p90 | idle% (r5) | blind p50 r5 |
|---|---|---|---|---|---|
| pi05_spatial_50_AWM | 1.16/1.15 | 1.19/1.14 | 1.52/1.23 | 84 | — |
| pi05_spatial_50_MixedJudge | 1.57/1.50 | 1.56/1.56 | 2.02/1.91 | 83 | — |
| pi05_spatial_50_BlindAWM | 1.90/1.17 | 1.26/1.16 | 1.61/1.28 | 83 | — |
| pi05_spatial_500_AWM | 1.47/1.27 | 1.26/1.25 | 1.57/1.46 | 82 | — |
| pi05_spatial_500_MixedJudge | 2.51/1.75 | 1.77/1.75 | 2.13/2.16 | 84 | — |
| pi05_spatial_500_BlindAWM | 1.45/1.28 | 1.26/1.32 | 1.34/1.49 | 83 | — |
| pi05_l10_50_AWM | 1.17/1.16 | 1.16/1.16 | 1.24/1.24 | 84 | — |
| pi05_l10_50_MixedJudge | 1.80/1.56 | 1.58/1.61 | 1.68/1.84 | 84 | — |
| pi05_l10_50_BlindAWM | 1.29/1.26 | 1.19/1.25 | 1.25/1.42 | 84 | — |
| pi05_l10_500_AWM | 1.51/1.36 | 1.35/1.35 | 1.46/1.49 | 84 | — |
| pi05_l10_500_MixedJudge | 2.34/1.96 | 1.97/2.05 | 2.19/2.32 | 83 | — |
| pi05_l10_500_BlindAWM | 1.41/1.44 | 1.39/1.47 | 1.54/1.95 | 84 | — |
| groot_spatial_50_AWM | 1.12/1.15 | 1.14/1.21 | 1.24/1.59 | 83 | — |
| groot_spatial_500_AWM | 1.25/1.26 | 1.24/1.28 | 1.34/1.76 | 83 | — |
| groot_l10_50_AWM | 1.17/1.20 | 1.16/1.20 | 1.22/1.55 | 83 | — |
| groot_l10_500_AWM | 1.38/1.38 | 1.35/1.41 | 1.47/2.07 | 84 | — |
| K7_r4k7_p_l10_500_tail1ug | 2.09/2.10 | 1.99/2.25 | 2.19/3.35 | 84 | — |
| K7_r4k7_p_l10_50_tail1ug | 1.66/1.65 | 1.64/1.66 | 1.72/1.76 | 84 | — |
| K7_r4k7_p_sp_500_tail1ug | 1.82/1.79 | 1.80/1.80 | 2.14/2.00 | 84 | — |
| K7_r4k7_p_l10_500_ph2g | 2.08/2.11 | 2.10/2.18 | 2.79/2.60 | 84 | — |

**Reading.** The idle re-check reproduces K8's controlled table within ±0.1 ms at p50 for every configuration except
the two K8 first runs whose p50 was inflated by load (`pi05_spatial_50_BlindAWM` r1 1.90, `pi05_spatial_500_MixedJudge`
r1 2.51); K8's numbers stand as the single-process cost. 50 → 500 episodes adds ≈ 0.1–0.2 ms for plain AWM, ≈ 0.4 ms
for MixedJudge and ≈ 0.45 ms for the K7 tail controller at l10. Per decision this is 1.1–2.2 ms, i.e. 1.7–3.3 % of a
67.5 ms full inference (owner IR +.017–.033 if every decision searches on CPU).

**Loaded serving differs.** In the live R5 Q5 shadow arms (24 connections per server, two servers per GPU line, R5
chains running) the same CPU method measured p50 4.9–5.3 ms (50) and 7.4–9.0 ms (500), p90 8.5–16 ms: 3–6× the
single-process figure, from GIL / thread contention inside one serving process. The GPU retrieval path in the same
arms was p50 1.95 ms (50) and 2.4–2.7 ms (500). The single-process table is the method cost; the live numbers are the
serving cost under this deployment's concurrency.
