# DP-LIBERO concurrent batching — 2026-10-10

Use **1 replica, max batch8, window5ms, 12 workers on timan108** for the new
official baseline. At the measured12-client load, DDPM100 delivered14.5 calls/s,
72.5 committed action rows/s, p50/p90 **708/1248ms**, and155.3W. Stage3 reached
B8 and averaged6.69. This is the measured configuration that fills its selected
cap within the current node RAM estimate.

Two replicas/B4 is the throughput alternative:18 calls/s,575/779ms,214.5W;
Stage3 averaged/reachedB3 under12 clients. Larger configured caps offered little
benefit at this load. Keep replica count, maximum, window, stage workers,
language memo and client concurrency identical between the new control and arms.

Compared with the owner's old1409ms median, the selected configuration has
about **2.0x lower median latency** in this local frame replay. If12 continuously
busy old replicas and median≈mean imply12/1.409=8.52 calls/s, the provisional
capacity gain is **1.70x**. The old aggregate throughput was not measured here;
its cost_ms and the new queued WebSocket latency have different timing scopes.

## Measurement table

All rows use the sealed final `/data/mw_finetune/dp/libero_policy_100000`, real
paired256px camera observations, CUDA graph denoising,12 concurrent closed-loop
clients,50ms think time, and commit5. K100 is unchanged DDPM; K10/K5 are trailing
DDIM with eta0. Actions/s counts the5 committed rows returned per call; native
predicted positions/s is16×calls/s. Latency includes local transport and queue
wait. Max diff is the maximum absolute difference in these committed physical
actions against each frame/seed's GPU B1 eager reference.

Throughput and board power use a fixed window after concurrent warmup. Latency
includes all calls started in that window, including their drain. The first30
points have4s windows; K5/two-replica points have1s windows (37–50 latency
samples each), so their rate/power estimates are coarse. Resident model/graph
workspaces were retained between rows. Dataset replay measures serving cost;
success-rate equivalence remains untested.

The B1 rows explicitly use the same S07 runtime and frozen text memo as the
batched rows, with its cap set to1. The default-off formal B1 entry still uses
its original path. These measurements include both the opt-in runtime and
batching changes; they do not isolate batching's contribution to the old cost.

| K / replicas × max B | calls/s | actions/s | p50 / p90 ms | W | max abs diff |
|---|---:|---:|---:|---:|---:|
| 100 / 1×1 | 6.50 | 32.50 | 1651 / 2950 | 167.7 | 0 |
| 100 / 1×4 | 13.50 | 67.50 | 758 / 1298 | 162.9 | 1.59e-04 |
| 100 / 1×8 | 14.50 | 72.50 | 708 / 1248 | 155.3 | 3.15e-04 |
| 100 / 1×16 | 15.00 | 75.00 | 706 / 1008 | 152.0 | 1.75e-04 |
| 100 / 1×24 | 15.00 | 75.00 | 709 / 1276 | 153.7 | 3.46e-04 |
| 100 / 1×32 | 15.00 | 75.00 | 716 / 1091 | 153.1 | 3.46e-04 |
| 100 / 2×1 | 7.50 | 37.50 | 1443 / 2053 | 187.5 | 0 |
| 100 / 2×4 | 18.00 | 90.00 | 575 / 779 | 214.5 | 1.59e-04 |
| 100 / 2×8 | 16.00 | 80.00 | 615 / 812 | 206.6 | 3.46e-04 |
| 100 / 2×16 | 18.00 | 90.00 | 584 / 810 | 208.9 | 3.15e-04 |
| 100 / 2×24 | 17.25 | 86.25 | 555 / 804 | 207.7 | 3.46e-04 |
| 100 / 2×32 | 18.00 | 90.00 | 564 / 748 | 209.1 | 3.15e-04 |
| 10 / 1×1 | 18.50 | 92.50 | 577 / 939 | 87.8 | 0 |
| 10 / 1×4 | 25.75 | 128.75 | 387 / 499 | 81.4 | 1.23e-04 |
| 10 / 1×8 | 26.75 | 133.75 | 391 / 507 | 80.4 | 2.73e-04 |
| 10 / 1×16 | 27.00 | 135.00 | 373 / 517 | 80.2 | 2.73e-04 |
| 10 / 1×24 | 25.75 | 128.75 | 384 / 529 | 79.7 | 2.73e-04 |
| 10 / 1×32 | 25.50 | 127.50 | 392 / 521 | 78.8 | 2.61e-04 |
| 10 / 2×1 | 37.00 | 185.00 | 267 / 285 | 137.1 | 0 |
| 10 / 2×4 | 48.50 | 242.50 | 192 / 213 | 125.6 | 1.23e-04 |
| 10 / 2×8 | 45.00 | 225.00 | 209 / 256 | 124.5 | 2.61e-04 |
| 10 / 2×16 | 44.75 | 223.75 | 202 / 274 | 122.8 | 2.73e-04 |
| 10 / 2×24 | 45.00 | 225.00 | 205 / 258 | 128.0 | 2.61e-04 |
| 10 / 2×32 | 45.50 | 227.50 | 203 / 262 | 129.8 | 2.61e-04 |
| 5 / 1×1 | 19.00 | 95.00 | 548 / 960 | 78.5 | 0 |
| 5 / 1×4 | 29.50 | 147.50 | 340 / 450 | 75.2 | 1.18e-04 |
| 5 / 1×8 | 27.75 | 138.75 | 368 / 498 | 74.4 | 2.56e-04 |
| 5 / 1×16 | 27.50 | 137.50 | 376 / 481 | 73.6 | 2.56e-04 |
| 5 / 1×24 | 28.50 | 142.50 | 364 / 467 | 73.8 | 2.28e-04 |
| 5 / 1×32 | 28.25 | 141.25 | 369 / 477 | 73.7 | 2.56e-04 |
| 5 / 2×1 | 37.00 | 185.00 | 264 / 607 | 94.7 | 0 |
| 5 / 2×4 | 43.00 | 215.00 | 215 / 604 | 78.7 | 1.18e-04 |
| 5 / 2×8 | 38.00 | 190.00 | 235 / 561 | 73.3 | 2.56e-04 |
| 5 / 2×16 | 40.00 | 200.00 | 250 / 612 | 75.0 | 2.50e-04 |
| 5 / 2×24 | 37.00 | 185.00 | 232 / 436 | 95.7 | 2.50e-04 |
| 5 / 2×32 | — | — | — | — | unmeasured |

**35/36 points measured.** K5/r2/B32 is unmeasured: the final CPU-preloaded
attempt hit its4.8s GPU activation cutoff before readiness. It has no throughput
receipt. Aggregate GPU allowance used, including failed/restarted attempts and
owned cleanup: **1199.589s /1200s**. No further GPU measurement was performed.

Raw results: [CSV](samples/dynamic_batch_results.csv),
[JSON](samples/dynamic_batch_results.json),
[summary/budget/source receipts](samples/dynamic_batch_summary.json).
Each measured row links its own result, calls, board-power samples, native logs,
stage-membership ledger, ready/closed metadata and owned PID cleanup.

The earlier64-client capacity runs remain separate diagnostic evidence under
`samples/dynamic_batch_gpu_final_20261010/` and
`samples/dynamic_batch_gpu_persistent_20261010/`. One-replica K100 reached16.67
calls/s at cap24 but averagedB19.61; cap32 averagedB29.19 and reachedB31.
These high-concurrency queues exceeded the RAM-feasible client load and do not
size the launch below. GPU native B32 qualification covered every arm directly.

## RNG, graph and framework path

The owned runtime uses the existing stock S07 `BatchingCore`, P1 connection
factory and shared WebSocket lifecycle. Concurrent connections submit to the
same resident stage queues. There is one worker per stage and no resident
inference lock on this explicit path. Stage1 and denoising use native vector
forwards; cheap Stage2 retains its per-row conditioning. Mixed schedules,
snapshot positions and capture specifications remain separate buckets.

Each request owns its generator. Initial and DDPM posterior draws retain the
B1 shape/stream; the vector scheduler receives a list of private generators.
Seeds depend on semantic request identity/explicit seed, independently of
peer identity, row, ordering or queue timing. GPU qualification checked
**352 B32 request RNG receipts exactly against B1** across the three samplers.
The native B32 test uses B1 conditioning and compares all16 physical action rows;
the serving table also includes vision batching. Its largest committed action
deviation was0.000345707. Batch arithmetic can still vary with membership.
Matched configurations need a fresh quality control; cache arms also need the
membership ledger to investigate differences in keys, donors and actions.

The opt-in CPU text memo evaluates the frozen language function at B1 per exact
prompt and retains up to128 immutable embeddings. CPU tests prove identical
values and prevent output aliasing. The default path has no memo or queue.

E's `DenoiserGraph` is consumed without editing `exp/mw_finetune/dp/`. Every
allowed B1..max graph signature is captured before S07 threads start. Serving
permits replay only. This fixed the real GPU failure where a lazy global graph
capture overlapped another stage's readback; the failed receipt is preserved.
The measured release's native sources match the CPU seal, model identity
`sha256:ed3ae8c67f9686d81227ccd5de627391a7b3ef37f7f302ae8a3f8214b7a16229`.

CPU: **185 passed,0 skipped**, plus **49 legacy/default-off passed,0 skipped**.
New tests cover private scheduler streams through B32, per-request order,
actual cross-connection S07 buckets for all three arms, commits5/10/15/16,
partial-tail declarations, graph prewarm ordering, runtime restoration, launch
delegation and Cosmos RAM reservation. The default B1 numerical/reference,
resume, capture and production guards continue to pass.

## Exact formal launch command

Prepared and argument-validated; **not launched**. P3 published the normal
`ops/formal_eval.sh` seam for DP max batch1..32: it seals `dp_batching`, sets the
measured stage configuration and dispatches to the owned DP resident while
preserving CUDA graphs. CPU regressions cover parsing, dispatch and environment.
The command below uses that normal framework path. The DP-owned launcher shim
remains a local fallback; default B1 delegates unchanged. P4 edited no common
serving files.

```bash
set -euo pipefail
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src:packages/openpi-client/src \
  /data/mw_finetune/dp/env/bin/python \
  exp/r17_integration/models/dp/samples/node_ram_preflight.py \
  --workers 12 --worker-memory-mib 4096 \
  --receipt exp/r17_integration/models/dp/samples/dp_launch_ram_admission.json

flock -w 600 /tmp/r17_gpu.lock \
  env CUDA_VISIBLE_DEVICES=GPU-98d36ed2-a8e9-fb07-701c-41f337a6c1f9 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src:packages/openpi-client/src \
  exp/r17_integration/ops/formal_eval.sh \
  --max-batch 8 --batch-wait-ms 5 \
  --model dp_libero --checkpoint /data/mw_finetune/dp/libero_policy_100000 \
  --server-node local --worker-node timan108 --extra-workers 0 \
  --replicas 1 --workers 12 --worker-memory-mib 4096 \
  --gpu cuda:0 --memory-budget-gib 40 --port 23160 --driver-port 23179 \
  --official-50 --suites libero_spatial libero_object libero_goal libero_10 \
  --seconds 43200 --episode-seconds 1800 --retries 2 \
  --run-name dp_libero_k100_c5_r1_b8_w5_official50_20261010 --run
```

This uses the native K100/C5 defaults. The shared CLI rejects combining explicit
step and horizon overrides, even when requesting their native defaults. For a
plain K10 or K5 arm add only `--denoising-steps 10` or5 and a distinct run name;
retain the same batching/workers/benchmark. For horizon-only studies use
`--commit-horizon 10`,15 or16 with the default K100. Model hooks support all
three samplers with all four horizons; combined launch overrides need P6's
separate CLI admission change.

Reserved ports23160/23161/23179 were free at handback. A newly started, separately owned
DP control occupied23180/23181/23189 during the final read-only check; it was
not touched. The command uses a fresh run name and the cooperative GPU lock.

At11:33 CDT timan108 had251.63GiB total and138.85GiB available. Cosmos24 live
workers used90.98GiB RSS; reserving6GiB each adds53.02GiB future use. The normal
25.16GiB headroom plus DP12×4=48GiB leaves about12.67GiB spare. DP16 workers
would exceed this observed budget. The read-only preflight reuses P6's host/
cgroup capacity function, validates PID start ticks, and adds the owner's6GiB
Cosmos floor even for older receipts missing their reservation field. Normal
P6 launch admission remains enabled.

**Latest RAM admission is blocked:** timan108's Tether agent was offline
(exit64; `samples/dynamic_batch_ram_preflight_launch_handoff.json`). The command stops
before GPU launch until a fresh live admission succeeds. No agent, worker or
existing campaign was restarted. timan1, H100 and the other campaigns are unused.

## Handoff limits

- P4/Q: the single K5/r2/B32 concurrent timing cell needs a later GPU allocation.
  Native B32 RNG/math passed for K5; that missing timing cell remains empty.
- Coordinator: fresh matched-batch official50 control/arms establish SR.
  Neither the previous B1 baseline nor these three frames establishes quality.
- P3's shared DP batching CLI/dispatch seam is published and CPU-tested.
  P6: corrected6GiB legacy Cosmos RAM reservations and combined step/horizon CLI
  admission remain requested in `REQUESTS/P4_to_P6.md`. DP emits a per-decision
  membership ledger; shared paired-control enforcement is currently Fast-WAM
  only. Coordinator/P4/P6 must add that gate before automated paired DP studies.
- ModelSpec describes the default-off resident and retains its sealed provider
  identity. Active batching is advertised separately in `r17_batch_policy`,
  `r17_batching` and `batch_runtime_capabilities`; max/window/numerics policy and
  memberships are recorded per decision. Default B1, MetaWorld and K1 admission
  remain covered by the prior restrictions. This formal entry admits pure
  inference LIBERO; cache/recipe GPU qualification is a separate gate.

For a later <=15min mechanical smoke, use the same command with a fresh run
name, replace official50/suites with `--smoke --smoke-episodes 4`, and use
`--seconds 780 --episode-seconds 600 --retries 0`. This command was not executed.
