# Fast-WAM LIBERO batching — qualified normal launcher, 2026-10-10

**Recommend `--max-batch 8 --batch-wait-ms 5`, default off.** One local resident
with eight concurrent connection adapters delivered **5.98 decisions/s**, versus
**1.10 decisions/s** on the original direct B1 path: **5.44×**. Against S07's
queued B1 (1.00 decisions/s), the gain is **5.97×**. Mean request latency is
1.33 s, p95 1.45 s. These are local 4090 measurements; they are not H100 results
or an episode-success/episodes-per-minute certificate.

The slow part is the **K10 action head: 84.6% of a B1 decision**. At B1 it runs
ten forwards with only 32 action tokens per request. B1 and B2 cost almost the
same wall time; increasing the useful work per forward gives almost twice the
throughput. This supports the owner's small-batch diagnosis. GPU utilization
reports time with GPU activity, not SM occupancy or proximity to the power cap.
Our local B1 used about 94 W of a 450 W cap. The H100 smoke below includes
board telemetry while Cosmos shared the GPU; that power cannot be attributed
to Fast-WAM alone.
CUDA event spans below include host launch gaps, so they must not be interpreted
as pure kernel compute time or used to distinguish memory stalls from launch
overhead. Nsight occupancy was not collected: collecting/replaying thousands of
native kernels was unnecessary for this bounded batching experiment.

## Measurement and provenance

Actual resident: NVIDIA GeForce RTX 4090, 49,140 MiB, UUID
`GPU-98d36ed2-a8e9-fb07-701c-41f337a6c1f9`, Torch 2.7.1+cu128. Raw-Wan P3 loader,
published `/data/mw_finetune/models/fastwam_libero`, BF16, native shifted-flow
K10, system protocol (256-pixel wire views, per-camera 224 resize, fused
224×448 input, H32/replan5). Weights/Wan/tokenizer/norm/source digests and the
native grid are in [local_descriptor.json](measurements/local_descriptor.json)
and [local_sweep.json](measurements/local_sweep.json).

Inputs are three **recorded**, original 256×256 paired LIBERO frames from
dataset episode 1272, frames 0/1/2, one task/prompt. The provenance manifest is
`/data/mw_finetune/qualification/observations/libero_original256_fix2_v1/OBSERVATIONS.json`.
Observation NPZ SHA256:
`403187908f0bbbbbc6196b2e7f6d1224acc382ce99dee34d4199dba33cadb3d2`.
The replay request's episode number identifies this dataset episode, not an
official benchmark init index. Seeds 42/43/44 belong to the corresponding
frames and remain fixed across B sizes, ordering and repetitions. B>3 repeats
these real frames; this is not a diverse 16-task quality evaluation.

Each B has one excluded warmup and three measured repeats. Capture/snapshots/CP2
are off. Composition logging is on. Direct timing includes prepare, all native
stages and the wire transform. S07 timing includes connection preparation,
queueing, the same native stages and wire output. S07 uses one worker per stage,
CUDA stage streams and a 5 ms window. The GPU concurrent sweep uses real S07
queues with distinct connection adapters in one process; it excludes WebSocket,
network, simulator and episode-dispatch time. CPU regressions exercise the
actual shared connection policies and formal-entry composition.

Board power/utilization are sampled via NVML every 100 ms, with 27–77 samples per
measured B row. They are short-window observations, not long-run power averages.
VRAM columns are PyTorch peak allocated/reserved GiB, including resident weights;
CUDA context/driver allocations are excluded. Reserved memory retains earlier
allocator/stream pools in this sequential sweep and is a conservative high
water mark, not an independently cold-loaded B measurement.

## B1 decision profile

One warmed decision, synchronized phase boundaries; total **970.27 ms** versus
908.68 ms mean for the sweep without phase instrumentation. Phase barriers add
overhead. CUDA event elapsed measures the phase's stream interval.

| Work | Host wall ms | CUDA event ms | Wall share |
|---|---:|---:|---:|
| VAE encode | 11.05 | 11.00 | 1.14% |
| Video prefill | 83.51 | 83.49 | 8.61% |
| Action-head K10 | 820.85 | 820.62 | 84.60% |
| Other, including text/conditioning/prepare/keys/wire/logging | 54.86 | — | 5.65% |
| Total | 970.27 | — | 100% |

Other includes UMT5/context **40.82 ms**, native pre-DiT conditioning 1.80 ms,
patchify/key pooling 0.51 ms, and residual 11.73 ms. Text runs only on the
MISS/full path. The profile is 92.85 W mean / 93.89 W maximum, with nine NVML
samples; SM occupancy is unmeasured. The raw-Wan loader's `BatchedVAE` actually
stacks the inputs into one encode, rather than using the disabled legacy
per-video loop; no VAE rewrite was needed.

## Native full-batch sweep

This gives the throughput when all requests reach each stage together. Latency
is the whole batch's wall time, hence the same service latency for its members.

| B | Decisions/s | Gain over direct B1 | Mean latency s | Peak allocated/reserved GiB | Mean/max W |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.10 | 1.00× | 0.909 | 23.36 / 23.66 | 94.1 / 98.2 |
| 2 | 2.18 | 1.98× | 0.917 | 23.41 / 23.67 | 101.4 / 103.2 |
| 4 | 4.09 | 3.72× | 0.978 | 23.64 / 23.96 | 158.2 / 172.8 |
| 8 | 7.30 | 6.63× | 1.096 | 24.13 / 24.53 | 197.9 / 199.0 |
| 16 | 11.07 | 10.06× | 1.446 | 25.11 / 26.05 | 257.8 / 267.6 |

## S07 concurrent connection sweep

One resident, B simultaneous connections, three synchronized request rounds.
Latency includes queueing and preparation. B1 here deliberately uses S07; the
default production B1 still uses the original direct path.

| Max B | Decisions/s | Gain over direct B1 | Mean / p95 latency s | Peak allocated/reserved GiB | Mean/max W | Actual mean S1/S2/S3 B |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.91× | 0.998 / 1.001 | 23.43 / 26.86 | 104.3 / 123.4 | 1 / 1 / 1 |
| 2 | 1.94 | 1.76× | 1.033 / 1.044 | 23.53 / 27.92 | 101.5 / 105.2 | 2 / 2 / 2 |
| 4 | 3.59 | 3.26× | 1.113 / 1.149 | 23.79 / 29.65 | 130.1 / 148.6 | 4 / 4 / 4 |
| **8** | **5.98** | **5.44×** | **1.334 / 1.448** | **24.31 / 32.44** | **183.0 / 191.0** | **8 / 8 / 8** |
| 16 | 6.26 | 5.69× | 2.292 / 2.669 | 25.30 / 38.28 | 185.3 / 232.7 | 13.33 / 13.33 / 9.875 |

B16 gives only **4.6%** more server throughput than B8, with **72%** higher
mean latency and larger reserved memory. Its 5 ms window fragments the action
stage into smaller batches. Longer windows or a deeper continuous request pool
may improve B16, but were not measured. B8 is the present recommendation, with
at least eight active connections per replica and one worker per stage. Actual
batch size must be monitored; asynchronous simulator traffic may fill less
well than this burst replay. The existing two H100 residents should be assessed
with their real arrival pattern before changing replica count.

## Normalized action numerics

Same frame, seed, checkpoint and K10 versus that frame's direct B1 output. All
elements of each 32×7 chunk count in these summaries; 'typical' is median
absolute deviation and p95 is the elementwise 95th percentile.

| B | Direct max abs | S07 max abs | S07 median abs | S07 p95 abs | S07 mean abs |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0.015625 | 0.015625 | 0.00048828 | 0.00390625 | 0.00142116 |
| 4 | 0.0078125 | 0.0078125 | 0.00097656 | 0.00390625 | 0.00153351 |
| 8 | 0.0078125 | 0.0078125 | 0.00097656 | 0.00390625 | 0.00148246 |
| 16 | 0.0078125 | 0.015625 | 0.00097656 | 0.00390625 | 0.00148055 |

No byte parity is claimed for B>1. There is no per-batch normalization and no
shared RNG draw: images, proprio, context masks, video KV, noise, actions and
snapshots keep their row identity. Vector behavior and request-order/RNG/output
isolation are tested on CPU; the GPU sweep establishes the arithmetic deviations
on these three recorded frames only. Heterogeneous prompts, physical gripper
threshold effects, policy success rate and all reduced-K tolerances are not
qualified by this receipt. Runtime metadata labels this
`native_bf16_batch_study_v1`, rather than silently advertising exact parity.

## Enabled entry points and composition

`serve --max-batch N --batch-wait-ms 5 --batch-ledger PATH` creates P3's runtime
only for explicit B>1 LIBERO serving. `BatchingCore` is the unchanged stock S07
implementation. P3's native StageBatcher groups compatible shapes/model IDs,
preserves sampler K/native grid/noise buckets and emits private outputs. Native
VAE cache state has a shared lock; each stage has one worker. Per-connection
state, bundles, retrieval, tails and samplers remain in the P1 factory.

The normal `ops/formal_eval.sh` / `formal_eval.py` launcher now accepts
`--max-batch` (default1), `--batch-wait-ms` (default5) on **local and H100**.
It dispatches explicit Fast-WAM batches to the owned `formal_batch_server`;
sealing, worker deployment, conductor, process ownership, hot-switch and
aggregation retain the P6/P1 path. The factory preserves the sealed provider/
YAML identity and concurrent inference omits the B1 resident lock. Default B1
opens no batch ledger, starts no batch threads and preserves actions/metadata.
`formal_launch` now delegates to this normal entry; the old local shim is retired.

Campaign, CPU seal, summary, resident attestation, decision ledger and final
receipt bind batching settings. Treatment v2 pins the implementation source,
GPU family, replica/worker topology, renderer GPU allowlist, stage streams and
workers, worker recycling and RAM/VRAM reservations. Each non-control batched
formal arm requires `--paired-control <completed run directory>`. The control
must be a closed PASS official-50 K10/C5 baseline, with the same treatment,
sealed model/norm/protocol and all requested suites; its campaign, receipt,
summary and attestations are hash-pinned. B1/B8, different windows/GPU types/
concurrency, failed controls and mismatched resume are rejected on CPU.
Actual member ordering can differ with arrivals; every decision carries it.

Each completed stage writes `r17.fastwam.stage_batch.v1` with batch ID, stage,
ordered request/connection members, semantic request context, row and config
identity. Each decision writes `r17.fastwam.decision_batch.v1`, and the response
contains `__hit_meta__.batching`; formal `serving.jsonl` repeats that receipt.
Cache hits can record stage1 alone; tails record no stages and no model calls.
The measured [composition ledger](measurements/local_sweep.batches.jsonl) and
per-decision memberships in the sweep receipt make partial B16 batches visible.

DP has native hooks but its production B>1 gate and graph workspaces require
P4's separate qualification. P4 is implementing its own DP runtime and has
requested the same P6 launcher seam. Cosmos has no StageBatcher and mutable B1
state. Neither model was changed or GPU-tested by P3.

## Exact launcher command

**Prepared for a new local smoke campaign; not executed.** It requires free
reserved ports and the normal P6 worker/preflight admission. It does not alter
an existing campaign. After its smoke/quality check, use the same batching
settings for a fresh control and every arm. Do not transplant this local shim
onto H100; wait for the P6 seam and an H100 qualification slot.

```bash
flock -w 600 /tmp/r17_gpu.lock env CUDA_VISIBLE_DEVICES=GPU-98d36ed2-a8e9-fb07-701c-41f337a6c1f9 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src:packages/openpi-client/src \
  .venv/bin/python -m exp.r17_integration.models.fastwam.formal_launch \
  --max-batch 8 --batch-wait-ms 5 \
  --model fastwam_libero --checkpoint /data/mw_finetune/models/fastwam_libero \
  --server-node local --worker-node timan107 --replicas 1 --workers 16 \
  --gpu cuda:0 --memory-budget-gib 40 --port 23160 --driver-port 23179 \
  --smoke --smoke-episodes 4 --seconds 1200 --episode-seconds 900 --retries 0 \
  --run-name fastwam_libero_b8_w5_smoke --run
```

The coordinator stopped the H100 B1 C10 arm and the B1 baseline had failed.
Those records describe B1 and cannot be the paired control
for B8 arms because batching changes arithmetic and latency. A fresh batched
control must match max batch, window, stage workers, stage-stream setting,
replica count and dispatch/concurrency settings with its arms, and retain every
composition receipt. Exact row composition may still vary with arrival times;
the settings match is necessary but does not restore byte-identical paired
actions. Report/replay membership if arithmetic differences affect a result.

## Reproduce the bounded GPU sweep

The successful run took **150.35 s including 86.44 s model load**, and exited
normally with zero GPU memory left. A preceding attempt failed the content seal
before CUDA allocation because a source file changed during verification; its
log is retained. The successful run used `flock -w 600`, UUID-pinned CUDA and a
1200 s internal deadline plus 1280 s outer bound. No H100 connection, existing
process signal, commit/push, training or frozen-source edit was performed.

The CPU `--seal-only` first creates the descriptor. The replay can then run:

```bash
flock -w 600 /tmp/r17_gpu.lock timeout --signal=TERM 1280s env \
  CUDA_VISIBLE_DEVICES=GPU-98d36ed2-a8e9-fb07-701c-41f337a6c1f9 \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false BATCHING_STAGE1_WORKERS=1 BATCHING_STAGE2_WORKERS=1 BATCHING_STAGE3_WORKERS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src:/data/mw_finetune/vendor/lerobot/src:/data/mw_finetune/vendor/FastWAM/src \
  /data/mw_finetune/envs/fastwam/bin/python -u -m exp.r17_integration.models.fastwam.batch_profile \
  --descriptor exp/r17_integration/models/fastwam/measurements/local_descriptor.json --refresh-code-seal \
  --observations /data/mw_finetune/qualification/observations/libero_original256_fix2_v1/fastwam_observations.npz \
  --receipt exp/r17_integration/models/fastwam/measurements/local_sweep.json \
  --expected-gpu-uuid GPU-98d36ed2-a8e9-fb07-701c-41f337a6c1f9 --deadline-seconds 1200 --repeats 3
```

`--refresh-code-seal` rehashes current code only; `load_adapter` still rehashes
and checks **all** model/Wan/norm/tokenizer assets before loading. Do not edit
source during that verification. Preserve old receipts when repeating.

CPU verification: **238 passed / 0 skipped** (129 Fast-WAM + 109 common),
including 19 new batching regressions and real localhost WebSocket batching,
with CUDA hidden/JAX CPU. Legacy checks:
**89 passed / 6 initially skipped / 1 EGL deselected**, followed by all **6
localhost socket skips passing** under explicit manual flags. The four shared
serving/cache files have no git diff. Receipts are
`tests/models/fastwam/receipts/batching_final2.xml`, `batching_legacy.xml`,
`batching_sockets.xml`, and earlier `batching_targeted2.xml`. These are CPU/framework
checks, not a physical LIBERO run. The formal launcher command above and H100
batching were unexecuted at that earlier local measurement; the H100 normal-
launcher smoke and CPU close-transfer fix below supersede that limitation.

## H100 normal-launcher qualification

Run: `P3_h100_fastwam_B8_smoke_20261010_1209`. One resident, four timan107
workers, ports23170/23171/23179, renderer GPU1, max-B8/5ms, capture/cache off,
K10/C5. H10080GB UUID `GPU-7ea0f760-d3aa-7023-c19b-95f49123311b` had
67,035MiB free before admission while two Cosmos residents occupied about13GB.
Only this run's receipt-bound processes were signalled. No local CUDA workload
or lock was used; P4's 4090 slot remained available to P4.

The resident lived **287.00s (4.78 GPU minutes)**, below the780s process bound;
its closed receipt reports284.35s inside serve. Four original-protocol episodes,
one per suite/init0, **4/4 successes**, zero errors/retries/rejected probes;
153 decisions, driver192.21s (**0.796 decisions/s, 1.25 episodes/min** including
worker registration/setup/shutdown). Mean resident request latency0.953s.
The physical four-worker run formed S3 meanB1.50/maxB3; it cannot establish a
16-worker filled-B8 episode rate or statistical success-rate equivalence.

The original final receipt was PARTIAL solely because P3's batched entry omitted
the parent `servers/manifest.json` required by `snapshot_run`. All four episodes
and native execution checks had passed; every owned process had closed. The
entry now writes the same manifest as the ordinary resident. A CPU regression
serves real S07 fake connections, closes, snapshots and installs the evidence.
CPU-only recovery verified the archived implementation pin and closed PIDs,
created the missing manifest, transferred the SHA-attested server archive and
audited all153 compositions. **No GPU/episode rerun**; the original receipt is
preserved and the recovered receipt is PASS. The final two-line transfer fix and
treatment-v2 resource bookkeeping were checked on CPU after this GPU run; native
model/stage/RNG/batcher/S07 code did not change.

- [Recovered smoke receipt](/data/mw_finetune/eval/P3_h100_fastwam_B8_smoke_20261010_1209/RECOVERED_RECEIPT.json)
- [Original receipt](/data/mw_finetune/eval/P3_h100_fastwam_B8_smoke_20261010_1209/RECEIPT.json)
- [Physical summary](/data/mw_finetune/eval/P3_h100_fastwam_B8_smoke_20261010_1209/SUMMARY.json)
- [Native replay/profile](/data/mw_finetune/eval/P3_h100_fastwam_B8_smoke_20261010_1209/servers/replica_0/batch_qualification.json)
- [Qualification summary](/data/mw_finetune/eval/P3_h100_fastwam_B8_smoke_20261010_1209/servers/replica_0/BATCH_QUALIFICATION.json)

One synchronized H100 B1 profile:704.62ms total; VAE15.34ms(2.18%), video
prefill136.18ms(19.33%), K10 action head443.23ms(62.90%), other109.87ms(15.59%).
Whole-board mean461.5W/max469.5W, including Cosmos; these figures do not isolate
Fast-WAM power or explain standalone H100 utilization. SM occupancy unmeasured.
Replay uses the same three recorded frames/seeds as the local sweep, one
excluded warmup and **one measured repeat** per row:

| H100 path | Decisions/s | Mean / p95 request latency s | Peak allocated/reserved GiB | Max / median / p95 normalized drift | Actual mean S1/S2/S3 B |
|---|---:|---:|---:|---:|---:|
| Direct B1 |1.562|0.640 / —|23.39 /23.66|0 /0 /0|1/1/1|
| Direct B8 |8.397|0.953 / —|24.15 /24.53|0.015625 /0 /0.00390625|8/8/8|
| S07 queued B1 |1.308|0.764 /0.764|23.50 /25.38|0 /0 /0|1/1/1|
| S07 max-B8,8 clients |**4.952**|1.418 /1.614|24.17 /27.60|**0.015625 /0 /0.00390625**|4.25/4.25/4.25|

H100 max-B8 gains **3.78× vs queued B1**, **3.17× vs direct B1**; full native
B8 gains5.38× vs direct B1. Eight-client preparation fragmented into batches of
five and three. The B8 capacity itself passes, but the replay's small arrival
pool does not keep the queue full. Whole-board mean/max power was431/446W for
S07 max-B8 and499/508W for directB8, again with Cosmos sharing the device.

## Replicas, workers and the two-GPU schedule

Use **one resident on each GPU,16 active workers per resident** for the joint
schedule. Local48GB cannot fit two ~26GB replicas. H10080GB can budget two solo
residents at34GiB each (68GiB total; at least4GiB free headroom), but two36GiB
budgets plus the observed13GB Cosmos fleet exceed80GB. Two H100 residents were
not throughput-tested. With two active Fast-WAM GPUs, timan107's measured
clean-node RAM ceiling is32 workers, so H1002/local1 with16 each is inadmissible.
Choose H1001/local1 rather than divide16 H100 arrivals across two residents.

Sixteen connections give two B8 groups in flight and enough backlog for the
slow S3 stage. This is the starting concurrency recommendation, not a guarantee
that every stage/arrival forms B8. Read `RECEIPT.batching_audit` and
`servers/replica_*/batching.jsonl` for actual S3 mean/max/member composition;
the four-worker smoke explicitly showed insufficient fill. Do not claim the
burst multiplier as an episode-rate multiplier. If S3 remains unfilled, inspect
the recorded arrival pattern before adding replicas; the accepted window stays5ms.

M's **final** measured recommendation is `--worker-max-episodes 4
--worker-memory-mib 6144`. Driver memory is not captured by worker RSS alone:
task-dependent host depletion reached0.544GiB/new dispatch; their conservative
bound is Pss2.4 +4×0.6 +0.8 transient/driver =5.6GiB, rounded to6GiB. N8 would
instead require9GiB/worker. Q's new allocator reserves1024MiB per renderer
worker plus per-card headroom and distributes over an explicit GPU allowlist.
The commands give H100 workers GTX1080s1/2/3 and local workers4/5/6; device0 is
left to the existing Cosmos fleet. No workers on timan108/timan1. Full32-worker
concurrency waits for RAM admission, normally after Cosmos exits naturally;
do not reduce the reservation or stop it to force admission. The smoke used
the earlier explicit N8 setting for one episode/worker; formal runs use N4.

Schedule: **H100 official four-suite control, then C16**; concurrently **4090
two-suite control, then C10,C20,C32,K2,K1**. All arms run spatial+libero_10 with
original init0..49. The extra 4090 control covers just those two suites, avoiding
another unnecessary58k worst-case K10 decisions. Each arm requires its own GPU
lane's completed control. Both controls are K10/C5; max-B8/5ms, one resident,
16 workers and the lane's recycling/renderer settings remain identical.
`--official-50` means original50-init coverage in the system C5/seed7 protocol;
it does not relabel this as the released seed42/32-step protocol.

The CPU enumeration of64 arm assignments in
[formal_schedule.json](measurements/formal_schedule.json) balances measured
4.95 H100 /5.98 local decisions/s: worst-step native estimates8.72h H100,
8.50h local, versus ~16.3h for the measured one-resident H100-only lane. C20
on H100 differs by only ~32s in this noisy estimate. This split is faster for
the measured one-resident alternatives and uses both GPUs within the host
budget; linear scaling from two H100 B8 residents is unqualified.

Projected campaign durations below assume50–100% of suite step caps, the
measured burst rates, and20% extra for transport/render/recycling. They are
**planning estimates**, not measured campaign durations;16-worker fill and
success/step-length distributions can move them. K2/K1 estimates scale the
local measured84.6% action-head cost with native step count and retain other
work; reduced-K GPU throughput has not been measured. Expected combined
elapsed ~5–10.5h after both lanes can start (plus any P4 lock/RAM wait).

| Campaign | GPU | Episodes | Native selection | Planning hours |
|---|---|---:|---|---:|
| Official four-suite control |H100|2000|K10/C5 default|4.4–8.9|
| C16 |H100|1000|`--commit-horizon 16`, defaultK10|0.8–1.6|
| Two-suite paired control |4090|1000|K10/C5 default|2.1–4.1|
| C10 |4090|1000|`--commit-horizon 10`, defaultK10|1.0–2.1|
| C20 |4090|1000|`--commit-horizon 20`, defaultK10|0.5–1.0|
| C32 |4090|1000|`--commit-horizon 32`, defaultK10|0.3–0.7|
| K2 |4090|1000|`--denoising-steps 2`, defaultC5|0.7–1.3|
| K1 |4090|1000|`--denoising-steps 1`, defaultC5|0.5–1.0|

## Exact normal-launcher commands

[formal_commands.sh](formal_commands.sh) contains all eight commands and stops
the lane if a control/campaign fails. It prints by default; the following two
commands run the lanes in **separate terminals**, without backgrounding another
campaign inside either lane. The local lane takes `flock -w 600 /tmp/r17_gpu.lock`
for each campaign, waiting for P4's measurement. Use new run names if repeated.
No formal baseline/study campaign in this command list was started by P3.

```bash
cd /home/weiland/projects/openpi
bash exp/r17_integration/models/fastwam/formal_commands.sh h100 --run
```

```bash
cd /home/weiland/projects/openpi
bash exp/r17_integration/models/fastwam/formal_commands.sh local --run
```

Review the fully expanded literal CLI commands before launch:

```bash
bash exp/r17_integration/models/fastwam/formal_commands.sh h100 --show
bash exp/r17_integration/models/fastwam/formal_commands.sh local --show
```

Expanded lists are also retained as
[formal_h100_commands.txt](measurements/formal_h100_commands.txt) and
[formal_local_commands.txt](measurements/formal_local_commands.txt).
[formal_commands_cpu.json](measurements/formal_commands_cpu.json) proves all8
parse with the actual normal launcher, all6 requested arms, B8/5ms/N4 and
timan107 only. Normal baseline CLI, expanded from the H100 lane:

```bash
bash exp/r17_integration/ops/formal_eval.sh --model fastwam_libero \
  --checkpoint /data/mw_finetune/models/fastwam_libero --server-node h100 \
  --worker-node timan107 --replicas 1 --workers 16 \
  --max-batch 8 --batch-wait-ms 5 --worker-max-episodes 4 \
  --worker-memory-mib 6144 --worker-vram-mib 1024 \
  --worker-egl-device 1 --worker-egl-devices 1,2,3 --memory-budget-gib 36 \
  --port 23170 --driver-port 23179 --official-50 --seconds 43200 \
  --episode-seconds 1800 --retries 2 \
  --run-name R17_FW_B8W5_H100_official50_control --run
```

Latest CPU suite: **413 passed,0 skipped,29 subtests passed**, including shared
default-off HEAD byte checks and Q's concurrent fleet/memory/recycle checks.
Final acceptance **62 passed,0 skipped**, including the close-transfer and
HEAD byte guards. XML and exact commands are under `measurements/cpu_formal_*.xml`.
Final legacy refresh89 passed/6 initial skips/1 EGL deselected, then all6
skipped socket tests passed explicitly (cpu_formal_legacy_final.xml and
cpu_formal_sockets_final.xml).
Tracked serving/cache files remain HEAD-identical. DP's owned batch entry is
also wired through the same normal flags (max32); its GPU/quality qualification
belongs to P4 and is not certified by this Fast-WAM smoke. Cosmos has no native
stage-batching hook here and is unchanged by P3.


## CUDA MPS qualification — 2026-10-10

The normal `ops/formal_eval.sh` / `formal_eval.py` now accept **`--mps`**, default
off. Recommend it only as an optional **4090 DP K100 / two replicas / B8** path:
**1.155×** versus that same two-replica configuration off. Keep Fast-WAM MPS off
on4090; H100 Fast-WAM throughput with MPS is **unqualified**. No live campaign
was stopped, signalled, or reconfigured by P3. No GPU compute-mode write occurred.

The daemon has a private campaign/GPU pipe and log directory,
`<campaign>/mps/<GPU hash>/{p,l}`, mode0700. Replicas alone receive those paths,
the selected full GPU UUID and100% active-thread percentage. An exec wrapper
removes other inherited CUDA_MPS_* settings before CUDA initializes; proxy,
workers and conductor retain the original environment. The leased, receipt-bound
MPS guard starts before replicas and closes after them, including ordinary
startup/model failures. Receipt records daemon PID/start ticks, exact attached
clients, private-server VRAM overhead, closure and existing context identity.
Only the private daemon receives `quit`; the default socket is never addressed.
NVIDIA documents the private-pipe isolation and client/server interface in its
[environment reference](https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html)
and [control reference](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html).

MPS management/probe files live in `models/fastwam/ops/`, outside the native
root-file seal; CPU regression verifies the native code bytes exactly equal
the sealed hs3 B8 control. Full runtime archives still pin the helpers. Prepare
a **fresh** normal runtime; do not use `--runtime-from` with a pre-MPS release.
Existing Default compute mode is required; `--gpu cuda:0` plus physical
`CUDA_VISIBLE_DEVICES=<GPU UUID>` selects the device. The pipe must fit85bytes;
automatic run names fit; custom names are checked during CPU preparation. Reusing an existing private namespace is refused.

All actual local A/B cases used32 clients, three recorded real LIBERO frames,
fixed independent request seeds,5ms queues and50ms client think time. DP uses
P4's unchanged actual formal WebSocket resident/graph load generator; Fast-WAM
uses the native S07 full-call path, including prepare and all three stages.
Fast timings exclude network/episode simulation. Rates pool completed calls
and measured time; p50 pools caller latency samples; board means include any
background. Cold startup is excluded from throughput and charged to the GPU
allowance. Each row has two repetitions, ordered off/on/off/on.

| GPU / config | MPS | calls/s | p50 ms | board util / W | original background DP calls/s, repetitions |
|---|---|---:|---:|---|---|
|4090 DP K100 R1 B16|off|14.381|1747|64.7% /202.5|12.774,6.954|
|4090 DP K100 R1 B16|on|13.653|1815|47.7% /170.2|6.357,0|
|4090 DP K100 R2 B8|off|22.659|1138|59.5% /192.5|0,0|
|4090 DP K100 R2 B8|on|26.163|977|59.0% /209.9|0,0|
|4090 Fast K10 R1 B8|off|6.946|4338|37.1% /194.4|0,0|
|4090 Fast K10 R1 B8|on|6.768|4404|35.6% /190.9|0,0|
|4090 Fast R2|both|—|—|—|does not fit48GB; one B8 peaks25.9GiB allocated /28.1GiB reserved|
|H100 new Fast R2 + live Cosmos R2|both|—|—|—|memory admission refused; four existing contexts retained|

DP R2 pair gains were14.9% and16.1%; pooled ratio1.1546. DP R1 ratio0.9494 is
confounded: the original background stopped independently during the sweep.
Fast ratio0.9744 shows no demonstrated single-replica gain. A later local
background context appeared during a tiny guard test and was retained.
**This is not an A/B under one constant original live background.** Original
hs3 DP/Fast receipts became PARTIAL with campaign-stopped/deadline messages
and all_owned_closed=true; a new H100 Fast fleet appeared. First local DP-on
pair retained the original background GPU PID/start ticks. Do not attribute
the background rate loss solely to MPS. Later zero-rate pairs describe their
observed load; a general busy-GPU interference benefit is not established.

At H100 admission there were14,408MiB free; two owned Fast residents required
63,488MiB including a4GiB margin. Loading was refused before CUDA. An earlier
3.43s private-MPS CUDA compatibility check passed with both Cosmos contexts
unchanged. It did not load Fast-WAM and does not qualify mixed-model throughput.
A future H100 probe requires a window after the Fast fleet closes while the
same Cosmos fleet remains; use16 clients per owned Fast replica,32 total.

DP peak whole-board VRAM was3,126/3,130MiB for R1 off/on and2,284/3,799MiB
for R2 off/on, reflecting changing background as well as probes. Actual stage3
batch means were15.95/15.92 and7.95/7.96. Fast peak owned allocation was
26,492/26,521MiB; reservation28,774/28,618MiB off/on; actual stage3 mean7.89.
Fast eager normalized outputs were **bit-identical on/off** for the three fixed
frames. All batched outputs versus the common off/eager reference had max abs
**0.015625**, median0.000488–0.000977, p950.003906. DP eager wire outputs were
bit-identical across all four on/off pairs; batch-vs-eager max0.000345707.
These checks provide no physical episode SR certificate.

Conservative charged windows, including CPU load time and two failed setup
attempts:1,073s, **17.9 GPU minutes total across both GPUs**, under20. All owned
clients and MPS daemons closed. Failed Fast attempts are preserved: first stale
code descriptor, then an incorrect profiler device property; both were corrected
and rerun. No throughput numbers come from failed runs.

Receipts:

- `measurements/mps_results.json`: pooled metrics, individual rounds, numerical checks.
- `/data/mw_finetune/eval/p3mps_ab_1010`: eight DP A/B cases, original background snapshots.
- `/data/mw_finetune/eval/p3mps_fw2_ab_1010` and `p3mps_fw3_1010`: four passing Fast cases.
- `/data/mw_finetune/eval/p3mps_exec_1010/receipt.json`: actual guard/exec/attachment/closure PASS; inherited1MiB limit removed before16MiB allocation, existing context retained.
- `measurements/mps_h100_check.json`: tiny CUDA compatibility/context-preservation PASS.
- `measurements/mps_h100_results.json`, `mps_h100_closed.json`: model sweep SKIP_MEMORY, contexts preserved.

Example normal **new** DP campaign, not executed; only launch after worker RAM,
renderer, ports and GPU admission are available. This changes topology/batching
from the old R1/B16 control and requires a matching control for paired studies.
Keep the existing non-MPS study arms as they are.

```bash
CUDA_VISIBLE_DEVICES=GPU-98d36ed2-a8e9-fb07-701c-41f337a6c1f9 \
  exp/r17_integration/ops/formal_eval.sh \
  --model dp_libero --server-node local --worker-node timan107 \
  --replicas 2 --workers 32 --max-batch 8 --batch-wait-ms 5 --mps \
  --worker-memory-mib 6144 --worker-max-episodes 4 --memory-budget-gib 8 \
  --port 23160 --driver-port 23179 --official-50 \
  --run-name mpsdp_r2_$(date +%m%d_%H%M%S) --run
```

Rerun local shared probes through the owned CPU client driver (not executed
again). `--models dp` uses K100; hold the 4090 lock only if claiming exclusive
timing. Keep shared background metadata and use a new unique output directory.

```bash
PYTHONPATH=.:src:packages/openpi-client/src \
OPENPI_DISABLE_STAGE_STREAMS=0 BATCHING_STAGE1_WORKERS=1 \
BATCHING_STAGE2_WORKERS=1 BATCHING_STAGE3_WORKERS=1 \
  .venv/bin/python -m exp.r17_integration.models.fastwam.ops.mps_probe \
  --out /data/mw_finetune/eval/mpsdp_ab_NEW --models dp \
  --settings off on off on --seconds 16 --budget-seconds 700 \
  --background /data/mw_finetune/eval/ACTIVE_DP_CAMPAIGN
```

H100 bounded mixed qualification helper is `ops/mps_h100_probe.py`; deployed
copy `/data/mw_finetune/eval/p3mps_h_ab_1010/probe.py` has explicit source/descriptor
and Cosmos background pins. Its attempted execution used `--budget-seconds315`
and returned SKIP_MEMORY. Create a fresh copy/output for any later run. The
normal Fast launcher option is simply its existing matched B8 command plus
`--mps`, a short new run name and fresh preparation; no H100 model recommendation
is made without the mixed throughput/parity receipt.

**Mid-study decision:** STUDY.md §3 and §6.2 require service-ms/control-step in
addition to model-call or actual-NFE/control-step reductions. MPS changes
scheduling and wall time. Remaining on arms cannot use old off baselines for
the complete cost adoption gate. Either keep MPS off, or acquire a matched on
timing reference. Existing SR/count baselines may be retained under unchanged
native/batch identity and parity gates; throughput cannot prove SR invariance.
The MPS receipt explicitly marks unlike-MPS control wall-time comparisons false.
On/off resume mixing is rejected before attach; direct shared-resume enforcement
is requested from Q. No STUDY.md, baseline or live campaign was edited.


Additional background audit: DP R2's first off/on pair had no external CUDA
context and gained14.9%. During the second on repetition another CUDA context
appeared (PID1890260,1486MiB at the final snapshot). Its call rate was not
captured; the old hs3 background counter remained0. Thus pooled1.1546x is an
observed ratio, not a two-repeat estimate under identical production load.
The second pair's16.1% gain has that qualification. Do not describe all later
zero old-background counters as an idle GPU. A further context appeared during
the tiny guard test and was retained. No additional context was signalled.


## Exclusive H100 capacity sweep — 2026-10-10,14:16–14:24 CDT

This supersedes the earlier H100 B8 and MPS-pending recommendations. H100 was
exclusive and empty, confirmed before and after. No DP/Cosmos process, worker,
GPU mode or live release was changed. Total conservative parent wall time,
including two failed CPU-only packaging attempts, was **410.17s /6.84min**.
All own children and the private MPS daemon are closed. Native descriptor is
`measurements/h100_capacity_descriptor.json`; non-code digests were retained
and fully verified by the loader. Only the opt-in BatchConfig ceiling changed
from16 to32; model forward, weights, normalization, and stock S07 were unchanged.
Fresh baseline preparation is required because the native code seal changed.

48 independent closed-loop clients,50ms think, K10, system commit5; three real
recorded LIBERO observations (256 camera input, native224 preprocessing),
fixed private seeds42/43/44. Clients are evenly assigned to replicas. This is
full native MISS through the production stage adapter and stock S07, **not a
websocket/proxy/conductor/simulator campaign**. Calls equal model decisions;
no policy tails/retrieval hits occur. Every configuration gets a concurrent
warmup, then23s of sustained requests; completion rate is measured over the
20s window after3s of priming. Priming and drain are saved but excluded from
rate/latency/power. Fill percentiles count S3 batch events, not request rows.
Short windows provide point estimates, not confidence intervals.

| Configuration (replicas × max B / wait) | MPS | Calls/s | Decisions/s | p50 / p90 ms | S3 fill p50 / p90 | GPU busy %* | Mean W | Max normalized drift | Peak allocated / reserved GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1×B16 / 5 ms | off | 17.60 | 17.60 | 2767 / 2921 | 16 / 16 | 45.6 | 293.8 | 0.015625 | 27.78 / 31.83 |
| 1×B24 / 5 ms | off | 19.20 | 19.20 | 2425 / 2671 | 24 / 24 | 44.6 | 312.0 | 0.015625 | 28.82 / 34.13 |
| 1×B32 / 5 ms | off | 17.60 | 17.60 | 2586 / 2707 | 16 / 32 | 41.8 | 300.1 | 0.015625 | 28.89 / 38.54 |
| 1×B16 / 10 ms | off | 16.80 | 16.80 | 2680 / 2920 | 16 / 16 | 44.8 | 295.2 | 0.015625 | 27.90 / 31.79 |
| 1×B24 / 10 ms | off | 19.30 | 19.30 | 2354 / 2984 | 24 / 24 | 50.1 | 311.0 | 0.015625 | 28.88 / 33.60 |
| 1×B32 / 10 ms | off | 19.20 | 19.20 | 2486 / 2811 | 24 / 32 | 43.4 | 301.7 | 0.015625 | 28.96 / 34.85 |
| 2×B16 / 5 ms | off | 24.80 | 24.80 | 1858 / 1984 | 16 / 16 | 86.6 | 374.4 | 0.015625 | 51.99 / 59.55 |
| 2×B16 / 10 ms | off | 26.35 | 26.35 | 1765 / 1901 | 12 / 16 | 87.7 | 385.4 | 0.015625 | 52.06 / 60.04 |
| 2×B16 / 5 ms | on | 28.00 | 28.00 | 1679 / 1779 | 16 / 16 | 75.7 | 412.6 | 0.015625 | 51.98 / 60.16 |
| 2×B16 / 10 ms | on | 27.60 | 27.60 | 1656 / 1730 | 16 / 16 | 77.1 | 419.2 | 0.015625 | 52.19 / 60.43 |
| 3×B16 /5 ms | on | — | — | — | — | — | — | — | Not admitted: B16 workspace + context/headroom exceeds80 GiB |

*GPU busy is NVML utilization.gpu (nvidia-smi's utilization counter), **not SM
occupancy or percentage of peak compute**. No SM-occupancy measurement was
made. Under MPS this counter decreased despite higher throughput and power;
rank by completed decisions/s, not this counter or the700W cap.

**Recommendation: H100 R2/B16/5ms, private MPS on,48 workers (24 per replica).**
28.0 calls/s, p50/p901679/1779ms, full16-row S3 batches,412.6W,52.0GiB allocated /
60.2GiB reserved across the replicas. It gains12.9% over the same R2/B16/5ms off
and45.1% over the fastest single-replica point. Its capacity ratio to the
owner's observed oldR2/B8 campaign3.4calls/s is **8.24×**; campaign episode
throughput improvement is unmeasured. The10ms MPS point was27.6calls/s, so use5ms.
R2 off already overlaps host preparation/launch gaps: the assertion that a
second replica can never help without MPS is contradicted by this replay.
B32/5ms alternates16/32-row batches at48 clients; B24 fits that client count
better, but R2/MPS beats both. There is no evidence more workers are needed.

Three replicas were not launched. Using even the R2 measured allocated peak
(approximately26,721MiB per resident), three residents plus1GiB/context and4GiB
headroom need approximately87,331MiB, versus81,081MiB free. The conservative
B16 reserved-memory admission asks104,956MiB. This is a memory refusal, not a
measured throughput result. An initial overly conservative admission reused
B32 workspace for B16; the corrected CPU-tested rule uses targetB16 memory,
and the R2 cases were rerun successfully. Raw first-sweep skip rows remain
preserved for audit and are superseded by the successful multi-replica sweep.

All ten configurations have full-chunk normalized max drift **0.015625**,
median0, p950.00390625 versus that replica's eagerB1 with the same frame/seed.
Recommended first5 committed actions max0.0078125. This stays within the
previously observed/qualified0.015625, not merely the broader0.03125 smoke
threshold. Only three replay observations/K10 were measured; this proves no
success-rate equivalence and no new K1/K2 numerical certificate. The fresh
control and arms measure those outcomes. Batch composition is in every raw
stage ledger; actual formal decisions continue to use the existing receipt seam.

Worker feasibility uses the **owner-selected hs4** reservations: Fast48 at
RAM2048/VRAM768MiB and DP32 atRAM2048/VRAM576MiB, N4 recycling, EGL spread0..7.
Read-only timan107 admission at14:19 returnedadmitted=true:156,898,279,424B
available,23,652,019,814B headroom,103,079,215,104B requested,0 pending growth;
32 live DP workers were observed, not altered. These are cooperative owner
reservations, not certified all-suite peak-memory limits. Launcher must recheck
RAM and renderer VRAM at launch; do not bypass refusal or add workers. The
combined per-card reservation is6×768+4×576=6912MiB on eight8GiB cards.

Exact coordinator launch, **not executed** (default model K10/C5; all four suites):

```bash
bash exp/r17_integration/ops/formal_eval.sh \
  --model fastwam_libero --server-node h100 --worker-node timan107 \
  --replicas 2 --replica-startup-width 2 --max-batch 16 --batch-wait-ms 5 --mps \
  --workers 48 --extra-workers 0 --worker-memory-mib 2048 --worker-vram-mib 768 \
  --worker-max-episodes 4 --worker-egl-devices 0,1,2,3,4,5,6,7 \
  --gpu cuda:0 --memory-budget-gib 70 --port 23173 --driver-port 23177 \
  --official-50 --suites libero_spatial libero_object libero_goal libero_10 \
  --run-name fwh_r2b16mps_ctrl_1010 --seconds 43200 --run
```

**Every arm uses the same GPU/R2/B16/5ms/MPS/48-worker/reservation/recycling/EGL
configuration and the freshly completed control's frozen runtime.** Do not use
the stopped/failedB1/B8 controls, or combine an MPS arm with an off timing control.
The new code/treatment seal rejects old batch identities; MPS comparison scope
also explicitly records wall-time comparability. Match the MPS flag, which is
not implied by `--runtime-from`. Study cost includes service time/GPU wall hours,
so this scheduling change requires the new matched control.

Exact sequential matching arm commands, **not executed**, after the control is
closedPASS with all2000 episodes accepted:

```bash
set -e
CONTROL_RUN=/data/mw_finetune/eval/fwh_r2b16mps_ctrl_1010
FW_ARGS=(
  --model fastwam_libero --server-node h100 --worker-node timan107
  --replicas 2 --replica-startup-width 2 --max-batch 16 --batch-wait-ms 5 --mps
  --workers 48 --extra-workers 0 --worker-memory-mib 2048 --worker-vram-mib 768
  --worker-max-episodes 4 --worker-egl-devices 0,1,2,3,4,5,6,7
  --gpu cuda:0 --memory-budget-gib 70 --port 23173 --driver-port 23177
  --official-50 --suites libero_spatial libero_10 --seconds 43200
  --paired-control "$CONTROL_RUN" --runtime-from "$CONTROL_RUN"
)
for C in 10 16 20 32; do
  bash exp/r17_integration/ops/formal_eval.sh "${FW_ARGS[@]}" \
    --commit-horizon "$C" --run-name "fwh_mps_c${C}_1010" --run
done
for K in 2 1; do
  bash exp/r17_integration/ops/formal_eval.sh "${FW_ARGS[@]}" \
    --denoising-steps "$K" --run-name "fwh_mps_k${K}_1010" --run
done
```

C arms keep nativeK10; K arms keep systemC5. Do not specify both knobs in one
command. All arms stay onH100; local4090/DP is left undisturbed. Capacity rate
alone cannot give a reliable campaign duration: accepted episode control counts,
simulator/network latency, recycling, and task mix have not been measured here.

CPU validation after the final cap/test changes: **539 passed +43 subtests,
0skip**, full Fast-WAM/common/ops; **71passed,1 real-EGL test deselected,0skip**
legacy serving/proxy/MW, including real localhost proxy cases. Focused large-B
native-row isolation, private noise, admission, default-off and CLI regressions:
72passed before the admission regression was added; final full suite includes
it. Earlier full run's obsolete17 ceiling assertion failed, was corrected to33,
and the full suite was rerun successfully. Default-off actions/manifest and
metadata remain byte-identical; create_runtime(B1) creates no queues/threads/files.
Shared cache/S07/websocket sources have zeroHEADdiff. No launch or SR is claimed.

Receipts: `measurements/h100_capacity_results.json`, full raw JSONL/NPZ/logs under
`measurements/h100_capacity_raw/`, exact first probe script snapshot
`measurements/h100_capacity_probe_v1.py`, tests under
`tests/models/fastwam/receipts/h100_capacity_*xml`. Final H100 query: no contexts,
Default mode,0MiB used,0% utilization. Private MPS receipt daemon_closed=true.
