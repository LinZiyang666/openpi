# P3 → R: MPS results for RUN_CONFIGS (2026-10-10)

The normal formal_eval.py/sh now accepts opt-in --mps. Default remains off.
Private pipe/log directories live under campaign/mps/<GPU hash>/{p,l},0700.
Only replicas attach; the receipt-bound leased guard starts first and closes
last. Receipt records config, UUID, daemon PID/start ticks, exact client list,
physical VRAM, closure and background context preservation. Existing compute
mode must already be Default; no mode change. Automatic MPS names fit; custom names must keep pipe<=85B (CPU checked).
Fresh normal CPU/runtime preparation is required; do not reuse a pre-MPS
--runtime-from release. Native policy/code seals remain identical to hs3 B8.

Actual local4090,32 clients,5ms, real fixed LIBERO frames, interleaved repeats:

| Config | MPS | calls/s | pooled p50 ms | mean GPU util / W | background DP calls/s per repetition |
|---|---|---:|---:|---:|---|
| DP K100 R1 B16 | off |14.381|1747|64.7% /202.5|12.774,6.954 |
| DP K100 R1 B16 | on |13.653|1815|47.7% /170.2|6.357,0 |
| DP K100 R2 B8 | off |22.659|1138|59.5% /192.5|0,0 |
| DP K100 R2 B8 | on |26.163|977|59.0% /209.9|0,0 |
| Fast K10 R1 B8 | off |6.946|4338|37.1% /194.4|0,0 |
| Fast K10 R1 B8 | on |6.768|4404|35.6% /190.9|0,0 |

Recommend MPS only as an optional DP R2/B8 configuration:1.155x versus
the same two-replica configuration off, consistently +14.9/+16.1% in the
two pairs. Do not substitute this topology for an existing R1/B16 control.
DP R1 has no demonstrated gain; its background changed, so no causal estimate.
Fast single replica has no gain (0.974x), so retain MPS off on4090.

H100 R2 Fast + live Cosmos2: refused own new replicas at admission (14,408MiB
free vs63,488MiB needed). Existing four contexts preserved; no model throughput
A/B and no H100 speedup qualification. Keep H100 MPS off pending a window with
only the fixed Cosmos background. Tiny H100 private-MPS CUDA compatibility check
passed and preserved both Cosmos contexts; that is not model qualification.

No campaign was stopped/reconfigured by P3. During the local tests the original
hs3 DP and Fast receipts independently changed to PARTIAL / campaign stopped
or hard deadline exceeded / all_owned_closed=true; a new Fast fleet appeared.
First DP on pair retained its original background GPU PID/start identity.
The observed background rate drop cannot be attributed solely to MPS; later
zero-rate pairs describe that observed background, not the initial live load.

DP eager B1 outputs are bit-identical across all four on/off pairs; batch-vs-eager
wire max0.000345707. Fast eager normalized B1 outputs are bit-identical on/off;
batch-vs-fixed-off-eager normalized max0.015625 (composition drift); no SR claim.
All owned MPS daemons/clients closed. Total conservative probe windows17.9min,
including failed setup/load attempts and both tiny CUDA checks.

STUDY.md §3/§6.2 includes service-ms/control-step and GPU wall hours as well as
M/T and actualNFE/T; adoption requires comparable service time. **Do not enable
MPS only on remaining arms and claim wall-time cost savings against old off
baselines.** SR/count baselines may remain under unchanged native/batch identities
and parity gates; throughput alone proves no SR invariance. Keep current arms
off, or collect a matched on timing reference. Receipt marks mixed MPS-control
wall-time comparison false; resumed on/off mixing is refused in the P3 shim.

Details/results/commands: models/fastwam/BATCHING.md MPS section and
models/fastwam/measurements/mps_results.json. Real guard/exec CUDA check:
/data/mw_finetune/eval/p3mps_exec_1010/receipt.json (PASS, old context retained).


Additional background audit: DP R2's first off/on pair had no external CUDA
context and gained14.9%. During the second on repetition another CUDA context
appeared (PID1890260,1486MiB at the final snapshot). Its call rate was not
captured; the old hs3 background counter remained0. Thus pooled1.1546x is an
observed ratio, not a two-repeat estimate under identical production load.
The second pair's16.1% gain has that qualification. Do not describe all later
zero old-background counters as an idle GPU. A further context appeared during
the tiny guard test and was retained. No additional context was signalled.


## New exclusive H100 result —14:24 CDT; supersedes prior H100 MPS-pending advice

Measured2026-10-10,14:16–14:24 CDT

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
