# R17 concurrent launch reference

Last verified: **2026-10-11 00:1x CDT** (cache-study topology section); other sections 2026-10-10 14:33.
Coordinator launch decisions belong here. **Replace a recommendation immediately
when better evidence invalidates it; do not leave obsolete advice as an alternative.**
Refresh after each completed campaign, qualification, or resource change.
Commands below are templates; writing this reference did not launch, stop,
signal, deploy, or alter a campaign.

**Current launch reference:** Fast-WAM LIBERO on H100 = **R2/B16/5 ms/MPS on,
group-commit, timan107 x48**, baseline `hs6fw_ctl_143332`. DP LIBERO on the
48 GB4090 = **R2/B8/5 ms/MPS on, group-commit, timan107 x32**, baseline
`hs5dp_ctl_142812`. Their campaign/orchestration records verify these settings;
startup PARTIAL receipts are not completed baselines. The stopped hs3/hs4
controls and their queues are superseded and cannot authorize these arms.

**Cache study (current runs):** see [Cache-study topology in use](#cache-study-topology-in-use-2026-10-12) at the end.

P3/S/Q inputs are resolved. Fixes address host sampler/launch gaps and conductor
durability stalls, not a shortage of nominal workers. Replicas share GPU compute,
but independent host pipelines can overlap even without MPS; P3 measured that
gain. Select a measured treatment before a baseline and freeze it for its arms.

## Launch and pairing rules

- Server nodes are `h100` (80 GB) and `local` (the **48 GB** 4090, 49,140 MiB,
  not a stock 24 GB card). Simulators run only on worker nodes.
- **timan1 MetaWorld workers are owner-authorized on physical GPU3 only, max40.**
  Use `--worker-counts timan107=N,timan1=M`; the launcher pins timan1 to GPU3.
  timan107 has 8 x GTX 1080/8 GB and 220.28 GiB host
  RAM. The owner has run 64 workers there. That is useful concurrency history,
  not a universal capacity limit or permission to ignore current admission.
- timan108 has 4 x A5000; **GPU1/2 belong to another user**. It is currently
  offline: current launch capacity is zero. After recovery, explicitly select
  `--worker-egl-devices 0,3`, verify those two cards and old fleet closure,
  and repeat host/VRAM admission. Older documents describing three usable
  cards or timan108 as the automatic preferred host are superseded.
- Formal selectors include `pi05_mw`, `groot_mw`, `dp_mw`, `fastwam_mw`,
  `dp_libero`, `fastwam_libero`, `cosmos_libero`, `pi05_libero`, `groot_libero`.
  The two VLA LIBERO selectors use the shared formal/library/cache path; see
  [LIBERO VLA handback](../handbacks/HANDBACK_LIBERO_VLA.md) for commands.
- `--max-batch > 1` is admitted for **Fast-WAM/DP LIBERO and MetaWorld MT50** in the normal
  formal launcher. Other formal selectors use B1; a batching window has no
  effect on that direct path. Formal LIBERO π0.5/GR00T use B1 per replica.
- Baseline and arms must use the **same GPU type and batching configuration**.
  Also pin checkpoint/norm/protocol, replica count, worker node/count,
  per-stage workers/streams, renderer selection, RAM/VRAM reservation, and
  recycling. Fast-WAM treatment v2 enforces those resource fields, including
  the distinction between a null automatic EGL allowlist and an explicit list.
  Equivalent effective placement is insufficient if the sealed fields differ.
- **Worker count is fixed at baseline.** The coordinator must refuse arms
  unless resource topology is identical; Fast-WAM's pairing pins enforce this.
  An arm cannot compensate for a slow baseline with more workers,
  replicas, a different batch/window or MPS. Use
  `--runtime-from /data/mw_finetune/eval/<closed_baseline>` for its frozen
  server/model/worker runtime. Fast-WAM `--paired-control` also inherits it;
  DP requires explicit `--runtime-from`. Never reseal arms from the working tree.
  DP's current runtime verifier checks model/server/batching/MPS; it does not
  independently enforce every replica/worker resource field. Compare those
  sealed baseline fields explicitly when preparing DP's queue.
- Every batched Fast-WAM arm needs `--paired-control /data/mw_finetune/eval/<baseline>`:
  a completed, closed PASS K10/C5 official-50 baseline covering its suites,
  with valid composition audit and matching treatment. DP needs the same
  scientific pairing discipline, but **its CLI rejects `--paired-control`**;
  record its baseline path in the experiment/queue record.
- Batching preserves private request RNG but can change arithmetic. Keep
  batch membership ledgers; equal caps/windows do not guarantee equal rows
  in every realized batch. Qualification replay is not an SR certificate.
- **Match baseline MPS explicitly with `--mps`; `--runtime-from` does not
  enable it.** New hs6 Fast-WAM and hs5 DP baselines both use MPS. P3 measured
  H100 Fast-WAM R2/B16/5 ms at28.0 calls/s (+12.9% over the same topology off)
  and4090 DP R2/B8 at26.163 (+15.5% observed over off). Other model/benchmark
  MPS configurations remain unqualified; off-baseline arms stay off.
  MT50 DP/Fast-WAM now admit batched replicas and private MPS; throughput and
  production GPU batch drift remain coordinator measurements, not CPU test claims.
- Q's driver fix uses a separate **orchestration pin**, leaving model/worker
  bytes frozen. **Use `--driver-mode group-commit` for Fast-WAM/DP LIBERO.**
  Q's real-/data CPU replay and durability regressions passed. Its separate
  `ORCHESTRATION.json` records baseline/selected driver pins and permits an
  arm to select group-commit against an older-driver baseline while preserving
  model/worker runtime. This driver change is allowed; identify it separately
  in throughput/cost comparisons. Use sealed preparation/refresh, not flag
  overrides on a prepared launch.
- `--memory-budget-gib` is the **whole campaign's server budget**, divided
  across replicas, not a budget per replica. Budget co-resident campaigns
  together and retain device headroom. Startup waves are bounded by
  `--replica-startup-width`; they do not add inference compute.
- LIBERO official baselines require `--official-50`: original init0..49,
  500 episodes/suite, four suites = 2000. Earlier formal400 receipts below
  are throughput history, not acceptable replacement baselines. MT50 formal
  campaigns now have 2,500 episodes without this flag: seed7, init0..49 for
  every task. MT50 libraries use seed8, init0..49; dev is undecided and seeds9/10
  remain reserved. Horizon and denoising
  overrides cannot be supplied together, even if one equals the default.

All formal templates use a unique `--run-name`, `--gpu cuda:0`,
`--extra-workers 0`, and timan107. Replace `RUN_NAME` or `HHMMSS` with a fresh
short name and change ports only within the windows below. The current named
baselines are already launched. Older command recipes in linked handbacks or
BATCHING.md do not override this reference's selected configuration.
DP automatically receives `--cuda-graph` at the
resident entry; do not add that unsupported flag to the coordinator CLI.

### Ports and concurrent lane allocation

| Lane/use | Reserved window | Proxy/public | Backends | Local driver |
|---|---|---:|---|---:|
| π0.5/GR00T formal first slot | 23160..23169 | 23160 | 23161..(23160+R) | 23169 |
| Fast-WAM, either benchmark / live hs6 | **23170..23179** | 23173 | 23174,23175 for R2 | 23177 |
| DP, either benchmark / live hs5 | **23180..23189** | 23180 | 23181,23182 for R2 | 23189 |
| Cosmos normal / K2 | 23240..23249 | 23240 | 23241,23242 for R2 | 23249 |
| Cosmos retained-cell resume | same window | 23243 | 23244,23245 | 23248 |
| Legacy R16 local LIBERO, proposed allocation | legacy 23100..23199 | independent servers 23150..23155 | no formal proxy | 23159 |

Reserve driver ports on **local even when the server is H100**. All ports
within one campaign must be distinct. **The model windows above are coordinator
policy even where the parser accepts a broader legacy range.** π0.5/GR00T
share their first slot; nine model sections are not nine independent lanes.
R12/R16-era 23230..23233 H100 service ports and LIBERO-library 233xx ports
belong to other launchers, not formal_eval.sh. Workers connect directly (no
`tether expose`, owner rule 2026-10-12): an H100 server at `149.165.153.233:<port>`
(firewall open), weilandserver services through `ziyanglin.com`, which forwards
only TCP 23100..23199, and local workers through 127.0.0.1. With remote workers the
driver port (and a local server's public port) must lie in 23100..23199; the
launcher refuses anything else. Cosmos normal campaigns now default to driver 23112.

### Worker reservations and renderer spreading

**Every LIBERO fleet:** spread EGL over **all eligible worker GPUs** and
recycle after **4 episodes**. For new timan107 baselines use
`--worker-egl-devices 0,1,2,3,4,5,6,7 --worker-max-episodes 4`.
Current **hs6 Fast-WAM** seals the explicit0..7 list; **hs5 DP** seals
`worker_egl_devices: null`, which selects all eight eligible GPUs automatically.
Preserve each baseline's exact representation on its arms; equivalent effective
placement does not authorize changing a pairing field.

**Current hs6/hs5 reservations:** Fast-WAM x48 uses RAM2048/VRAM768 MiB;
DP x32 uses RAM2048/VRAM576 MiB; both N4. These are owner-selected cooperative
reservations, not process/driver memory caps or certified all-suite maxima.
These replace the old hs3 x32+x24 /RAM3072 entries. Do not silently increase
these fleets or change the reservation in an arm.

Renderer admission was the limiting resource for the simultaneous hs4 plan:
DP36/768 was refused with only16 slots available while Fast-WAM48/768 and
Cosmos8/1024 were present; after Cosmos closed, DP36/576 was still refused
because only32 fit. Fast-WAM48 x768 + DP32 x576 = **55,296 MiB (54 GiB)**
reserved across eight8 GiB cards: per card6 Fast workers +4 DP workers reserve
6912 MiB before physical occupancy/headroom. This filled the allocatable
worker slots; it is not a claim that all64 GiB were physically occupied.
Recycling gaps release physical contexts but retain fleet reservations.

The old6144 MiB RAM/1024 MiB VRAM envelope remains a conservative **unqualified
new-workload fallback**, not a measured32-worker hardware ceiling or the
current shared-fleet recommendation. Admission charges live RSS plus reserved
growth and max(8 GiB,10% node/cgroup capacity) headroom; recheck it at launch.

The former host-memory growth was mitigated by **D's renderer spreading plus
M/Q's process recycling**, not by RSS accounting alone. M measured node
depletion 0.091..0.544 GiB/new dispatch, strongly task dependent, including
driver allocation outside worker PSS. Its conservative bound is
PSS2.4 + 4 x 0.6 + 0.8 transient/driver = 5.6 GiB, rounded to 6144 MiB.
N8 instead needs 9216 MiB under that bound. Preserve planned recycling and
the separate bounded fault-restart limit; unlimited retry hides node failures.

Read-only live fleet snapshot, **13:04:19 CDT**, PID birth ticks verified:

| Fleet | Live workers | RSS mean / range MiB | PSS mean / range MiB | Fleet RSS / PSS GiB | Observed renderer memory |
|---|---:|---|---|---|---|
| Fast-WAM hs3 | 32 | 2349 / 1496..2772 | 2014 / 1164..2435 | 73.42 / 62.93 | 511 MiB mean among 15 visible contexts; max532 |
| DP hs3 | 24 | 1863 / 433..2750 | 1534 / 223..2414 | 43.67 / 35.95 | 479 MiB mean among 8 visible contexts; max532 |

All 56 worker PIDs were alive; Fast-WAM generations1/2 = 27/5 and DP = 9/15.
All were R/S, none D. A recycling/importing worker may have no current graphics
context, so these VRAM means are **not averages over all reserved workers**.
The allocator held all32/24 slots throughout those gaps. The per-card physical
usage snapshot ranged4..2650 MiB; assignments still spanned all eight cards.
MemAvailable rose111.58→116.51 GiB between13:03:15 and13:04:19; memory PSI
some/full avg10/60/300 were0. This verifies short-window recovery/headroom,
not a full four-suite peak or a proof of zero remaining driver growth.
No live Cosmos fleet appeared in this snapshot.

These older live PSS means verify approximately **1.5..2.0 GiB per worker**;
RSS was1.8..2.3 GiB. The coordinator records64 workers/~123 GiB at13:16.
That historical observation does not guarantee the newer80-worker footprint.
S's raw sample at**14:06:22 CDT** (worker census sampled about28 s earlier)
has Fast-WAM48 RSS126.68 GiB/**2.64 GiB each**, DP32 RSS66.99 GiB/**2.09 GiB
each**, total193.67 GiB; node MemAvailable48.22 GiB. The2 GiB reservation is
already below some observed RSS. D's uncontaminated spreading probe and the
earlier live contexts both observed **532 MiB renderer peak** (steady428 MiB
in D's probe). A576 MiB reservation leaves only44 MiB above that sampled peak;
use actual admission and task evidence, not an assumed unlimited RAM supply.

Evidence: [D's spreading probe](../handbacks/HANDBACK_COSMOS.md#libero-renderer-distribution-and-memory-probe-2026-10-10),
[M's bound](../REQUESTS/M_to_Q.md), [coordinator history](../../../logs/session_handoff.md),
[S raw census](/data/mw_finetune/perf/2026-10-10/perf.jsonl),
[DP admitted treatment](/data/mw_finetune/eval/hs4_dp_b16w5_4090_k100_c5_control_20261010_133915/campaign.json).

For MT50 use `--worker-memory-mib 2048 --worker-vram-mib 1024
--worker-max-episodes 0`, with the same explicit renderer allowlist for new
campaigns. MT50 host growth/recycling was not certified by the LIBERO fix;
do not invent a measured MT50 per-worker RSS from the live LIBERO fleets.

## π0.5 x LIBERO

**Formal selector:** `pi05_libero`. Original cached checkpoint:
`/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`.
Norms: `assets/pi05_libero/physical-intelligence/libero/norm_stats.json`, the
config-owned fallback used by the historical stock loader. The frozen runtime
now carries this file. Native K10, H10/action32 storage, 7-D wire action, 8-D
state, paired agentview/wrist RGB256; CP1 pools each 256-token view to 32768.

**Proposed LIBERO topology:** 4090 R5/MPS/B1, startup width 3, whole-server
budget 47 GiB, timan107 x24, EGL0..7/N4. Inherit NUMA-node-1 affinity with
`taskset -c 22-43,66-87` on the launcher (replicas and proxy inherit it).
R5/MPS/B1 and that CPU placement come from the measured MT50 topology above;
LIBERO peak VRAM and throughput remain unmeasured. Use a fresh matching pure
baseline. `--official-50` plans all 2000 episodes; selecting spatial/10 keeps 1000.

B500 libraries are image-free by default. The cache study uses all eight groups,
C5/C10 and native K10 MISS; reduced K1..5 remains opt-in. Commands, including
capture, CPU finish and six-size preparation:
[HANDBACK_LIBERO_VLA.md](../handbacks/HANDBACK_LIBERO_VLA.md).

## GR00T x LIBERO

**Formal selector:** `groot_libero`. Default checkpoint root is `/data/ckpt`
on local, `/home/exouser/ckpt` on H100, containing `n15_libero_spatial` and
`n15_libero_10`. Each uses its own `experiment_cfg/metadata.json` entry
`new_embodiment` (slot 31), native LIBERO transforms and gripper inversion.
The shared adapter selects the suite's backend from the canonical task name.
Library campaigns load only their suite; baselines/cache residents load both.

**Proposed topology:** H100 R2/MPS/B1, startup width 2, whole-server budget 48 GiB,
timan107 x32, EGL0..7/N4. Two paired residents hold four checkpoint copies, the
same weight-copy count as the historical H100 four-endpoint layout; throughput
and peak VRAM of this formal path remain unmeasured. Keep baseline/cache
placement identical. This selector evaluates spatial/10 only: the canonical
four-suite official-50 table is retained, with 1000 dispatched episodes.

Native K4, H16/action32 storage, 7-D wire action, 8-D state, two RGB256 views;
CP1 uses the historical mask-driven spatial16 builder. Cache groups use C5/C10
and native K4 MISS. Endpoint `live_k` follows the study YAML at startup, so
explicit reduced K1..4 works with the same fixed-K endpoint rule as MT50.
Both VLA models share ports23160..23169, including the local driver23169;
serialize their commands when using that tuple even with separate server GPUs.
See [HANDBACK_LIBERO_VLA.md](../handbacks/HANDBACK_LIBERO_VLA.md).

## DP x LIBERO

**Recommended: local4090 R2/B8/5 ms, MPS on, group-commit, timan107 x32.**
Launched baseline: `/data/mw_finetune/eval/hs5dp_ctl_142812`. This replaces
R1/B16/W24 and the later hs4 R1/B16/W32 control. Template for a fresh short
run name; the current baseline has already been launched:

```bash
bash exp/r17_integration/models/dp/formal_eval.sh \
  --model dp_libero --checkpoint /data/mw_finetune/dp/libero_policy_100000 \
  --server-node local --gpu cuda:0 --replicas 2 --replica-startup-width 2 \
  --memory-budget-gib 40 --max-batch 8 --batch-wait-ms 5 --mps \
  --driver-mode group-commit \
  --worker-node timan107 --extra-workers 0 --workers 32 \
  --worker-memory-mib 2048 --worker-vram-mib 576 --worker-max-episodes 4 \
  --worker-restart-limit 2 --port 23180 --driver-port 23189 --official-50 \
  --seconds 43200 --episode-seconds 1800 --retries 2 \
  --run-name hs5dp_ctl_HHMMSS --run
```

The omitted EGL list matches hs5's sealed automatic allowlist, currently0..7.
Its arms must preserve that null value, R2/B8/5 ms/MPS and all worker fields;
use `--runtime-from /data/mw_finetune/eval/hs5dp_ctl_142812` and
`--driver-mode group-commit`. DP rejects `--paired-control`; the coordinator
must check the closed baseline and replica/worker resource equality explicitly.

- **GPU alone:** R2/B8/5 ms/MPS is the measured throughput choice on4090;
  group-commit removes the demonstrated driver stalls. Keep32 workers as the
  selected study treatment, not an assumed GPU-alone hardware maximum. With
  Fast-WAM absent, capacity can change, but more workers have no demonstrated
  speedup and changing them requires a new baseline. H100 native batching is
  wired through `ops/formal_eval.sh`; no H100 DP throughput qualification is
  supplied by these4090 results. The40 GiB campaign budget is an admission
  allowance, not40 GiB observed model usage.
- **Sharing:** use the current two-GPU plan: DP on4090, Fast-WAM onH100.
  timan107's **48 Fast x768 +32 DP x576 MiB** renderer reservations fill
  the allocatable slots. Preserve DP32/RAM2048/VRAM576/N4 and spread0..7.
  A campaign sharing the actual4090 must pass combined peak admission; no
  measured rate here certifies that additional co-resident load. Do not
  retune the current study to fit a new GPU/worker-node neighbor.
- **Batch/drift:** one worker per stage, streams on, frozen CPU B1 language
  memo and private request RNG. Denoiser CUDA graph signatures are captured
  before serving; the **whole100-step sampler host loop is not captured**.
  Qualified batches through32 matched private draws for352 requests;
  largest measured committed physical-action deviation0.000345707.
  MPS on/off eagerB1 is byte-identical in P3's recorded-frame pairs;
  batching/composition still changes arithmetic and needs matched SR controls.
- **Measured server replay:** P3's32-client4090 R2/B8/MPS = **26.163 calls/s**,
  p50 **977 ms**, **209.9 W**; sameR2/B8 off22.659 calls/s (observed+15.5%,
  first clean pair+14.9%; the second on pair gained an external CUDA context).
  R1/B16 off14.381 calls/s is a different topology/background comparison,
  not a controlled1.82x campaign speedup. R1/B16 and Fast-WAMR1/B8 have no
  demonstrated MPS benefit. These replays exclude simulator/driver costs.
  The new hs5 baseline's completed episode rate is **not yet measured**.
- **Verified previous live workload:** S's hs4 R1/B16/W32 window14:13:22..
  14:19:34 CDT yielded **23.39 ep/min,14.73 calls/s**, fillp50/p903/16,
  mean5.50;4090 mean40.9% NVML busy/**164 W**. Worker slots spent49.0% waiting
  for policy/controls,32.8% waiting for driver, approximately10.5% simulating.
  Stage3 host batches were in flight approximately93% of wall time while
  policy requests were resident98.26%: this is **host/sampler bound**, with
  driver durability another major loss, not a shortage of workers.
- **Next lever:** qualify a CUDA graph for the **whole DDPM100 sampler loop**,
  including scheduler updates/launch bookkeeping and request-private RNG.
  S estimates **1.15..1.35x episode throughput**, not a measured result.
  Existing denoiser graphs are already on; merely enabling them again or
  increasing batch caps will not fix this host loop. New graphs must pass
  numerical/identity/pairing qualification before another study uses them.

Sources: [P3 MPS results](../REQUESTS/P3_to_R.md),
[MPS replay data](../models/fastwam/measurements/mps_results.json),
[S targeted breakdown](../REQUESTS/S_to_R.md#s-targeted-hs4-dp-workerserver-wait-breakdown-2026-10-10-1421-cdt),
[aligned S analysis](/data/mw_finetune/perf/2026-10-10/dp_wait_probe_window.json),
[DP batching](../models/dp/BATCHING.md),
[hs5 config](/data/mw_finetune/eval/hs5dp_ctl_142812/campaign.json),
[hs5 orchestration](/data/mw_finetune/eval/hs5dp_ctl_142812/ORCHESTRATION.json).

## Fast-WAM x LIBERO

**Recommended: H100 R2/B16/5 ms, MPS on, group-commit, timan107 x48.**
Launched baseline: `/data/mw_finetune/eval/hs6fw_ctl_143332`. This replaces
R2/B8/W32 and the stopped hs4 R2/B8/W48 plan. Template for a fresh short
run name; the current baseline has already been launched:

```bash
bash exp/r17_integration/ops/formal_eval.sh \
  --model fastwam_libero --checkpoint /data/mw_finetune/models/fastwam_libero \
  --server-node h100 --gpu cuda:0 --replicas 2 --replica-startup-width 2 \
  --memory-budget-gib 70 --max-batch 16 --batch-wait-ms 5 --mps \
  --driver-mode group-commit \
  --worker-node timan107 --extra-workers 0 --workers 48 \
  --worker-memory-mib 2048 --worker-vram-mib 768 --worker-max-episodes 4 \
  --worker-restart-limit 2 --worker-egl-devices 0,1,2,3,4,5,6,7 \
  --port 23173 --driver-port 23177 --official-50 \
  --seconds 43200 --episode-seconds 3600 --retries 2 \
  --run-name hs6fw_ctl_HHMMSS --run
```

Every arm keeps the explicit EGL list, R2/B16/5 ms/MPS, group-commit and
48-worker resources, with `--paired-control /data/mw_finetune/eval/hs6fw_ctl_143332`
and `--runtime-from` pointing to that same baseline. MPS must be passed explicitly.
Wait for closed PASS official-2000 before dispatching arms; oldB1/B8 controls
cannot qualify the new batch/code/MPS treatment.

- **GPU alone:** use the measured H100R2/B16/5 ms/MPS choice above,70 GiB
  total admission. P3 measured52.0 GiB allocated/60.2 GiB reserved acrossR2.
  **R3 does not fit**: minimum weights/workspace plus contexts/headroom need
  approximately85.3 GiB; conservative B16 reserved-memory admission asks
  approximately102.5 GiB. NoR3 rate was measured. Local48 GB4090's alternative
  remainsR1/B8/5 ms/MPSoff,36 GiB admission, measured local replay only;
  two approximately26 GiB residents do not fit there. H100's newB16 result
  does not qualify localB16 as a faster production lane.
  **Superseded for the local knob DEV lane (2026-10-12):** R1/B8/MPS-off with 22 local renderers on the 4090 ran 18.5 episodes/min
  (one GIL-bound process at ~144% CPU, GPU 17–47%, stage2 always full at B8). Caching the 20 LIBERO prompt encodings and
  moving T5 to the CPU makes a replica 13.5 GB, so R2 fits; with R2/B16/MPS, renderers moved to timan107/timan1 and the
  two server processes pinned to two cores each on NUMA node 1, the lane runs **51–66 episodes/min, GPU 31–46%**. Replica 0
  ran ~3× slower than replica 1, first from CPU downclocking, then from other jobs on its pinned cores; with the
  `performance` governor and whole-node pinning both match (~109 episodes/min on mixed cells). Test lanes keep
  the H100 configuration without the text cache, so they match their pure-inference controls.
- **Sharing:** retain Fast-WAM onH100 and DP on4090, sharing timan107 with
  Fast48/2048/768/N4 plus DP32/2048/576/N4. Both spread over0..7.
  A second campaign on the actualH100 needs admission alongside the70 GiB
  Fast allocation; the previous CosmosR2/20 GiB budget cannot simply be added.
  For a **new** smaller GPU-shared study, H100R1/B24/5 ms/MPSoff is the
  measured single-replica fallback (19.2 calls/s,34.13 GiB reserved; plan
  40 GiB admission), with a newly admitted/frozen worker plan. That replay
  was GPU-exclusive and is not a measured shared-GPU episode rate. Current
  hs6 arms cannot reduce replicas/workers or turn MPS off to accommodate it.
- **Batch/drift:** max16/5 ms; stage1/2/3 one thread each, stage streams on,
  native VAE state protected and request-private seeds. Clean H100 sweep
  used48 clients/three real observations/K10/C5; full-chunk normalized max
  drift **0.015625**, median0, p950.00390625, first5 committed-action max
  **0.0078125**. This stays within prior measured drift, not byte parity or
  an SR certificate; K1/K2 outcome/numerical qualification is not implied.
- **Measured server capacity:** **28.0 calls/s**, p50/p90 **1679/1779 ms**,
  actual S3 fillp50/p90 **16/16**, **412.6 W (~413)**, NVML busy75.7%.
  SameR2/B16/5 ms without MPS:24.8 calls/s,1858 ms p50,374.4 W.
  SingleR1/B16/5 ms:17.6 calls/s; R1/B24/5 ms:19.2; R1/B32/5 ms:17.6.
  **16.8 calls/s is R1/B16/10 ms**, not the5 ms comparator. Rates use20 s
  after priming, excluding warmup/drain, and are not campaign episodes/min.
  MPS improves same-topology throughput12.9%; independent host pipelines
  also improveR2 off overR1, contrary to the earlier blanket no-replica-gain
  statement. The new hs6 baseline's completed episode rate is not yet measured.
- **Bottleneck:** host launch/action-head work and actual batch supply after
  the group-commit correction. Power/fill/completed decisions are more useful
  than NVML busy for judging saturation: oldR2/B8 showed~99% busy/~200 W;
  the faster MPS replay has lower busy but greater throughput/power. Exact
  SM occupancy or kernel-only saturation was not measured. There is no
  measured reason to add workers or increase beyondB16 in this treatment.

Sources: [exclusive P3 sweep](../REQUESTS/P3_to_R.md),
[capacity/numerical qualification](../models/fastwam/BATCHING.md#exclusive-h100-capacity-sweep--2026-10-1014161424-cdt),
[raw capacity results](../models/fastwam/measurements/h100_capacity_results.json),
[hs6 config](/data/mw_finetune/eval/hs6fw_ctl_143332/campaign.json),
[hs6 orchestration](/data/mw_finetune/eval/hs6fw_ctl_143332/ORCHESTRATION.json),
[coordinator launch/old-util evidence](../../../logs/session_handoff.md).
H100 ledgers remain in the run's `servers/replica_{0,1}/{batching,serving}.jsonl`
onH100 until transfer; use S's local artifacts to avoid extra tether polling.

## Cosmos x LIBERO

**Fresh K5 baseline template** (independent of the retained-cell resume):

```bash
bash exp/r17_integration/ops/formal_eval.sh \
  --model cosmos_libero --checkpoint /data/mw_finetune/models/cosmos_libero \
  --server-node h100 --gpu cuda:0 --replicas 1 --memory-budget-gib 10 \
  --max-batch 1 --batch-wait-ms 5 \
  --worker-node timan107 --extra-workers 0 --workers 12 \
  --worker-memory-mib 6144 --worker-vram-mib 1024 --worker-max-episodes 4 \
  --worker-egl-devices 0,1,2,3,4,5,6,7 \
  --port 23240 --driver-port 23249 --official-50 \
  --seconds 43200 --episode-seconds 1800 --retries 2 \
  --run-name RUN_NAME --run
```

- **GPU alone:** fresh capacity starting point H100R1/10 GiB,12 workers
  after admission. Historical R2/18..20 GiB is a recorded study topology,
  not a demonstrated replica throughput gain; existing paired arms retainR2.
  Do not add residents without an overlap/throughput qualification. One resident
  is about6022 MiB ready /8462 MiB sampled H100 peak. Local4090 supports1
  (10 GiB), with a completed local smoke and byte-exact K2 parity. Owner's Cosmos
  study stays onH100; no extra local Cosmos baseline is queued.
- **Sharing:** H1001/10 GiB and8 workers/6144 MiB is a smaller proposed
  allocation. R2/20 GiB alongside Fast-WAM R2/64 GiB exceeds the device's
  conservative budgets despite lower momentary usage; use admitted smaller
  lanes or wait for a natural campaign boundary. Select shared workers from
  current per-card admission before starting a new baseline. The8-worker
  continuation subsequently ran and closed; it is no longer a proposed fleet.
- **Batch/drift:** B1 only, window inactive, mutable resident state serialized;
  no StageBatcher. Official/R17 native parity on recorded frames was byte
  exact. RequestedK2 **includes the clean-step budget and executes actual
  NFE1**, not two network evaluations. The defaultK5 executes NFE5; preserve
  C16. The current study requests onlyK2, noK1/K3 arms.
- **Measured rate:** clean formal400 H100 R2/timan108x24 =17.62 ep/min,
  2.930 calls/s,1362.33 driver s. Completed K2 spatial+LIBERO-10 H100R2 /
  timan107x24 =37.00 ep/min,7.352 calls/s,1621.54 s. That run used older
  4096 MiB/N8 treatment and was GPU/node shared; it is not the rate of the
  conservative command above. No isolated campaign GPU util/W receipt was
  found; Fast-WAM profile's mixed-board W cannot be attributed to Cosmos.
- **Bottleneck:** native K5 GPU work for the full baseline, then worker
  rendering/recycling/arrival supply atK2. Previously GPU0 renderer saturation
  caused driver growth and D-state stalls; distributing workers is mandatory.
- **Pitfalls:** retain the launcher-managed NVRTC CUDA_HOME/library paths and
  gymnasium runtime. The failed official2000 source accepted1726 cells; its
 274-cell continuation is D's identity-pinned `--resume-from` path, using
 23243/44/45+23248 and an explicit approved unreachable-timan108 closure
  waiver. Its archival source/treatment proof cannot be replaced with this
  fresh template. The completed K2 run's pairing must still be audited against
  the eventual closed baseline; transport/topology changed across recovery.

Sources: [Cosmos handback](../handbacks/HANDBACK_COSMOS.md),
[K5 history](/data/mw_finetune/eval/formal_cosmos_libero_formal400_20261009_231322_2b680ed5/SUMMARY.json),
[completed K2](/data/mw_finetune/eval/hs2_cosmos_h100_k2incl_c16_t107_20261010_115120/SUMMARY.json).

## MetaWorld MT50 measured topology (2026-10-13)

Seeds: test = seed 7, idx 0-49 (2,500 episodes, the official MT50 count); library = seed 8, idx 0-49. Seeds 9 and
10 are reserved. Workers: timan107 x40 + timan1 x36 (GPU 3 only) = 76, `--worker-memory-mib 2048
--worker-vram-mib 1024 --worker-egl-devices 0,1,2,3,4,5,6,7`. Servers: weilandserver 4090 (49 GB) and H100 (80 GB).
All rows below ran 2,500 episodes with 0 errored episodes. "span" runs from the first to the 2,500th accepted
episode (the H100 shard drain afterwards is not included). Util/W are medians of 1-minute nvidia-smi samples.

| run | model | server | topology | span | ep/min | calls/s | GPU util / W | bottleneck |
|---|---|---|---|---|---|---|---|---|
| `cpi_r5m` library | π0.5 | 4090 | R5 MPS B1 | 152 min | 16 | 6.2 | 46% / 211 | host: ~0.85 s/call per replica, one core each; R5 is the VRAM cap |
| `cgr_r6m` library | GR00T | 4090 | R6 MPS B1 | 68 min | 37 | 11.8 | 24% / 157 | capture: ~0.48 s/call vs ~0.15 s pure |
| `cdp4_r4b8` library | DP | H100 | R4 MPS B8 | 26 min | 95 | 31.9 | 97% / 317 | GPU |
| `cfw_r2b16` library | Fast-WAM | H100 | R2 MPS B16 | 45 min | 56 | 16.1 | 44% / 265 | per-replica host/stage pipeline (~8 calls/s each) |
| `tgr7_0502` test | GR00T | 4090 | R6 MPS B1 | 22 min | 113 | 34.6 | 70% / 314 | near the 4090's ~33 MB/s public ingress (0.69 MB/request) |
| `tpi7_0536` test | π0.5 | 4090 | R5 MPS B1 | 96 min | 26 | 9.1 | 71% / 283 | host (pin replicas + proxy to NUMA node 1) |
| `tdp7_0729` test | DP | 4090 | R4 MPS B8 | 19 min | 131 | 48.5 | 100% / 410 | GPU |
| `tfw7_0755` test | Fast-WAM | 4090 | R1 B16 | ~90 min | ~28 | ~8 | ~40% | one replica; R2 does not fit (27.8 GB/replica) |
| `csdp_1504` cache study | DP (DDIM10 MISS) | H100 | R4 MPS B8, timan107×30 + timan1×20 | 120,000 eps, 4.5 h | ~450–550 | — | ~20% / 135 | server host loop (4 replicas ~97% CPU each); more replicas would have helped |
| `cspi_k2` cache study | π0.5 (K2 MISS) | 4090 | R5 MPS B1, timan107×24 + timan1×20 | 60,000 eps | ~110–130, ~175 after DP ended | — | 40–55% / 240 | weilandserver public ingress (33 MB/s measured) |
| `csfw_k3` cache study | Fast-WAM (K3 MISS) | H100 | R3 MPS B16, timan107×30 + timan1×20 | 120,000 eps | ~410 (first cells) | — | 25–40% / 225; 76 GB | per-replica host pipeline (~85% CPU each) |
| `csgr_h5` cache study | GR00T (K2 MISS) | H100 | R8 MPS B1, timan107×40 + timan1×20 | 120,000 eps | ~280–370 | — | 50–65% / 230; 53 GB | host: each replica one full core (H100 host has 20 cores); ~27 GB VRAM free, so R11 fits for the next GR00T H100 campaign; the resume `csgr_h6` with timan1×40 only ran the same ~200–400 ep/min, so 40 workers suffice |
| `grmk1_ful` MT50 knob (full K4) | GR00T | H100 | R11 MPS B1, timan1×40, driver 23149 | 90,000 eps | ~340–430 | 10–18 | 45–75% / 230; 71 GB | host per replica; R11 vs R8 not compared on the same plan |
| `pimk1_ful` MT50 knob (full K10) | π0.5 | 4090 | R4 MPS B1, **local ×40** (server and MetaWorld workers on weilandserver, `fbe96912`), replicas+proxy NUMA 1, workers NUMA 0 | 90,000 eps | 75 (15-min window) | — | 40–50% / 200–235; 37–38 GB | host per replica (72–88% CPU each); a local MW worker holds 133 MiB VRAM, ~0.5 GB RAM, ~1.5% CPU; R5 would leave ~3 GB VRAM |
| `pimk1_fulb` MT50 knob (full K10), A | π0.5 | 4090 | R3 **B8 stage batching** W5, local ×96 | resume | 82–88 (k40 cells) | 35.9 req/s, h 0.63 | SM mean 81% (p10 50, min 23); 43 GB | server; 96 local EGL contexts share the GPU |
| `pimk1_fulc4` MT50 knob, B | π0.5 | 4090 | R3 B8, remote timan107 ×40 only | resume | 110 (k10) | 33.4 req/s, h 0.80 | SM mean 57%; ingress 23 MB/s | workers: remote round trip p50 244 ms (0.69 MB/request through the 33 MB/s ingress) |
| **`pimk1_fuld` MT50 knob, C (production)** | π0.5 | 4090 | **R4 B8 W5, `--wire-zlib`, timan107 ×80 (8 GPUs) + local ×32**, budget 40 GiB | resume | **131–176 (k10)** | **65.5 req/s, h 0.83 (98% of the sweep's R4 B8 prediction)** | SM mean 87%, p50 99%; 42 GB, 429 W; ingress 14 MB/s | server |

Full-step test success (seed 7, 2,500 episodes), reported as the equal-weight mean of the four difficulty groups (easy 28 / medium 11 / hard 6 / very hard 5 tasks; the convention of the RLinf and SmolVLA MT50 tables) and, second, the per-task mean: GR00T 67.1 / 75.8, DP 46.0 / 61.2, π0.5 46.7 / 56.3. Published π0.5 SFT (same RLinf checkpoint) is 43.8 group mean. Always state which mean a MetaWorld number is.

Next time: DP R4 B8 MPS saturates both GPUs, keep it. Fast-WAM on H100 leaves the GPU half idle at R2; try R3 B16
(3 x 24.4 GB fits in 80 GB) before adding workers. π0.5 and GR00T on the 4090 need host CPU, not GPU: pin to NUMA
node 1 with `pin_local.sh`, which keeps pinning while later replica waves load. A GR00T test lane beyond ~35 calls/s
needs request compression first. Cosmos is LIBERO-only by owner decision (2026-10-09); MetaWorld uses the four models above.

Library collection command (one model; legacy driver kept on NVMe, copied back after finish):

```bash
bash openpi_ext/scratch/r17_ops/mw/collect_one.v2.run.sh /data/r17_frozen/openpi_<sha> dp_mw h100 cdp4_r4b8 \
  --replicas 4 --mps --memory-budget-gib 24 --max-batch 8 --batch-wait-ms 5 \
  --worker-counts timan107=40,timan1=36 --port 23180 --driver-port 23189
```

Test baseline: `formal_eval --model <m> --server-node local --replicas R --replica-startup-width 3 --mps --max-batch B --driver-mode group-commit
--worker-counts timan107=40,timan1=36 --prepare-only`, then `check_direct.py` and `launch_direct_root.sh`; see
`openpi_ext/scratch/r17_ops/mw/run_base2.sh`. Always pass the startup width: the default for π0.5/GR00T is 1, which loads R5 π0.5 one replica at a time (~2.3 min each, ~12 min) instead of ~5 min at width 3; GR00T R6 at width 3 is ready in ~1.5 min.

## π0.5 x MetaWorld MT50

The formal launcher now accepts native K1/K2/K10. This MT50 checkpoint predicts
H5, so its maximum commit is C5; C10 needs a different checkpoint geometry.

**Measured-topology baseline template:**

```bash
bash exp/r17_integration/ops/formal_eval.sh \
  --model pi05_mw --checkpoint /data/ckpt/pi05_metaworld_rlinf \
  --server-node local --gpu cuda:0 --replicas 1 --memory-budget-gib 20 \
  --max-batch 1 --batch-wait-ms 5 \
  --worker-node timan107 --extra-workers 0 --workers 12 \
  --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-max-episodes 0 \
  --worker-egl-devices 0,1,2,3,4,5,6,7 \
  --port 23160 --driver-port 23169 \
  --seconds 43200 --episode-seconds 1800 --retries 2 \
  --run-name RUN_NAME --run
```

- **GPU alone:** localR1/x12 is evidenced by the earlier complete MT50 runner,
  not a completed formal_eval run. H100R1/20 GiB is a capacity option requiring
  that GPU's qualification. Extra replicas without MPS time-slice compute;
  no measured scaling result supports recommendingR2/R3. Memory allocations
  are planning budgets, not measured MT50 resident peaks.
- **Sharing:** local orH1001/x12,20 GiB,2048/1024/N0 and spread0..7.
  If another LIBERO fleet uses the worker node, account for its larger growth
  reservation; do not add another x24 by assuming all workers cost2 GiB.
- **Batch/drift:** formalB1, inactive window. G1 same-noise native parity passed;
  compiled-vs-eager differences previously failed the probe, so retain the
  qualified eager/native path. R17 LIBERO batch knobs do not enable MT50 batches.
- **Measured rate:** completed `pi05_rlinf_mt50_egl`, localR1/timan107x12:
  500/6504.16 s =4.612 ep/min;10744 recorded decisions =1.652 calls/s.
  GPU util/W not recorded in this receipt. This historical lane shared the
 4090 schedule; no isolated-GPU scaling certificate.
- **Bottleneck:** fullK10 inference/serialized resident and arrival queue;
  relative GPU-vs-frontend shares are unmeasured. Diagnose service/driver waits
  before proposing concurrency changes. Worker rendering stays remote.

Sources: [Q qualification handback](../qualification/HANDBACK_Q.md),
[G1 native PASS](/data/mw_finetune/qualification/runs/slot_G1_pi05_mw_A_20261009_191908_601423/receipt.json),
[completed invocation](/data/mw_finetune/eval/pi05_rlinf_mt50_egl/invocations.jsonl),
[decision counts](/data/mw_finetune/eval/pi05_rlinf_mt50_egl/episodes.jsonl).

## GR00T x MetaWorld MT50

The formal launcher now accepts K1/K2/K4 by loading an endpoint with the selected
immutable K, and C5/C10/C15 at K4. Cache YAMLs can hot-switch at the resident's K;
changing GR00T K requires a new resident.

**Baseline template matching the historical R1/x12 topology:**

```bash
bash exp/r17_integration/ops/formal_eval.sh \
  --model groot_mw --checkpoint /data/mw_finetune/groot/4090_mt50/checkpoint-60000 \
  --server-node local --gpu cuda:0 --replicas 1 --memory-budget-gib 20 \
  --max-batch 1 --batch-wait-ms 5 \
  --worker-node timan107 --extra-workers 0 --workers 12 \
  --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-max-episodes 0 \
  --worker-egl-devices 0,1,2,3,4,5,6,7 \
  --port 23160 --driver-port 23169 \
  --seconds 43200 --episode-seconds 1800 --retries 2 \
  --run-name RUN_NAME --run
```

- **GPU alone:** localR1/x12/20 GiB is the measured historical topology.
  H100R1/20 GiB is a capacity option requiring that GPU's qualification and
  installed GR00T island. The two-resident G2 mechanical PASS establishes
  execution, not extra compute or episode throughput. Do not recommendR2
  from that receipt or use LIBERO per-suite counts for this one checkpoint.
- **Sharing:** R1/x12,20 GiB,2048/1024/N0, spread0..7. Prefer reducing the
  second campaign's duplicate residents before taking worker memory below its
  reserve. Preserve the topology between its baseline and arms.
- **Batch/drift:** B1 only. The actual G2 C/B2 probe failed: reported maximum
 0.0029 exceeds its1e-3 gate. Implemented S3 hooks/CPU fixtures do not override
  that production restriction; S1/S2 remain unsupported. Full samplerK4.
- **Measured rate:** complete localR1/timan107x12 legacy MT50 run:
  500/1570.625 s =19.10 ep/min;9176 decisions =5.842 calls/s.
  Util/W unmeasured. Mechanical two-replica qualification is not a throughput
  or SR scaling certificate.
- **Bottleneck:** B1 resident/frontend and worker supply; the LIBERO R16
  preprocessing profile suggests a CPU hypothesis, not measured MT50 shares.
  Local simulators once consumed enough4090 VRAM to OOM GR00T; keep workers
  remote. Immutable final60000, not mutable checkpoint/latest paths.

Sources: [G2 replicas PASS](/data/mw_finetune/qualification/runs/slot_G2_groot_mw_R_replicas_20261009_235142_2713961/receipt.json),
[B2 failure](/data/mw_finetune/qualification/runs/slot_G2_groot_mw_C_B2_20261009_205753_1211464/receipt.json),
[complete invocation](/data/mw_finetune/eval/groot_mt50_60k_egl/invocations.jsonl).

## DP x MetaWorld MT50

MT50 now accepts B2..32, private `--mps`, native DDPM100 or trailing eta0
DDIM10/DDIM5, and C5/C10/C15. C10 uses two five-action transport calls; the
second is a committed tail with no model compute. Default B1 behavior is
unchanged. See [package B handback](../handbacks/HANDBACK_MW_B.md) for CPU evidence
and parameterized coordinator smoke commands; GPU throughput remains unmeasured.

**Fresh single-resident capacity template** (episode throughput unmeasured):

```bash
bash exp/r17_integration/ops/formal_eval.sh \
  --model dp_mw --checkpoint /data/mw_finetune/dp/mt50_policy_100000 \
  --server-node local --gpu cuda:0 --replicas 1 --memory-budget-gib 4 \
  --max-batch 1 --batch-wait-ms 5 \
  --worker-node timan107 --extra-workers 0 --workers 24 \
  --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-max-episodes 0 \
  --worker-egl-devices 0,1,2,3,4,5,6,7 \
  --port 23180 --driver-port 23189 \
  --seconds 43200 --episode-seconds 1800 --retries 2 \
  --run-name RUN_NAME --run
```

- **GPU alone:** localR1/x24/4 GiB is a capacity starting point, not a
  measured faster replacement. The completed measurement wasR12/x24.
  H100R1/4 GiB has no solo MT50 episode-rate certificate here. SpecifyR1
  explicitly: default12 is historical. LIBERO's R2/B8 MPS result does not
  qualify MT50B1/MPS; R4/R2 have no measured throughput advantage here.
- **Sharing:** localR1/x12/4 GiB,2048/1024/N0; avoid placing a12-process
  DP pool beside Fast-WAM onH100. Session history reports that arrangement
  reduced Fast-WAM from about10 to4 ep/min. Spare VRAM is not spare compute.
- **Batch/drift:** B1 remains the historical topology; MT50 B2..32 now uses
  native DDPM100 CUDA graphs and the existing graph/eager self-check. The
  batching window is active for B>1. GPU arithmetic drift and throughput still
  need MT50 measurements; LIBERO numbers do not establish those results.
- **Measured rate:** clean formal500 localR12/timan107x24:
  500/1941.295 s =15.45 ep/min,10655/1941.295 =5.489 calls/s.
  All12 resident VRAM allocations totaled10678 MiB (~888 MiB each).
  Util/W unmeasured. No linear R12→R4 rate estimate is asserted.
- **Bottleneck:** many small DDPM kernels and cross-process scheduling atR12;
  exact compute/launch/worker shares unmeasured. Runtime graph prewarm and
  source self-check must pass before dispatch; preserve canonical MT50 action
  dimensions and normalization, not the retired loader fixtures.

Sources: [completed campaign](/data/mw_finetune/eval/formal_dp_mw_mt50_500_20261009_225821_ef25fe4e/campaign.json),
[summary](/data/mw_finetune/eval/formal_dp_mw_mt50_500_20261009_225821_ef25fe4e/SUMMARY.json),
[GPU allocation](/data/mw_finetune/eval/formal_dp_mw_mt50_500_20261009_225821_ef25fe4e/GPU_MEMORY.json),
[Q handback](../qualification/HANDBACK_Q.md).

## Fast-WAM x MetaWorld MT50

The one-view MT50 geometry now uses the same native stage batching at B2..32,
with optional private MPS, K1/K2/K10, and C5..C30 in multiples of five. C10
retains five-action calls and reuses the second committed tail without compute.
MT50 cache studies register YAMLs on the resident and select them per episode;
see [package B handback](../handbacks/HANDBACK_MW_B.md) for smoke commands.

**Fresh single-resident capacity template on the usable worker node:**

```bash
bash exp/r17_integration/ops/formal_eval.sh \
  --model fastwam_mw \
  --checkpoint /data/mw_finetune/fastwam/mt50_frozen_b128/policy_016001 \
  --server-node h100 --gpu cuda:0 --replicas 1 --memory-budget-gib 36 \
  --max-batch 1 --batch-wait-ms 5 \
  --worker-node timan107 --extra-workers 0 --workers 24 \
  --worker-memory-mib 2048 --worker-vram-mib 1024 --worker-max-episodes 0 \
  --worker-egl-devices 0,1,2,3,4,5,6,7 \
  --port 23173 --driver-port 23177 \
  --seconds 43200 --episode-seconds 1800 --retries 2 \
  --run-name RUN_NAME --run
```

- **GPU alone:** H100R1/36 GiB,24 workers is an initial capacity plan, not
  a measured episode-rate optimum. The complete historical R2 campaign used
  timan108; it does not establish a throughput gain from two time-sliced
  replicas. Moving to timan107 requires new admission/measurement. Local4090
  fits1 (36 GiB), proposed12..16 workers;2 is unavailable at approximately
 26 GB/resident. Closed H100 replica0 peak allocated24295 MiB/reserved24666
  MiB; do not extrapolate that one receipt to all current working sets.
- **Sharing:** H1001/36 GiB,12 workers/2048 MiB plus1024 VRAM/N0. Prefer
  keeping other small-kernel campaigns on the other GPU. Local1 can coexist
  only after sum-of-peak admission, not a momentary free-memory observation.
- **Batch/drift:** B1 is the historical setting; B2..32 now batches the one-view
  MT50 delta geometry with an active window. P3's LIBERO B8 measurements do not
  cover this geometry. Production MW GPU drift and throughput remain unmeasured.
- **Measured rate:** clean formal500 H100R2/timan108x24:
  500/3457.273 s =8.677 ep/min,10514/3457.273 =3.041 calls/s.
  OtherH100 residents shared the historical schedule; no standalone util/W
  receipt or local MT50 rate. P3's local LIBERO5.98 calls/s is not this model
  x benchmark measurement.
- **Bottleneck:** K10 head/small-batch inference and cross-process scheduling;
  workers/node RAM can constrain sharing. Use the frozen policy_016001
  content seal; source/header-only and mutable trainer exports are not valid
  replacements. Keep the geometry/model/norm identity joins on close.

Sources: [completed summary](/data/mw_finetune/eval/formal_fastwam_mw_mt50_500_20261009_223917_0b845f00/SUMMARY.json),
[resident allocation](/data/mw_finetune/eval/formal_fastwam_mw_mt50_500_20261009_223917_0b845f00/servers/replica_0/closed.json),
[P3 handback](../handbacks/HANDBACK_P3.md), [P6 handback](../handbacks/HANDBACK_P6.md).

## Measurement evidence and grouped driver

**P3/S/Q recommendations are resolved.** Server capacity replays, live campaign
windows and CPU driver replays have different scopes; preserve those scopes
rather than projecting replay calls/s into campaign episodes/min.

### P3 server capacity selection

Exclusive H100 sweep:48 closed-loop clients, three real LIBERO frames, K10/C5,
50 ms think; each rate uses20 s after3 s priming. S3 fill percentiles count
batch events. Warmup/drain are excluded. These are short point estimates.

| H100 Fast-WAM config | Calls/s | p50 ms | S3 fill p50/p90 | Mean W | Peak reserved GiB |
|---|---:|---:|---|---:|---:|
| R1/B16/5 ms, MPS off | 17.60 | 2767 | 16/16 | 293.8 | 31.83 |
| R1/B24/5 ms, MPS off | 19.20 | 2425 | 24/24 | 312.0 | 34.13 |
| R1/B32/5 ms, MPS off | 17.60 | 2586 | 16/32 | 300.1 | 38.54 |
| R2/B16/5 ms, MPS off | 24.80 | 1858 | 16/16 | 374.4 | 59.55 |
| **R2/B16/5 ms, MPS on — selected** | **28.00** | **1679** | **16/16** | **412.6** | **60.16** |

The often-repeated16.8 calls/s comparator is **R1/B16/10 ms**. R3 was refused
before launch (~85.3 GiB minimum including context/headroom); it has no rate.
Full normalized max drift0.015625, first5 committed-action max0.0078125.
Evidence: [P3 handback](../REQUESTS/P3_to_R.md),
[H100 results](../models/fastwam/measurements/h100_capacity_results.json),
[qualification scopes](../models/fastwam/BATCHING.md).

Local4090 DP32-client replay selected **R2/B8/5 ms/MPS on,26.163 calls/s**,
977 ms p50,209.9 W. Same topology off22.659,1138 ms,192.5 W: observed+15.5%,
with first clean pair+14.9% and a changed external context in the second on
pair. R1/B16 off14.381 calls/s/1747 ms is a different topology/background;
it is not proof of1.82x episode throughput. R1/B16's MPS result and Fast-WAM's
localR1/B8 result show no demonstrated gain. Evidence:
[P3 MPS table/caveats](../REQUESTS/P3_to_R.md),
[local replay](../models/fastwam/measurements/mps_results.json).

### S's DP host-loop and worker-wait diagnosis

Aligned372 s interval **14:13:22..14:19:34 CDT**, stopped hs4 R1/B16/W32
before the new MPS/group-commit baseline:145 accepted episodes,23.39 ep/min,
14.73 calls/s,996 matched stage3 batches. This is historical live evidence,
not the rate of hs5 R2/B8.

| Measured component | Value | Interpretation |
|---|---|---|
| Worker policy/control waits | 49.0% of slot wall | inference/controls dominate |
| Worker driver waits | 32.8% | blocked dispatch/result processing |
| Simulation/rendering/images | ~10.5% | reconciled estimate; not the main limiter |
| Startup/recycle; setup/other | ~3.2%;~4.5% | estimates, not stack-derived exclusive time |
| Stage3 host batch in flight | ~93% | Python launches/scheduler/GPU waits, not kernel duty |
| Requests resident | 98.26% | extra workers mainly add queueing |
| GPU NVML busy /board power | 40.9% /164 W | below GPU saturation despite host activity |
| Actual S3 fill | p50/p90 3/16, mean5.50 | cap16;88% under cap |
| Client call = server +protocol residual | 1039 =867 +172 ms | matched completed episodes |
| Reconstructed stage3 queue wait | p50/p90 ~256/503 ms | estimate, not directly instrumented |
| Result-durability delay /next-dispatch p50 | 13.65 /13.70 s | old per-row-fsync driver |

DDPM100 still runs100 Python sampler iterations outside the denoiser graphs.
S's whole-loop graph estimate1.15..1.35x is future qualification work, not a
measured gain. In the earlier14:00..14:10 short-episode window,64.75% of cycle
wall was outside the runner and throughput17.6 ep/min/9.39 calls/s; this
explains why one idle GPU spot overstated the later driver share.

Current per-step queue/service fields remain null/unsupported. S reconstructed
queue from batch completion/member host spans and the5 ms assembly window;
receipt/thread skew prevents exact compute-exclusive timing. Under-cap batches
remain expiry proxies without enqueue/flush reasons. Do not relabel those
estimates as instrumented queue times or the assembly window as total wait.

Evidence: [S targeted report](../REQUESTS/S_to_R.md),
[aligned analysis](/data/mw_finetune/perf/2026-10-10/dp_wait_probe_window.json),
[earlier interval](/data/mw_finetune/perf/2026-10-10/dp_wait_1400_1410.json),
[raw recorder](/data/mw_finetune/perf/2026-10-10/perf.jsonl).

### Q's group-commit driver and durability

**Use `--driver-mode group-commit` for both current LIBERO lanes.** Q replayed
24 concurrent fake workers/32 episodes/24 step rows each on real `/data`,
with real JSON appends/fsync, no injected delay and no GPU work. Legacy
handlers came verbatim from the DP control's sealed runtime. Input/task and
terminal outcome hashes match; all CPU workers closed.

| Real-/data CPU replay | Legacy | Group-commit |
|---|---:|---:|
| Dispatch/pull p50 | **667.029 ms** | **0.1282 ms** |
| Reporting handler p50 | **27,628.697 ms (27.6 s)** | **0.0870 ms** |
| Report→next assignment p50 | 11,558.672 ms | 0.2161 ms |
|32 episodes with final durable drain | 59.982 s | 1.563 s |
|fsync count | 864 | 13 including namespace barriers |

Handlers enqueue bounded evidence with **no filesystem I/O under the scheduler
lock**. The writer groups at1 s or64 events. **Supporting step/attempt/dispatch
files and namespace entries must successfully fsync, then terminal journal must
fsync, then scheduler acceptance/counting occurs.** Fast handler return is not
an already-durable episode; compare the separate final drain/counting evidence.
Failed commits, writer overflow or an unclosed drain fail completion. Retry and
attempt-generation fences stay intact. Uncommitted assignments/diagnostics may
be lost on crash; recover with `--resume-from` into a new segment after source
closure/isolation, never restart the same driver in place. Grouped resumed
segments preserve SHA-bound durable raw evidence before merging/aggregation.

The driver has its own immutable **orchestration pin**. Only
`formal_driver_fast.py`/`group_commit.py` overlay the frozen runtime; native
model/server/worker/benchmark bytes remain identical. `ORCHESTRATION.json`,
receipt and summary record both baseline and selected driver identities.
**Arms may use group-commit against an older-driver baseline via `--runtime-from`**;
do not reject solely for the separate driver change. Batch/replica/worker/MPS
pins remain fixed. Disclose differing driver modes when attributing wall-time
or service cost; CPU replay does not measure end-to-end GPU campaign speedup.

For an unlaunched sealed queue, Q's supported CPU refresh is
`formal_queue.sh --prepared-queue PREPARED_QUEUE --prepare-only --refresh-driver group-commit`.
It cannot refresh a launched run or change the model/worker treatment. Current
hs5/hs6 already select group-commit and need no refresh or duplicate launch.
Library/capture/parity-only campaigns cannot use grouped mode; other templates
retain their historical/default legacy driver unless separately selected.

Evidence: [Q grouped-driver handback](../qualification/HANDBACK_Q.md#grouped-driver-orchestration),
[real-/data replay](/data/mw_finetune/qualification/driver_replay_20261010_group_commit_final/RESULTS.json),
[CPU durability regressions](../qualification/preflight/driver_group_commit_20261010/cpu_resume_final.xml),
[driver pin](driver_release.py), [queue instructions](FORMAL_QUEUE.md),
[hs5 pin](/data/mw_finetune/eval/hs5dp_ctl_142812/ORCHESTRATION.json),
[hs6 pin](/data/mw_finetune/eval/hs6fw_ctl_143332/ORCHESTRATION.json).

## Pitfalls (do not repeat)

### Replicas need measured host/kernel overlap

Replica processes share compute and duplicate weights/workspace, but **the
blanket claim that R2 cannot help without MPS is wrong**: P3's H100R2/B16/5 ms
off24.8 calls/s beatsR1/B16 off17.6 by overlapping independent host pipelines.
MPS then raises the sameR2 topology to28.0. DP's oldR12/B1 ~1409 ms/call does
not justify twelve replicas or a12/latency capacity claim. Choose the measured
replica/batch/MPS combination; do not extrapolate linearly. FastR3 does not fit.

Evidence: [exclusive H100 results](../models/fastwam/measurements/h100_capacity_results.json),
[P3 interpretation](../REQUESTS/P3_to_R.md),
[old DP comparator](../models/dp/BATCHING.md).

### NVML busy percentage is not compute saturation

Old H100R2/B8 had95..99% NVML busy at~200 W of700 W. The fasterR2/B16/MPS
replay has75.7% busy,413 W and full16-row batches. DP had93% host-stage activity
but only41% GPU busy/164 W. Judge completed calls/episodes, **power draw and
actual fill** together; busy is time with work, not SM occupancy or percentage
of peak compute. Power/fill are useful signals, not a substitute for an exact
kernel/occupancy measurement. Keep board occupants/time scopes with readings.

Evidence: [old coordinator spot](../../../logs/session_handoff.md),
[P3 capacity results](../models/fastwam/measurements/h100_capacity_results.json),
[S host/GPU accounting](../REQUESTS/S_to_R.md).

### Measure actual batch fill before changing caps

P3's selected H100R2/B16/MPS replay fills16/16 atp50/p90; R1/B32/5 ms
alternates16/32 and is slower thanR1/B24. S's old DP cap16 hadp50/p903/16
with a nearly continuously busy host loop. A larger cap cannot create arrival
continuity or remove host/driver waits. Measure per-stage/replica buckets and
hold the chosen cap/window fixed within a study. Short-window replay is not SR.

Evidence: [P3 fill/cap sweep](../REQUESTS/P3_to_R.md),
[S live fill](../REQUESTS/S_to_R.md),
[aligned histograms](/data/mw_finetune/perf/2026-10-10/dp_wait_probe_window.json).

### Use grouped durability instead of per-row fsync under the lock

The old driver held the scheduler lock across per-row evidence fsync, blocking
both reporting and worker pulls. Use the qualified group-commit mode and its
**evidence fsync → terminal fsync → acceptance** rule. Fast dispatch does not
permit early counting or unsafe in-place restart. Record the separate driver
pin; it permits an orchestration upgrade without changing model/worker pins.

Evidence: [Q durability/replay](../qualification/HANDBACK_Q.md#grouped-driver-orchestration),
[replay results](/data/mw_finetune/qualification/driver_replay_20261010_group_commit_final/RESULTS.json),
[S old-driver waits](../REQUESTS/S_to_R.md), [orchestration implementation](driver_release.py).

### Spread EGL rendering across GPUs and recycle workers

All workers rendering on GPU0 exhausted its 8 GiB while GPU1..7 sat empty, and the
host memory leaked until the node OOMed. Spread workers over all eligible GPUs
(`--worker-egl-devices`) and recycle every 4 episodes (`--worker-max-episodes 4`).
There are no RAM/VRAM/disk admission gates any more (owner 2026-10-10); size
fleets from measured node capacity (see the cache-study topology section), not
from reservation arithmetic.

Evidence: [GPU0 overload](../REQUESTS/D_to_M.md),
[D spreading/timan108 receipt](../handbacks/HANDBACK_COSMOS.md),
[M recycling bound](../REQUESTS/M_to_Q.md),
[coordinator admission failures](../../../logs/session_handoff.md),
[hs5 worker fields](/data/mw_finetune/eval/hs5dp_ctl_142812/campaign.json),
[hs6 worker fields](/data/mw_finetune/eval/hs6fw_ctl_143332/campaign.json).

### Decide baseline restart before arms and freeze its runtime

Choose batch/replicas/workers/MPS before collecting a baseline. Any decision to
replace/restart that control with a different throughput treatment must happen
**before arms dispatch**; regenerate the queue against its new closed PASS.
Changing an arm to rescue a slow baseline violates pairing. Fast-WAM enforces
resource topology, including explicit versus null EGL; DP's coordinator must
also compare worker/replica fields because its runtime verifier is narrower.
Use `--runtime-from`; Fast-WAM also needs `--paired-control`. Keep source model/
worker bytes and native identity; a working-tree reseal once changed the
ModelManifest merely by adding an MPS helper. hs3/hs4 queues are obsolete.
A separate group-commit orchestration upgrade is allowed and does not authorize
batch/replica/worker/MPS changes. Crash recovery still needs a fresh segment.

Evidence: [Q frozen-runtime rule](../qualification/HANDBACK_Q.md#frozen-paired-study-runtime),
[identity diff](../qualification/preflight/frozen_runtime_20261010/IDENTITY_DIFF.json),
[runtime field checks](formal_runtime.py),
[driver overlay exception](driver_release.py),
[new controls/restarted queues](../../../logs/session_handoff.md).

### Match MPS to the new measured baseline

hs6 FastR2/B16 and hs5 DPR2/B8 both useMPS; their arms must explicitly pass
`--mps`. Earlier off-baseline arms cannot turn it on. LocalDPR1/B16 and
FastR1/B8 show no demonstrated benefit; the H100R2 result now supersedes the
old compatibility-only/MPS-off advice. Fresh code/runtime preparation and a
matched baseline were required for this changed treatment. Replay arithmetic
and throughput do not prove K1/K2 SR invariance.

Evidence: [P3 H100/local MPS results](../REQUESTS/P3_to_R.md),
[H100 capacity data](../models/fastwam/measurements/h100_capacity_results.json),
[hs5 treatment](/data/mw_finetune/eval/hs5dp_ctl_142812/campaign.json),
[hs6 treatment](/data/mw_finetune/eval/hs6fw_ctl_143332/campaign.json).

### Two server processes on one GPU need one shared MPS server

Without MPS, two Fast-WAM replicas on the 4090 time-slice their CUDA contexts: stage1 went from ~120 ms to ~6 s and the lane
ran slower than one replica (`knobdevfwred3`). Any other non-MPS context on that GPU (another campaign's server, EGL
renderers) time-slices the same way, so do not co-locate a second campaign on a GPU that runs an MPS lane.

### Host CPU frequency limits host-bound servers

weilandserver is a Xeon E5-2696 v4. Under the old `schedutil` governor, server threads spread over many lightly loaded
cores ran at 1.2 GHz; the 4090 Fast-WAM lane went from 27–30 to 51–66 episodes/min once each server process was pinned
to two cores, and replica 0 ran ~3× slower than replica 1 the whole time. On 2026-10-12 the owner set the governor to
`performance` (persistent); cores now run at 2.7–3.7 GHz and the two replicas match. Pin server processes to the **whole** GPU NUMA node (node 1: CPUs 22–43, 66–87) with `taskset -a -p -c 22-43,66-87 <pid>`
(lost on restart), not to a few cores each: other sessions' jobs (go tests, codex) run on this box unpinned, and on
2026-10-12 two of them sat on replica 0's ten cores, which made its CPU-side steps 2× slower and its model calls 3–8× slower.
With the whole node both replicas match (hit 68 ms, call 1.2 s) and the lane ran 109 episodes/min. If a lane looks host-bound again, check `scaling_governor` and core MHz first.

### A weilandserver server with remote workers is capped by the public ingress (~33 MB/s)

Every request uploads two 256×256 images (~330 KB). On 2026-10-12 the 4090 Fast-WAM lane (workers on timan107/timan1,
entering through ziyanglin.com:23130) reached ~80 requests/s = ~26 MB/s on that port and 34 MB/s on the NIC, i.e. the
~33 MB/s ingress, while the servers still had headroom (stage1 batch ~1.6, GPU ~50%). So a local server fed by remote
workers tops out near 100 requests/s whatever its replicas do. Lanes whose workers sit on the timan nodes should prefer
a server those workers reach without this link (H100 has its own public IP), or keep local workers on 127.0.0.1 (but do
not put EGL renderers on a GPU that runs an MPS lane). Measure with `/proc/net/dev` on `enp8s0` and per-port
`ss -tin` `bytes_received` deltas.

### Watch per-stage latency right after launching a changed configuration

Three changed launches on 2026-10-12 failed within minutes (no MPS, text-cache keys missing the prompt template so every
model call raised KeyError, CUDA OOM). Run `r17_ops/knob/lane_speed.sh <RUN> 60` (episodes/min, per-replica stage batch
rate × mean size, hit/call latency, GPU, CPU) as soon as the replicas are ready, and grep the replica logs for errors.

### An overloaded Fast-WAM replica can fall into a 4× slower state

In `knobtestfw_red` (2 replicas, MPS, B16, 70 remote workers) the first ~15 minutes of the plan were call-heavy (~17 model
calls/s in total). Replica 0 then ran stage2 batches of 12–16 in a median 2.3 s, while replica 1 ran the same sizes in
0.63 s (the K1 control and `knobdevfwred8` ran them in 0.35–0.73 s on both replicas). Replica 0's calls took 5–10 s
against 1.2–1.5 s. Its stage threads sat mostly in futex waits while about one connection thread was always runnable,
which points to GIL contention. The cause is not confirmed: ptrace is restricted on this host, so py-spy cannot attach.
The lane ran 47 episodes/min. When the call load fell to ~9 calls/s (18:00), both replicas matched again (stage2 batch
~2.6, 0.33 s apart) and the lane ran ~83 episodes/min. CPU placement, NUMA memory and per-client MPS memory limits were
all checked and identical. To detect it, compare per-replica stage2 batch intervals at equal batch size: read the `time`
gaps between `r17.fastwam.stage_batch.v1` rows in `servers/replica_N/batching.jsonl`. If it recurs and lasts, first try
fewer workers per replica, then a smaller `sys.setswitchinterval` in the server.

2026-10-14 `csfw_k3w5` (H100, R3 B16 W5, `--wire-zlib`, timan1 + timan107 + timan108 x40 = 40 connections per replica)
against `csfw_k3w4` (same server, timan1 x40), whole-minute rates from `serving.jsonl` at hit share ~0.83: 62 -> 75-80
req/s (+25% for 3x the workers), mean service time 0.52 -> 0.95 s, calls 2.3-3.0 s -> 3.9-5.4 s, SM 44% -> 34%
(per-second dmon), replica CPU ~150-200%. The supply sweep predicted ~236 req/s at this mix: it has no connection
threads, so on Fast-WAM it overstates the live ceiling about 3x; the ceiling is host-side in the replica process.
Never attach py-spy or any ptrace tool to a live replica: `sudo py-spy record --nonblocking -p <replica>` hung for over
two minutes and stalled replica 0; its in-flight episodes exhausted their retries ("replica backend unavailable: timed
out"), 217 errored episodes in `csfw_k3w5` (15:42-15:45). Profile on a copy of the replica outside a campaign.

### H100 library capture staging can stall and then burn the plan as errors

MT50 library campaigns with an H100 server stage each episode's capture on H100 and pull closed shards back in
batches. The original policy reserved 256 MiB per active episode under a 20 GB limit (about 66 concurrent episodes)
and pulled at most 8 shards per batch (~45 s each, so ~10 shards/min). With 76 workers at ~45 episodes/min
(`cdp_r4b8`, 2026-10-13) the closed backlog grew to ~900 shards, reservations stopped fitting, episodes failed
mid-run, and library retries refused them (`FatalEpisodeError: library retries require zero steps and zero
decisions`): 912 errored episodes in a few minutes while the throughput probe still showed ~140 accepted/min.
Fixed in `0f1138aa`: no shard-count cap per batch (the 511 MiB byte cap remains) and 64 MiB slots for image-free
captures. Count errored episodes, not accepted ones (`r17_ops/mw/mw_speed2.sh`, `r17_ops/mw/err_watch.sh`); journal
`status: failed` with an empty `error` is an ordinary unsuccessful episode and is valid library data.

### Per-decision capture quota writes serialize every H100 replica

The H100 capture quota is one JSON file under one flock. Each decision called `budget.check()` twice and
`budget.charge()` once, and each call rewrote the JSON with two fsyncs on H100 `/data`. With 76 workers the
DP MT50 lane (`cdp2_r4b8`, R4 B8 MPS) ran ~6 calls/s with batch size 1.0, 12 s median request time and the GPU idle
between requests; the three model stages took under 0.5 s. Fixed in `35b042cc`: image-free shards skip the
per-decision check/charge and charge the written bytes once before `closed()` (which rejects a shard larger than its
charges; `3710d483` skipped the charges without this and every episode failed its final acknowledgment). If a capture lane is slow,
compare `elapsed_seconds` with the stage timestamps in `servers/replica_*/serving.jsonl` before adding replicas.

### Shard batch installs on the /data HDD can take hours

Each DP/Fast-WAM MT50 shard holds 30-70 small files. Without the 8-shard cap one DP batch carried 1,603 shards and
75,135 files, and `install_snapshot` looked every member up by name and fsynced every file on `/data`: about 6 files/s,
so the DP drain after `cdp4_r4b8` finished collecting (03:00) needed roughly 4-5 h and Fast-WAM's about 7 h. The GPU is
free during the drain, so start the next GPU job as soon as all episodes are accepted (cutover v5 does this) instead of
waiting for the driver copy-back. Fixed for new campaigns in `9cf5cfdc` (stream members, sync once: 3,000 files in
3.6 s on `/data`); launchers that are already running keep the old function in memory.

### H100 batched campaigns need a local copy of their frozen runtime

Finalize audits batched Fast-WAM evidence (and batched cache-study evidence) on weilandserver with the campaign's own
frozen runtime under `/data/mw_finetune/qualification/formal_runtime_<identity>`. A new H100 campaign installed that
release only on H100, so `cfw_r2b16` collected and transferred all 2,500 episodes and then failed with
`finalization_error: frozen native audit runtime missing`. Prestage now also installs the release locally for every
H100 campaign. A campaign that already failed this way is recoverable without the GPU: install the campaign's
`runtime.tar` (same sha as `RUNTIME_RELEASE.json`) at that root, then re-run `formal_finalize.finalize` on the closed
receipt with `LAUNCH_WORKER_DEPLOYMENT.json`, keeping the old receipt as `RECEIPT_BEFORE_RECOVERY.json`.

### A library only serves on the env it was collected with

The model spec (`checkpoint_identity`, `code_digest`) includes the dependency versions of the server env, and
`validate_teacher` used to refuse a resident whose spec differs from the library's teacher (gate since removed). The MetaWorld DP env on H100 is
a Python 3.12 venv whose `dp_readonly_base.pth` points at the Fast-WAM env (torch 2.7.1+cu128, diffusers 0.39.0,
transformers 5.5.4, numpy 2.2.6). On weilandserver it pointed at the openpi `.venv` (diffusers 0.37.1, transformers
4.53.2, numpy 1.26.4), so the DP MT50 library collected on H100 failed the 4090 hot-switch smoke prepare with
`recipe library model/norm/layout/features/protocol/code mismatch`. The weights were byte-identical. The local env was rebuilt the
H100 way (old one kept as `/data/mw_finetune/dp/env.openpi_py311`). Before a test baseline or cache run on another
host, compare its prepared `assets/model_spec.json` with the library teacher's (`bpool/READY.json` `binding.model`).

### MT50 cache studies: plan time, GR00T K, and preparing ahead

- One MT50 cache campaign holds up to 48 cells x 2,500 = 120,000 episodes. Computing its plan takes ~140 s and ~600 MB;
  `driver_release.planned_tasks` used a 120 s limit (`csdp_1451` failed), now 1800 s (`8bc82127`).
- GR00T's endpoint K is fixed when the resident starts (`live_k`, default 4). A study whose YAMLs use K2 failed at
  `load_cache_config` on every replica (`csgr_1608`); formal_eval now starts GR00T at the study YAMLs' K (`c758a6c9`).
  π0.5, DP and Fast-WAM take K per YAML.
- Preparing a cache campaign takes ~6.5 min (asset push plus a plan) and launching another ~3 min before replicas load.
  Prepare the next campaign of a lane while the current one runs (`mw/prep_cs.sh` + `mw/go_cs.sh`), otherwise the GPU
  idles ~10 min per switch (4090, 16:07-16:17 on 2026-10-13).
- An MT50 request carries one 480x480 image (~0.69 MB), so a 4090 lane fed by timan workers through ziyanglin.com tops
  out near 48 requests/s (`enp8s0` measured 33 MB/s during `cspi_k2`, π0.5 R5 at ~120 episodes/min). Cache hits make
  calls cheaper but not fewer: GR00T was already at that cap in its pure test, so its 120k-episode study would take
  ~13 h on the 4090. Run remote-worker MT50 cache studies of light models (GR00T, DP, Fast-WAM) on H100 (own public
  IP; DP took ~127 MB/s there at ~500 episodes/min) and give the 4090 the host-bound one (π0.5). When a study runs on
  another GPU type than its pure baseline, rerun that baseline on the same GPU.
- `direct_transfer` serializes every rsync on one host-wide lock. An H100 prepare pulls its `assets.tar` (~1-1.4 GB)
  back through the weilandserver ingress; while a 4090 lane with remote workers fills that ingress the pull runs at
  ~1 MB/s and holds the lock for 20+ min, so another campaign's launch waits on it (H100 idle 19:35-19:43 on
  2026-10-13, `csfw_k3` behind `csgr_h3`). Prepare an H100 campaign only after the current H100 lane is collecting,
  never across another lane's switch (`mw/wait_then_prep_gr.sh`).

### No identity-fingerprint gate between a library and its resident

`validate_teacher` used to refuse a resident whose model spec (code hash, env-derived checkpoint identity) differed
from the library teacher. Per the owner rule (no fingerprint gates), it now accepts the library; a model-owned
`library_validator` hook still runs where one exists. So widening a K list in model code (MT50 π0.5 K1-K5/K10,
GR00T K1-K4, Fast-WAM K1-K5/K10) or serving from another host's env no longer orphans a library. Libraries hold
full-step teacher data, which the serving K does not change.

### Never kill a chain's tmux session while it holds a launcher

A chain script runs `launch_direct_root.sh` in the foreground, so its tmux session is the launcher's parent. Killing
that session (2026-10-13 08:51, `mw_4090f`) sent SIGHUP to `formal_eval --run --prepared-run` of `tfw7_0755`; the
server, driver and remote workers survived, but nothing was left to close them or seal the receipt. Before killing a
session, list its children (`pgrep -P $(tmux list-panes -t S -F '#{pane_pid}') -a`); stop only a waiting chain by its
PID. A launcher-less pure-inference run can be finished by waiting for the driver to exit, closing that run's server
and worker supervisors by PID, and `formal_eval aggregate --output RUN_DIR` (see `openpi_ext/scratch/r17_ops/mw/finish_tfw7.sh`). The replica proxy's command line does not contain the run name (`run_proxy(PORT,[...])`): find it by its public port (`ss -ltnp | grep :PORT`) and stop it too, or the next campaign on that port fails with `ports did not release within 120s` (`gl_a2`, 2026-10-14). To stop a waiting chain, kill only its bash PID; never close the tmux session of a chain whose launcher is still running (`gl_k1a` lost its launcher that way).
Killing the chain's bash is the same as closing the session when that bash is the pane's own process (`tmux new -d -s X "bash chain.sh"`): the pane closes and SIGHUPs the launcher (`pimk1_ful`, 2026-10-14 12:07). To stop a running campaign, SIGTERM its launcher first and only then the chain; new chains start launchers with `setsid bash launch_direct_root.sh ...` (`mwlocal/chain_4090m.sh`), which keeps them out of the pane's session.

### Keep MPS pipe paths within 85 bytes

The UTF-8/filesystem-encoded **absolute pipe directory**, not just the run name,
must be<=85 bytes. Current layout is
`/data/mw_finetune/eval/<run>/mps/<6-character GPU hash>/p`.
Use short names such as `hs6fw_ctl_HHMMSS`, `hs5dp_ctl_HHMMSS` or `hs5dp_c10`.
Short names do not remove one-shot uniqueness/seal requirements; use a fresh
name for each preparation. Do not use long descriptive campaign names or
attempt to share another campaign's MPS pipe/daemon.

Evidence: [P3 path constraint](../REQUESTS/P3_to_R.md),
[encoded-byte enforcement](../models/fastwam/ops/mps.py),
[hs5 startup lesson](../../../logs/session_handoff.md).

### Wait for port release after campaign closure

After a stopped/closed campaign, ports can remain busy for about a minute;
DP restart hit **Address already in use**. A new run name alone does not release
a port. Wait for public/backends/driver to become free, or select fresh ports
inside the model's reserved window and use a fresh run name. Keep all distinct
and respect the coordinator's other lanes. Do not immediately relaunch the
same tuple or treat a broker TCP listener as resident readiness.

Evidence: [coordinator restart failure](../../../logs/session_handoff.md),
[port checks/readiness](formal_eval.py),
[hs5 exact port tuple](/data/mw_finetune/eval/hs5dp_ctl_142812/campaign.json).

### Run background queue scripts from an immutable copy

Q edited `queue_hs4_dp.sh` while its coordinator instance was running; shell
scripts may read later bytes during execution. Finish/review the script, then
run a separate **`*.run.sh` snapshot** that agents must not edit. Freeze its
referenced argv JSON too before queue preparation; sealed prepared directories
then supply the immutable runtime/settings. Do not overwrite a running copy.
Current coordinator snapshots are `queue_hs5_dp.run.sh` and
`queue_hs6_fw.run.sh`; their existing instances must not be duplicated.

Evidence: [coordinator script-edit incident](../../../logs/session_handoff.md),
[DP running copy](/home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs5_dp.run.sh),
[Fast running copy](/home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs6_fw.run.sh).

### Respect model ports, node exclusions, and tether budget

Cosmos23240..23249, Fast-WAM23170..23179, DP23180..23189. A broader parser
or old receipt does not authorize taking another model's lane. MetaWorld on
timan1 is authorized on physical GPU3 only, max40 workers; timan108 remains
offline andGPU1/2 belong to another user. After
recovery use eligible0,3 and fresh admission. Keep tether probes>=30 s apart;
prefer S's existing evidence and batch read-only calls.

Evidence: owner allocation rule;
[Cosmos transport constraints](../handbacks/HANDBACK_COSMOS.md),
[renderer exclusions](worker_gpu.py),
[recorder/probe budget](../REQUESTS/S_to_R.md).

### One failed broker dial must not abort a campaign

At 22:12 on 2026-10-13 one launcher probe `tether exec --safe --timeout 10s h100 -- env` failed with exit 69
`cannot reach broker ... i/o timeout`; the next tether call 13 s later worked. `tether_retry.transient` did not
classify exit 69 as transient, so `csfw_k3` (Fast-WAM MT50 cache study, 47,500 of 120,000 episodes after 2.5 h) was
aborted and its replicas stopped. Fixed in `d9c2c672` (exit 69 + "cannot reach broker" is retried: nothing reached the
node). Launchers already running keep the old code. When replicas die under a running driver, the group-commit
journal records the in-flight episodes as `accepted: true` with an `error` (1,734 in `csfw_k3`'s `ch n1000`): count
only rows without `error` (`openpi_ext/scratch/r17_ops/mw/cells.py` does) and resume the rest.

### Resuming a large cache campaign

`cache_resume.prepare(source, name)` re-reads the source's whole `per_step.jsonl` (5.3 GB for 47,500 MT50 Fast-WAM
episodes: ~20+ min single-threaded, ~27 GB RAM). Call `prepare` once; do not call `select` first (it reads the same
evidence again). Prepare the next campaign of the lane in parallel so the GPU is not idle meanwhile. GR00T MT50 on
H100 also needed `7db095a3` (`load_benchmark_spec` imported the LIBERO bridge; `csgr_h4` died at replica start).

### Chain waits must match the tagged line, not a phrase

Queue scripts wait on `grep -q PHRASE chain.log`. At 23:36 on 2026-10-13 the H100 v6 chain waited for
"h100-v5 chain done", found that phrase inside a coordinator note describing the queue, and launched the Fast-WAM
resume on top of the running GR00T campaign (stopped by PID 25 s later, before any replica loaded). Grep the full
tagged line (`"\[h100-v5\] h100-v5 chain done"`) and never write a launch trigger phrase into a coordinator note.

### H100 root disk fills with campaign staging

H100 `/` is only 58 GB and every H100 campaign stages 0.6–2.7 GB under `/tmp/r17_formal/<run>` (startup assets,
server campaign, replica logs) that nothing removes. On 2026-10-13 23:51 `/` reached 100% (137 MB free) while
`csgr_h5` was serving; a prepare failed with `No space left on device`. Closed campaigns' staging dirs were moved
(not deleted) to `/data/r17_formal_moved/` on H100 (`/` back to 61%). Check `df -h /` on H100 before a queue of
H100 prepares and move closed runs' `/tmp/r17_formal/<run>` to `/data/r17_formal_moved/`. Never move a run that is prepared but not yet launched: its launch reads `/tmp/r17_formal/<run>/server/campaign.json` and the staged assets.

### Size a lane from measured supply and demand, then fill the ingress before adding local workers

Supply: `exp/r17_integration/profile/supply_sweep.sh` measures X(R,B) per phase (hit/call) with R replicas under one
MPS on an exclusive GPU (4090 π0.5 R1-4 x B1/4/8/16 in `openpi_ext/scratch/r17_ops/mwlocal/sweep_4090_1209`: mixed
throughput at hit share 0.82 R1 7->39, R3 21->65, R4 26->69 req/s; past B8 the GPU saturates). Demand:
`mwlocal/demand.py RUN` gives worker think time and the hit/call mix from serving.jsonl (local MT50 worker: 19 ms,
i.e. ~52 req/s per worker at zero server latency). Calls are only the call share of requests, so a call batch of B
per replica needs about B / call-share requests in flight per replica (Little's law). On the 4090: remote workers cost
ingress (0.69 MB raw, 0.25 MB with `--wire-zlib`, lossless) and add ~150-250 ms per request; local workers cost VRAM
(133 MiB each) and GPU time slices for EGL (96 local workers kept the SM at 81% with large dips). Fill the ingress with
remote workers first, then add local workers until the server is saturated: R4 B8 + zlib + timan107 x80 + local x32
reached 98% of the sweep's prediction. Worker nodes have 1 GbE (timan1 sent 122 MB/s to H100 with the GR00T lane, 98%
of the link, while the H100 SM averaged 55%), so H100 lanes need several worker nodes and `--wire-zlib` as well.

### Worker-node GPUs: shared cards, timan108 EGL numbering, write races

- Another user's job on a worker card used to refuse EGL admission outright (`tdp7h1_full`, 2026-10-14 14:04: yurenh2
  started two 868 MiB jobs on timan1 GPU3). Sharing a card with free VRAM is allowed; since `d2019b8c` foreign processes
  are only recorded.
- timan108 (rebooted 2026-10-14): EGL device k = kernel minor k, which is the reverse of the nvidia-smi index; EGL 0/1
  (index 3/2, the idle cards) fail to initialize, EGL 2/3 (index 1/0) render but carry other users' memory. The worker
  inventory checks `nvidia-smi -i <minor>` against the minor and rejects every card there. Not used until fixed.
- Three replicas starting together raced on `servers/manifest.json.part` (`pimk1_fulc` replica_0 died); `write_json`
  uses a per-process temp name since `c15a78dc`.
- H100 supply (14:03 sweep, `mtk/h100_sweep_report.txt`): DP K100 calls are GPU-bound at ~131/s (R8 B32, 95%; R12 +5%);
  DP K10 supplies >1,000 calls/s even at R4 B16, far above what 40 workers can send; Fast-WAM K10 R3 B16 ~205 req/s at
  hit share 0.82 (B8 ~166), R3 is the VRAM cap.

### Check the worker node's disk before preparing, and watch long MT50 lanes

timan107 went down at ~00:21 on 2026-10-14 (no ping, no ssh, tether agent offline) while it carried an 8 h π0.5 MT50
lane (`cspi_k2`) plus ~64 EGL workers over its eight 8 GB GPUs. Its shared /scratch + /tmp disk was at 97% (~6.9 GB
free of 227 GB) after a dozen prepares that day; MT50 workers run with `--worker-max-episodes 0` (never recycled).
The cause is not established (free RAM was 191 GB at 00:05; timan1 MT50 workers grew ~0.7 MB/min each, 37/503 GB
used). Before a prepare, run `df -h /scratch /tmp` on every worker node and move closed runs' `/tmp/r17_formal/<run>`
and `/scratch/zixuans8/mw_r17/formal/runs/<run>_*` out of the way (move, never delete). After a crash, read
`journalctl -b -1 -k` before restarting anything.

### Local LIBERO workers need VRAM headroom on the 4090

Local LIBERO workers render with EGL on the 4090 and are recycled every 4 episodes; each new generation must allocate
an offscreen framebuffer. `pl_k1a` (π0.5 LIBERO, R5 MPS ~38 GB + 24 local workers) ran the GPU at 48.4/49 GB; after
789 episodes the recycled workers failed with mujoco `Offscreen framebuffer is not complete (0x8cdd)`, exhausted the
worker restart budget and the campaign failed (0 errored episodes). Keep ~8 GB free: π0.5 with local workers uses
R4 + 20 workers (~40 GB). DP R4 MPS + 48 local workers (`fsreddp2`, ~23 GB) and Fast-WAM R1 + 22 (`fsredfw`) were fine.

### Prestage queued campaigns

Use `ops/formal_queue.sh` and immutable preparation while the previous job runs.
The old60..110 s switch figures are **estimates**, not measurements of the new
R2/MPS/group-commit path. Prestaging removes CPU packaging/transfer from the
critical path; warm validation/admission, GPU loading and real first-episode
execution remain. New control/treatment requires corrected queue references;
prepared jobs cannot receive arbitrary flag overrides. The current snapshots
prestage arms from hs5/hs6 and dispatch only after baseline PASS/closure.

Evidence: [queue instructions](FORMAL_QUEUE.md),
[Q startup reconstruction](../qualification/preflight/formal_prestage_20261010/STARTUP_BREAKDOWN.json),
[current queue snapshots](../../../logs/session_handoff.md).

## Refresh and queue procedure

1. Correct this reference immediately when a better configuration or failed
   recommendation is verified: replace model advice, template and final row
   together. Preserve only measured historical comparators with their original
   topology/scope. All P3/S/Q selection inputs for this correction are resolved;
   full campaign episode rates and SR require completed control evidence.
2. Record exact run, GPU/occupants, model/worker runtime, orchestration pins,
   replicas, batch/window/MPS and worker/RAM/VRAM/N4/EGL fields. Closed baselines
   need PASS/`all_owned_closed=true`; a startup PARTIAL receipt is not PASS.
   Use verified completed episodes/driver seconds for historical ep/min and
   policy decisions for calls/s; per-step line counts include duplicate echoes.
3. Prefer S's aligned artifacts. Preserve cutoff, task mix, actual fill and
   driver/report waits; label reconstructed queue estimates. Assess completed
   rate with power/fill, not NVML busy alone. New hs5/hs6 campaign episode rates
   remain unmeasured in this reference; replay rates are explicitly labeled.
4. Read-only probes need PID birth checks and tether spacing>=30 s. This refresh
   launches/stops/reconfigures no campaign or background queue.
5. Review queue argv/resources against the exact new control, then run only a
   frozen `*.run.sh` copy or the sealed prepared queue. Native model/worker
   treatment is frozen; a selected driver overlay is recorded separately.

## Current controls, queued arms, and next tuning

- **Fast-WAM hs6:** R2/B16/5 ms/MPS/group-commit, timan10748,2048/768/N4,
  explicitEGL0..7;70 GiB campaign budget, startup width 2. Preserve the launched
  control; noR3, cap32 or worker expansion is justified by these measurements.
- **DP hs5:** R2/B8/5 ms/MPS/group-commit, timan10732,2048/576/N4, automatic
  EGL0..7;40 GiB budget, startup width 2. Next server-code experiment is a
  qualified whole-sampler CUDA graph, estimated1.15..1.35x. More workers or
  cap16 are not a substitute. Any adoption requires its own matching treatment.
- **Fast arms C10/C16/C20/C32,K2/K1:** pin
  `--paired-control /data/mw_finetune/eval/hs6fw_ctl_143332` and
  `--runtime-from /data/mw_finetune/eval/hs6fw_ctl_143332`, explicitly retain
  `--mps --driver-mode group-commit` and every hs6 resource/batch field.
  The coordinator's reviewed jobs are in
  [queue_hs6_fw.json](/home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs6_fw.json);
  the existing run copy prepares `/data/mw_finetune/eval/fq_hs6fw` and waits
  for closed PASS. Do not restart its already running queue.
- **DP arms C10/C15/C16,DDIM10/DDIM5:** pin
  `--runtime-from /data/mw_finetune/eval/hs5dp_ctl_142812`, explicitly retain
  `--mps --driver-mode group-commit` and every hs5 resource/batch field.
  [queue_hs5_dp.run.sh](/home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs5_dp.run.sh)
  is the existing snapshot and baseline wait. Old hs4 preparations are not
  substitutes. Both studies plan spatial/10 original-init cells, one horizon
  or denoising override per arm, and require completed closed controls.
- **Future older-driver baselines:** group-commit is allowed through Q's separate
  preparation/refresh pin while frozen native treatment remains identical.
  Keep orchestration-mode cost attribution explicit. Decide a control restart
  for batch/replica/worker/MPS changes before arms, never midway through them.

## Per-model launch summary

MPS/driver describe the recommended template or named current control. Rates
are labeled with their measured topology; a replay is not an episode-rate
certificate. `legacy` is the default formal driver; R16 uses its own conductor.
For historical rows no isolated GPU power/occupancy measurement is implied.
All worker placements below use timan107 for new campaigns; timan108 is offline.

| Model x benchmark | GPU | Replicas | Batch /window | MPS | Driver | Workers | Measured rate | Principal bottleneck |
|---|---|---|---|---|---|---|---|---|
| π0.5 LIBERO |4090 |proposed R5/startup3; NUMA1 |B1 |on |group-commit |t10724, EGL0..7/N4 |LIBERO rate/peak VRAM unmeasured; topology from MT50 |host stages/retrieval |
| GR00T LIBERO |H100 |proposed R2/startup2; two suite backends/replica |B1 |on |group-commit |t10732, EGL0..7/N4 |formal LIBERO rate/peak VRAM unmeasured |host stages/retrieval |
| **DP LIBERO — hs5** |**4090/48 GB** |**2** |**B8/5 ms** |**on** |**group-commit** |**t10732;2048/576 MiB,N4;automatic EGL0..7** |**26.16 calls/s replay**,977 ms p50,209.9 W; hs5 ep/min unmeasured |DDPM100 host loop; old-driver stalls addressed |
| **Fast-WAM LIBERO — hs6** |**H100/80 GB** |**2** |**B16/5 ms** |**on** |**group-commit** |**t10748;2048/768 MiB,N4;explicit EGL0..7** |**28.0 calls/s replay**,1679 ms p50,413 W; hs6 ep/min unmeasured |host/action-head launches and fill; compute occupancy unmeasured |
| Cosmos LIBERO |H100;4090R1 qualified option |R2 history/pairing;freshR1 capacity start |B1/inactive |off |legacy |newt10712;historicalt10824 K5,t10724 K2 |17.62 ep/min/2.930 calls/s K5;37.00/7.352 K2 atR2, historical |serial queue/K5 compute; renderer/dispatch |
| π0.5 MT50 |4090;H100 requires qualification |R1 |B1/inactive |off |legacy |t10712 |4.61 ep/min/1.652 calls/s,R1/W12 historical |fullK10 inference; service/driver shares unmeasured |
| GR00T MT50 |4090;H100 requires qualification |R1;R2 mechanical PASS only |B1/inactive |off |legacy |t10712 |19.10 ep/min/5.842 calls/s,R1/W12 historical |B1 frontend/inference, worker supply |
| DP MT50 |4090;H100 capacity option |fresh/sharedR1;historicalR12 |B1/inactive |off |legacy |t10724 solo/12 shared |15.45 ep/min/5.489 calls/s atR12/W24;R1 unmeasured |small kernels/host/context scheduling |
| Fast-WAM MT50 |H100;4090R1 capacity option |fresh/sharedR1;historicalR2 |B1/inactive |off |legacy |t10724 solo/12 shared;historicalt10824 |8.68 ep/min/3.041 calls/s atR2/W24;R1 unmeasured |K10 head, shared GPU/driver scheduling |

## Cache-study topology in use (2026-10-12)

One resident model per GPU; all groups and library sizes of a model hot-switch by
YAML inside one campaign. Identity/seal checks and RAM/VRAM/disk admission were
removed on 2026-10-10 (owner ruling). Worker counts below are measured capacity.

**Network path (2026-10-12, owner rule: never `tether expose`, no traffic through the VPS broker).**
Until 2026-10-12 every cache and knob campaign exposed its policy and driver ports through
the broker (linziyang.top 14xxx), which exhausted the VPS traffic quota. `open_tunnels()` now
only sets direct endpoints:
- H100 server: `149.165.153.233:<public port>` (23420 Fast-WAM, 23440 Cosmos).
- Driver on weilandserver: `ziyanglin.com:<driver port>`, using **23110 DP, 23111 Fast-WAM, 23112 Cosmos**
  (registered in `gpu_share/LEDGER.md`). Older prepared runs carry 234x9 drivers; resume them with
  `cache_resume.prepare(..., driver_port=2311x)`.
- Local server with local workers only (DP): server and driver on 127.0.0.1.
- Launch through `r17_ops/knob/launch_direct.sh <run> <log>`: `check_direct.py` checks the campaign
  before launch and the worker launch configs after; any broker address or a `TUNNELS.json` stops it.

| Lane | Server GPU | Replicas / batch | MPS / driver | Workers | Per worker | Ports |
|---|---|---|---|---|---|---|
| DP, sizes 50–400 (`csztd_r3_l48`, resumed) | local RTX 4090 48 GB | R4, B8 / 5 ms | on / group-commit | local × 48 | RAM 3584 MiB, renderer ~706 MiB on the same 4090, recycle N4 | 23400–23404, driver 23419 |
| Fast-WAM, sizes 50–400 (`fastwamc6066547406`) | H100 80 GB | R2, B16 / 5 ms | on / group-commit | timan107 × 45 (EGL 0–7) + timan1 × 40 (GPU 3 only) | 2048 / 768 MiB, N4 | 23420–23422, driver 23439 |
| Cosmos, size 500 (`cosmos500w85_220813`) | H100 | R2, B1 | off / legacy | timan107 × 45 + timan1 × 40 | 3072 / 768 MiB, N4 | 23440–23442, driver 23459 |
| Cosmos, sizes 50–400 (`cosmossizes145_r8c`, resumed; `driver/` on NVMe; finished 15:26) | H100 | **R8, B1, memory budget 76 GiB** | **on (private MPS)** / legacy | timan107 × 45 + timan1 × 40 + local × 60 | 3072 / 768; local 4096 / 1024; N4 | 23440, backends 23441–23448, driver 23459 |
| DP per-task knob DEV/TEST (`knobdevdp2`, `knobtestdp*`; reduced `knobdevdpred2`) | local RTX 4090 | R4, B8 / 5 ms | on / group-commit | local × 48 | 2048 / 717, N4 | 23400–23404 on 127.0.0.1, driver 23110 (23419 before 10-12) |
| Fast-WAM per-task knob DEV/TEST (`knobdevfw4`, `knobtestfw3`) | H100 | R2, B16 / 5 ms | on / group-commit | timan107 × 45 + timan1 × 40 | 2048 / 768, N4 | 149.165.153.233:23420–23422, driver ziyanglin.com:23111 (23439 via broker before 10-12) |
| Fast-WAM reduced-step knob DEV on the 4090 (`knobdevfwred6`, 2026-10-12 12:2x) | local RTX 4090 48 GB | **R2, B16 / 5 ms**, T5 prompt cache (`fastwam_text_cache`, T5 on CPU, 13.5 GB per replica), budget 54 GiB, patched runtime `formal_runtime_3faa6dcf_fwtextcache` | on (private MPS) / group-commit | timan107 × 40 + timan1 × 30, **no renderers on the 4090** | 2048 / 717, N4 | public 23130 (backends 23131–23132), driver 23113; server processes pinned with `taskset` to the whole NUMA node 1 (22–43, 66–87); from 13:0x `knobdevfwred8` with timan107 × 30 + timan1 × 40 (timan107 RAM: each Fast-WAM worker ~3.1 GB RSS) |
| Cosmos per-task knob DEV/TEST (`knobdevcos4`) | H100 | R8, B1, budget 76 GiB | on (private MPS) / legacy, `driver/` on NVMe | timan107 × 45 + timan1 × 40 | 3072 / 768, N4 | 149.165.153.233:23440–23448, driver ziyanglin.com:23112 |

Knob campaigns are prepared with `r17_ops/knob/prepare_dev.sh <dp|fastwam|cosmos> <RUN_NAME> [STUDY]` (it sets `R17_INHERIT_WORKER_DEPLOYMENT`; a fresh prepare otherwise fails on timan107 because the legacy LIBERO config under `/scratch/zixuans8/openpi_trace_h100` is gone). A failed launch cannot be relaunched: prepare again under a new name.

**Node capacity.**
- timan107: 45 workers in use; 64 have run before.
- timan1: GPU 3 only, 40 workers, inside the bwrap device namespace; its other GPUs are forbidden.
- Local weilandserver: local workers render (EGL) on the local 4090, about 706 MiB each, on the same card as a local server.
  - With the DP server (4 replicas) on that 4090, the limit is **48 local workers**. On 2026-10-11, 60 workers (54 renderers, 37.5 GB) filled the 48 GB card, and the DP replicas hit CUDA OOM in batched inference; the errored episodes were retried.
  - When the 4090 has no server, 60 local workers fit (about 42 GB of renderers).
  - With the two-replica Fast-WAM server on the 4090, put **no** local renderers there: 22 renderers (~13 GB) plus two B16 replicas (~34 GB) ran out of memory (`knobdevfwred5`).
  - Local workers serve the 4090 lane first, and join an H100 lane only while the 4090 lane is idle. 60 DP plus 60 Cosmos local workers at once would also exhaust the 251 GB of RAM (about 3 GB RSS each).

**Measured.**
- DP: 100–115 episodes/min; the 4090 runs at 100% and about 26 GB.
- Fast-WAM: the H100 sits at about 23% and 51–59 GB, so the lane is bound by workers and latency, not by the GPU. Per-group 500-episode spans at N50:

  | Suite | Ungated groups | Gated groups |
  |---|---|---|
  | spatial | 2.4–2.7 min | 2.9–4.4 min |
  | LIBERO-10 | 4.2–6.2 min | about 12 min (stall MISS calls) |

  That makes about 80–85 min per library size.
- Cosmos is server-bound even in groups with no model MISS: each decision first runs the stage-1 encoder (about 130–190 ms) to build the retrieval key.
  - With R2 and no MPS, the request latency median was 8.9 s (H100 about 90% busy). With R4 and MPS it was 3.6 s; with R8 and MPS it was 0.35 s.
  - After R8 the lane is bound by workers and episode length. Groups with the stall gate are still slower because each stall runs a full K5 call.
  - A replica peaks at about 8 GB while loading. `memory_budget_gib` is split evenly across replicas, so R8 needs at least about 72 GiB (60 GiB OOMed at load). Steady state is about 5–8 GB per replica.
- Worker output on timan107: about 0.24 MB per episode under `/scratch/zixuans8/mw_r17/formal/runs`.

- **Driver storage.** The `legacy` driver fsyncs every journal, per-step and attempt row under its scheduler lock.
  - On 2026-10-11 /data (a busy HDD, 97% utilized) cost 71 ms per fsync. Cosmos cache episodes write about 33 per-step rows each, so the driver handed out only 10–20 episodes/min and the 145 workers were busy 11% of the time.
  - Moving the campaign's `driver/` directory to NVMe `/tmp` (a symlink created before launch; 6.7 ms per fsync) raised throughput to about 50 episodes/min with workers 95% busy.
  - For `legacy` campaigns, put `driver/` on NVMe or use `group-commit`; copy the directory back to /data afterwards.

**Prepare (CPU only).**
- DP: `cache_capacity_lanes.prepare('4090', <lane dir>, <prefix>)`, which uses `FLEETS['dp_libero']`.
- H100 models: `queue_retry.clone(snapshot, name, ..., worker_counts=nodes, ...)`. Example: [prepare_cosmos145.py](/home/weiland/projects/openpi_ext/scratch/r17_ops/prepare_cosmos145.py); run it with `PYTHONPATH=<repo>`.

**Run.**
- `exp/r17_integration/ops/formal_eval.sh --run --prepared-run /data/mw_finetune/eval/<run>`.
- The DP lane is `python -m exp.r17_integration.ops.cache_lane --lane 4090 --directory <lane dir> --run`; it resumes failed segments as `<run>_rN_xxxx`. If it gives up, move `LANE_RECEIPT.json` aside and rerun the same command.
- The H100 order runs from [chain_h100_cosmos.sh](/home/weiland/projects/openpi_ext/scratch/r17_ops/chain_h100_cosmos.sh): Fast sizes, then Cosmos 500, then Cosmos sizes.
- Do not use the `cache_model_order` or `queue_retry` queue runners. They treat a running predecessor's PARTIAL receipt as a failure and refuse existing run logs.
