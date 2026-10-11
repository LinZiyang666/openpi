# P4 → P6: DP native capture bridge

P4 consumes P1's published `r17.contracts.v1` directly. `DPTraceAdapter.capture`
accepts P1 `DecisionCapture` and returns `{metadata: JSON, arrays: named ndarray}`.
`from_stages(prepared,s1,result,wire)` consumes already-computed values and also
exports every native snapshot x_t / loop index / integer timestep / private
posterior RNG state. `save` writes numeric NPZ (no pickle) plus JSON.

The generated fixtures will be placed in `models/dp/samples/`. MW includes only
vision_0:64 + robot_state:4; LIBERO includes two64 keys + robot_state:8. H16,
internal-affine normalized chunks, distinct DDPM/DDIM sampler identities.
No debug method evaluates a model, reads the GPU, or consumes RNG.

Shared `ConnectionPolicy` currently exposes DecisionCapture but not the native
SampleResult. A request to P1 will cover forwarding already-computed native
capture for online writers; direct stage capture fixture remains usable now.

Update: P1 has forwarded native_snapshots and model_capture on DecisionCapture.
P4 consumes both. Snapshot arrays are now **flat** names for P6.numeric_arrays:
`snapshot_x`, `snapshot_loop_index` int64, `snapshot_native_timestep` int64,
`snapshot_rng_state` uint8; named keys are `key_vision_0`, etc. JSON snapshots
carry per-snapshot identity/conditioning/schedule metadata. Please update
DecisionSink to retain already-computed native_snapshots/model_capture (and
preserve posterior RNG uint8), or allow an injected ModelTraceAdapter so online
writer output is equivalent to P4's from_stages fixture. No new model calls.

Resolved: P6 DecisionSink now retains native snapshots/RNG and sample_fields;
P4 consumes the published bridge. CPU roundtrip includes closed numeric NPZ
and H5 with uint8 RNG states and int64 native timesteps. The fixture export uses
stacked flat names; online P6 uses snapshot_<index>_x/rng with JSON metadata.
Both represent the same native values; no legacy-flow time conversion.

## Fix round 1: Review B B3/B9/B11

P4 corrected reduced profiles to native trailing DDIM (first t99); DDPM100
remains default and K1 is explicitly lossy. New source/model/sampler identities
require resealing qualification assets and YAML; old leading bundles fail closed.
Please update qualification/model_runtime.wire_observations to use P4's
models.dp.fixtures.libero_observation (real paired images from libero_96), not
image[::-1], or preserve true paired wrist already in wire_observations.npz.
P4 probe will use the helper and a CPU real-pair regression test.
Capture timing remains overhead-bearing even after one conditioning digest per
request; pricing must exclude capture-enabled rows.

Current CPU release is sealed in samples/fix_round1_summary.json,
fix_round1_{libero,metaworld}_cpu/{receipt.json,capture/,load_cpu.json,k*.yaml}.
Both native probes pass13 checks; all10 K1/2/4/10/100 bundle files plus both
pinned loads pass12 final checks. DDPM100 unchanged; corrected low-K DDIM.
Reviewer48-frame proxy receipt fix_round1_quality.json includes exact indices,
paired PNG hashes, matched B48 seed0 and quantile/wire MSE. K1 remains lossy.

Round2 real fixture now generated and CPU-read successfully:
`models/dp/samples/fix_round2_qualification/wire_observations.npz` with sibling
JSON. 3 real recorded96 pairs, both wire camera keys, no-pickle arrays/strings,
both independent PNG hashes, no held-out/real-reset/SR claim. Q can use the
owned writer/reader or directly preserve the wrist key in its runtime reader.

Round2 final P4 suite114 passed/0 skip. The B11 regression now directly consumes
Q's qualification.model_runtime.wire_observations on the owned real paired NPZ
and compares BOTH cameras byte exactly to recorded PNGs. Q's current missing
wrist rejection is tested too. The old fabricated-camera fault makes this
regression fail. Q's full asset preparation/raw256/pass-count/GPU receipts
remain Q/coordinator scope; no P4 edits under qualification/.

## GPU fix 1 — Q_to_P4 native input API

Probe accepts Q's sealed --load-spec and real paired --observations NPZ, keeps
checkpoint/domain fallback, overrides only device and reports actual diagnostic.
Q must reseal expected_model_identity/runtime payload after probe-source edits;
stale pins remain rejected. P4 owns CPU tests/receipts and handback only. Local
Q original256 paired assets are available; formal MT50 export exists on H100
but not locally, so real formal checkpoint qualification stays Q/coordinator.
Final read-only P4 check confirms qualification/model_runtime.wire_observations
preserves both supplied cameras and rejects a missing wrist. No reader change
is pending; Q must refresh asset/runtime seals for the changed DP source identity.
P4 helper libero_observation(frame=...) and the new sealed256 probe provide
CPU-tested input paths. No qualification files edited and no GPU executed.

## GPU fix 2 — DP B1 costs and diagnostic isolation

Owner-authorized22.026s GPU diagnostic found composition-dependent DP outputs;
production B>1 is disabled, capabilities unsupported. All cache/MISS/recipe/
teacher arms must keep B1, including fitted teacher generation. Explicit
for_batch_diagnostic(label) clones are offline only and cannot register/serve;
their captures carry batch_diagnostic_label. Treat these composition measurements
as unqualified batch diagnostics; do not use them for production throughput or
pricing. CPU153 DP+49 legacy passed with0 skips; current native CPU probes pass13
each under the B1-only gate. Q has P4_to_Q.md with revised gate/reseal notice.
No P6/shared/qualification files edited; new-release GPU retry remains Q's work.

## 2026-10-10 DP-LIBERO matched dynamic batching launch seam

Owner now rules measured small numerical differences acceptable with identical
batch config in baseline/arms, and requests real DP server batching/launch config.
P4 reuses stock S07 core (P3 runtime is Fast-WAM-specific) and implements owned
models.dp.batch_runtime (max1..32), DDPM100/DDIM10/5 vector scheduler with exact
private per-row RNG draws, and models.dp.formal_batch_server. Please extend the
default-off --max-batch/--batch-wait-ms seam you're coordinating with P3 to DP
LIBERO. Campaign field dp_batching=BatchConfig(...).metadata(); choose owned DP
formal_batch_server when max_batch>1 and omit resident_lock. It accepts existing
--campaign/--port/--directory/--cuda-graph and attests cuda_graph_self_check,
execution sampler/commit and composition ledger. Default max1 stays old entry.
No shared edits by P4. Commit5/10/15/16 remains in bundle, not batch geometry.

Workers must be timan108 only (timan1 forbidden). Its total RAM is about252GiB;
Cosmos24×6GiB +10% headroom leaves~82GiB for DP, at most20 workers×4GiB before
other pending fleets. Existing Cosmos receipt lacked worker_memory_mib on read;
please reserve the owner's6GiB/worker, not the old default4GiB. Do not touch any
other campaign. Old stopped-DP fleet still appears RUNNING in receipts; new
launch must fail admission until its owner closes it if those PIDs remain live.
Recommendation will choose measured max batch that actually fills with admitted
worker count, probably≤16 with20 workers rather than blindly max32.
# Fix round 2 — B11/N1 qualification paired-view handoff to Q/P1

2026-10-09: REVIEW_B2 confirms qualification/model_runtime.py still fabricates
DP wrist via image[::-1], qualification/assets.py emits 480x480 LIBERO inputs,
and run_slot.py asserts old 11/12 pass counts instead of current 13/13. These
files are outside P4 ownership. Please forward to Q/coordinator/P1 and change
before GPU slots: use the owned `models.dp.fixtures.libero_observation` helper
for real paired 96x96 recorded views; serialize/consume the wrist key itself;
reject a missing wrist instead of synthesizing it; accept probe status pass
with skipped==0 rather than fixed pass counts. P4 is adding an owned numeric
NPZ fixture writer/reader and CPU regression for direct Q reuse, with PNG
provenance and both independently recorded cameras. 256 inputs must come
from real 256 pairs; do not upsample the 96 fixture and claim raw resolution.

# 2026-10-10 DP batching launch/RAM update

Owned DP local shim now exists: models/dp/formal_eval.sh / formal_launch.py,
--max-batch1..32 / --batch-wait-ms(default5). It strips its opt-in arguments
before calling the existing launcher, injects sealed campaign dp_batching,
and redirects only the formal resident module via formal_python.sh.
The DP side keeps the sealed provider/YAML identity and reports graph
self-check plus execution sampler/commit and per-stage batch membership.
Default B1 still delegates to your original launcher. No common files edited.

The initial GPU B4 run exposed global CUDA graph capture overlapping another
stage's readback. Fixed in DP: pre-capture every signature B1..max before any
S07 threads start; forbid all live graph captures. Final CPU180/0skip and
legacy49/0skip passed before the corrected GPU sweep. Measurement is ongoing.

RAM issue: at about11:33 CDT timan108 had251.63GiB total/138.85GiB available,
Cosmos24 live workers ~90.98GiB RSS. Its older receipt omits worker_memory_mib,
so current ops.worker_memory.admission falls back4GiB, while owner requires
6GiB. Correct future Cosmos reservation is144−90.98≈53.02GiB; plus25.16GiB
headroom leaves60.67GiB for new DP workers (12×4GiB fits;16×4GiB does not).
Please persist6GiB in Cosmos fleet receipts/defaults and use it for admission
of existing legacy Cosmos receipts. Do not signal/restart Cosmos.

DP supplies samples/node_ram_preflight.py: read-only remote PID/start-tick RSS
census, owner6GiB floor, normal10%/8GiB headroom, fail closed when Tether offline.
The exact DP launch will run this check first and still use your normal worker
admission. At12:01 CDT Tether timan108 was offline (exit64); we did not start
agents or touch any campaigns. Latest launch admission remains pending.

# Final DP result / ready owned launch shim

models/dp/BATCHING.md has actual35/36 matrix, recommended1replica/max8/window5ms,
12 timan108 workers (actualStage3 B8/mean6.69); K10014.5calls/s,708/1248ms,
155.3W. Two/max4 gives18calls/s575/779ms butStage3 max3 at12clients. DefaultsB1.
CPU182/0skip+legacy49/0skip pass. Graph startup now capturesB1..max then denies
live capture. Native B32 private RNG352requests exact; maxwire drift0.000345707.
GPUbudget1199.589s complete, all own PIDs closed. K5/r2/B32 timing empty;
Q/coordinator must allocate later. SR/fresh matched control pending.

Exact formal_eval.sh command is the owned shim models/dp/formal_eval.sh; it
preserves your current P6 path and strips DP batch opts before the Fast-only
parser, injecting dp_batching before prepare and redirecting only residententry.
No shared/ops/P3/qualification code edited. Native K100/C5 defaults are used:
your parser rejects even --denoising-steps100 with --commit-horizon5 together.
Model queues have CPU coverage for all100/10/5 × commit5/10/15/16, so combined
step/horizon launch studies need your admission change separately.

Older Cosmos receipt omits6GiB worker reservation; DP supplies stronger readonly
samples/node_ram_preflight.py before normal launch admission, using your host/
cgroup capacity function and6GiB Cosmos floor. Latest Tether timan108 OFFLINE
exit64; no agent/campaign restarted. New official baseline was not launched.

## P3 normal DP seam consumed — final command

P3's12:07 publication resolves the shared DP batch flags/campaign/dispatch seam.
BATCHING.md now uses ops/formal_eval.sh directly, max8/window5ms, one local4090
resident,12 timan108 workers, four suites/official50, native K100/C5 defaults.
Exact normal arguments parse without main/deploy. CPU185passed/0skip and
legacy49passed/0skip; native regression checks graph forwarding and stage env.
No shared edits by P4; owned local shim remains fallback. Latest readonly
admission still exits64/OFFLINE and stops the supplied command before GPU use.

Remaining P6 items:6GiB floor for older Cosmos worker receipts, separate
combined step/horizon CLI admission, and coordinator/P4/P6 DP paired-control
gate (DP emits membership ledger; common enforcement currently Fast-WAM only).
GPU budget is exhausted1199.589s;35/36 timing cells and SR limit unchanged.

## 2026-10-10 cache-study request completed on CPU

All five items in P6_to_P4 are implemented in the owned DP area:

1. `formal_batch_server.py:14` explicitly admits cache_study at the accepted
   control R2/B8/5ms/MPS/local settings, plain full.yaml startup, native DDPM100;
   the existing stage queues, graph prewarm and connection factory are reused.
2. `library.py:57` supplies explicit r17.teacher_compatibility.v1/PASS with
   teacher/resident retrieval identities, 19 pinned producer files, 10 asset
   pins, six dependency versions, normalization/feature/layout/output checks.
   Unknown teacher code and producer edits fail. Accepted hs5 control producer
   and asset pins match exactly; native E CPU producer/reference bytes pass.
3. Native DP/S07/real-WebSocket eight-group test passes256 decisions on one
   resident generation:160 HIT,48 library tails,32 stall-only DDPM100 MISS,
   16 policy tails. Native NFE3200, only gated groups; stage1/2 B8. CPU window
   20ms for reliable fixture fill; formal admission remains5ms.
4. `admission.py:2` uses sealed_store_descriptor VIEW binding with the view
   digest; real size500 regression and other-eta K1 VIEW rejection pass.
5. `provider.py:10,28` exports/attaches frozen configure_library_provider;
   lazy/default-off registration and same ID/spec regression pass. Shared P6
   sealer and resident both consume it without replacing the registry provider.

CPU full201/0skip/0fail; legacy49/0skip/0fail; your cache framework33/0skip/0fail.
Actual 475-tensor CPU replay on a real paired256 frame from each suite passes
exact native conditioning, wire DDPM100 and output transforms. All16 real
fitted groups load, actual calibrated Q3 initializes inactive; stored GPU/CPU
visual-key max diff0.000889719 diagnostic (no quality claim).

Unmodified `smoke.sh dp --prepare-only` exited0/PASS/cpu_only at
`/data/mw_finetune/eval/mps_dp_libero_1010_164432_2c1f7e32`; all16 YAMLs sealed,
32-episode plan and worker imports/pools prepared, dispatch_started=false.
Fresh resident ID sha256:bd1dd4c1b2e1bfbb4c660ed8ec0a50362f461dc8a631c16edfa9d3097879d0c5.
**Worker memory_admissible=false**: current timan107 fleets reserve the node;
no available bytes after headroom. Preserve32 workers and rerun admission when
the campaigns finish; no running fleet was touched. This is not launch PASS.

Coordinator command, NOT EXECUTED (CPU-only task):
`flock -n /tmp/r17_gpu.lock bash exp/r17_integration/ops/cache_study/smoke.sh dp --run`
Estimate15–25min for32 episodes/16 barriers; actual GPU duration unmeasured,
script ceiling120min. Keep R2/B8/5ms/MPS/group-commit and verify your audit.
Please refresh your DP template model-admission status using these receipts;
P4 did not edit STUDY.json/AUDIT.json, shared framework or qualification/.
Details and remaining gates: models/dp/CACHE_STUDY.md, HANDBACK_P4 latest section.

