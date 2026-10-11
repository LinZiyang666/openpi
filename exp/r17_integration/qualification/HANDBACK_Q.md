# HANDBACK Q — R17 qualification preparation

**GPU fix4 CPU preparation PASS, 2026-10-09.** All remote archive transfers now
use SHA-checked allowed /tmp stages, including H100 relay pushes and remote
LIBERO pulls. Four real tether CPU transfer proofs PASS. The three historical
MW B runs also pass recovered CPU capture/journal/physical-control audits;
their original PARTIAL receipts remain unchanged. All34 final slot preflights:
30 PASS/four expected PARTIAL, zero FAIL. Q/default-off62 PASS; conductor73 PASS,
zero skips. Both worker probe replays and pinned DP CPU probes PASS (13/0 each).
The current619-file runtime has no source drift; nine published native asset
identities match. Native assets are retained because this fix changes Q transfer
code. No Q GPU use or training/eval process interference occurred.

**CPU-ready order, subject to GPU admission:** π G1 B; GR B/current;
GR final60000 B; DP-LIBERO replicas. Preserve the coordinator's running GR_k2,
π_k2 and DP-LIBERO evals and timan workers. Default wrappers acquire the GPU
lock without signaling its holder. π G1 A, GR A/current+60000/K2/K1 and
DP-LIBERO A+B already GPU PASS and need no repeat while their audited native
identities stay unchanged. G3_dp_libero's intended scoped PARTIAL was observed.
The appended GPU fix4 section and [PLAN.md](PLAN.md) give exact commands and
conditional follow-ups. H100 still waits for Fast-WAM training to exit; the
latest CPU census has no DP-LIBERO training rows. Both training-exit conditions
and ordinary GPU admission remain required; clocks do not grant admission.

Current final index: [GPU_FIX4](receipts/GPU_FIX4.json). Earlier release sections
are superseded history, including old /scratch transfers and DP-LIBERO retries.

## Historical pre-fix-round-2 handback (superseded)

2026-10-09. **34 concrete, executable slots prepared; zero GPU use by Q.**
All **28 current-checkpoint slots pass CPU preflight**. Six explicitly named
future-checkpoint slots return the expected PARTIAL until publication. No GPU
gate has been certified. Start the first end-to-end target after GR00T training
actually exits and the local driver is usable:

```bash
bash exp/r17_integration/qualification/slot_G1_pi05_mw_A.sh
bash exp/r17_integration/qualification/slot_G1_pi05_mw_B.sh
```

[PLAN.md](PLAN.md) orders every slot by GPU, gives duration/peak/free-memory
estimates, sharing eligibility with the formal 500-episode evaluations, exact
checkpoints, Python islands, routes, protocols and pass criteria. Estimated
release times are 18:40 CDT local / 20:30 CDT H100. Admission checks actual
training exit, locks, ports and free memory; H100 also waits for DP training to
exit. Q did not access anyone's tmux or signal any training process.

## Ready artifacts and operation

- G1 π A/native, B/two real mechanics episodes, C/diagnostic tail.
- G2 GR00T MW A/K4, A/K2, A/K1, B/mechanics, C/native S3 B2, R/two replicas.
- G2 DP MW and LIBERO A/B/R; Fast-WAM MW A/B/C-B2/R and LIBERO A/B/C-B2.
- Six G3 model/domain slots with implemented native capture/library/fit/replay
  checks and explicit PARTIAL for the remaining full-G3 requirements.
- Six later A/B variants: GR checkpoint60000, Fast policy016001 and the
  explicitly selected DP step100000 snapshot. No live/latest checkpoint is used.

Every shell command starts a fresh run in
`/data/mw_finetune/qualification/runs`, records owned PIDs/argv/exit status,
stops owned children through their Popen handles or attested remote PID/birth/
argv, writes `receipt.json`, and prints PASS/PARTIAL/FAIL (exit 0/2/1).
Work stops at 840s; outer TERM is 885s and KILL 890s, with a stdlib timeout
receipt. Remote MW and LIBERO supervisors have independent 500s work bounds.
Missing/error/late attempts, forced shutdown and incomplete transfers cannot
produce PASS. No process-name kill or `pkill` is used.

Both timan payloads are **already installed and import-probed through tether**
using P6's real SHA installer. Latest full deployment:
[deployment_13](preflight/deployment_13/WORKER_DEPLOYMENT.json).
Both destinations are
`/scratch/zixuans8/mw_r17/qualification/payload_eeede63f12132ffc1ba1bcab3e4fbdb5dcb9ed808b5ea087c6e63d08ab649eb9`.
The final H100 runtime is `/data/mw_finetune/qualification/runtime_q1`, archive
SHA `32526d4af0a2200ee20afa9ec2680f769f2bb1ca185bbae491ec7685e64ea040`
([deployment_14](preflight/deployment_14/RUNTIME_DEPLOYMENT.json)).
All slot specs name the current payload destination.

Policy/driver routes passed actual CPU echo tests from their assigned nodes:
timan107 → `ziyanglin.com:23156/23157`; timan108 →
`149.165.153.233:23256/23257`. H100 → local Q relay23158 → timan108 also passed.
H100 needs no tether credentials; its slot automatically starts the bounded
authenticated relay on the coordinator host. Replica ports are local23160/61
or H10023260/61. Formal eval ports and I018's23180 are untouched.
The final read-only `ss -tlnp` census found no listeners on Q's assigned ports.

Eight sealed CPU assets are in `/data/mw_finetune/qualification/assets`:
local `pi05_mw_v3`, `groot_mw_v3`, `groot_mw_k2_v3`, `groot_mw_k1_v3`,
`dp_libero_v5`; H100 `dp_mw_v3`, `fastwam_mw_v3`, `fastwam_libero_v4`.
Each has a manifest/load descriptor, validated configs, sealed synthetic
teacher/library/fit products, and an explicit SYNTHETIC label. Donors are three
fabricated episodes / nine decisions with production geometry. They certify
mechanics only. LIBERO donors/recipes use the actual spatial task0 instruction,
not a generic task label. Earlier sealed assets and failed diagnostic receipts
are retained; none was silently rewritten.

The current checkpoints are the requested π RLinf, GR30000, Fast MW sanity4000,
published Fast LIBERO and brandonyang DP LIBERO30k. DP MW uses H100's complete
immutable `checkpoint-030000`; its step/complete marker is checked. Fast models
remain on H100; no large local Fast checkpoint copy was necessary.

## CPU verification

| Evidence | Result |
|---|---|
| [CPU_SLOTS.json](receipts/CPU_SLOTS.json) | 34 wrappers executed with `--preflight` and GPUs hidden: 28 PASS, six unpublished variants PARTIAL; all `gpu_executed=false`. Final runs are under `preflight/slots_cpu_sealed` on the relevant host. |
| [ASSET_SEALS.json](receipts/ASSET_SEALS.json) | Eight current asset sets PASS, with checkpoint/norm/sampler/layout and file-content seals. |
| [DRIVER_PLANS.json](receipts/DRIVER_PLANS.json) | Six MW plan-only drivers and two actual Python3.8 LIBERO task/init-state plans PASS. No simulator environment or GPU renderer was created. |
| [DP_NATIVE_CPU.json](receipts/DP_NATIVE_CPU.json) | Real CPU checkpoint probes: DP LIBERO11 / DP MW12 checks pass, including K100/2/1, normalization, native suffix/DDIM and B2. |
| [cpu_suites_final.xml](receipts/cpu_suites_final.xml) | 176 cross-package CPU tests pass; zero failures/errors/skips. |
| [q_tests_sealed.xml](receipts/q_tests_sealed.xml) | 11 Q tests pass: owned deadline/foreign-process isolation, source-drift rejection, lossless BF16, actual LIBERO vocabulary binding, strict physical joins, H100 startup, native π composition/tails, full CPU G3 composition and real WS hot bundles. |
| [owner_last_refresh2.xml](receipts/owner_last_refresh2.xml) | 56 overlapping capture/conductor checks pass after the latest owner edits. This is a refresh, not an additive test total. |
| [STATIC.json](receipts/STATIC.json) / [SOURCE_SNAPSHOT.json](receipts/SOURCE_SNAPSHOT.json) | 24 Python files parse; 34 shells pass `bash -n`; all specs match the deployed payload; 597 archived source/spec files match at final verification. |
| [routes_local](preflight/routes_local.json), [routes_h100](preflight/routes_h100.json), [relay_cpu](preflight/relay_cpu.json), [PORT_CENSUS](receipts/PORT_CENSUS.json) | Actual CPU network and deployed-payload proofs PASS. Owned listeners exited. |
| [VERIFICATION.json](receipts/VERIFICATION.json) | Aggregate CPU result and evidence counts; no GPU qualification claim. |

Original π config emission and owner plan-only commands also exited0; retained
logs are `preflight/pi_owner_emit_v3.log` and `pi_owner_plan_v3.log`.
GPU admission and renderer attestation remain for the coordinator's future
slots. The B verifier requires exactly two accepted error-free real episodes,
finite native geometry, contiguous physical controls and exact server execution
counts/cursors **including the terminal acknowledgment**. A policy failure at
the full control cap is an accepted mechanics outcome. No two-episode result
is reported as a formal 500-episode SR or isolated pricing certificate.

## Fixes, remaining requests and limits

Q fixed its own canonical GR observation adapter/provider registration,
R17 server/config/recipe composition, H100 relay, Python3.8 LIBERO context/ack
shim, final execution validation, NumPy/tensor parity checks and H100 startup
scope. No P-owned implementation was edited by Q.

[Q_to_P1](../REQUESTS/Q_to_P1.md): the zero-scatter numerical error is now
**resolved by P1** with a clear rejection and a passing Q witness. Full G3
still needs certified native warm/reset and CP2/twins/shadow/routing executor
dispatch. Q does not bypass the fail-closed capability checks.

[Q_to_P2](../REQUESTS/Q_to_P2.md): still pending a distinct physical π-MW
replan1 protocol/provider bind and real control/terminal receipt. The C slot
checks four exact zero-NFE native suffixes using synthetic one-control acks
and prints PARTIAL. Formal H5/replan5 remains unchanged.

[Q_to_P6](../REQUESTS/Q_to_P6.md): production BF16 serialization is now
**resolved by P6** with lossless float32 values plus exact uint16 bits and
metadata. That fix is in the final payload, and Q retains its tested observer
boundary. Advanced/real-cohort G3 remains pending. All six G3 slots print
PARTIAL even when their implemented subset passes: real dev/test freeze,
semantic physical restore, advanced dispatch and full coverage are still
missing. Coordinator should dispatch the remaining P1/P2 requests.

The ledger reports a local driver/library mismatch. Q did no repair or GPU
query. A live slot defers with PARTIAL if the owner's driver repair is still
needed. Memory-sharing entries in PLAN are conservative budgets, not measured
GPU peaks; free-memory admission checks them again.

## Storage, source refresh and ownership

[H100_SPACE_AND_ASSETS.json](receipts/H100_SPACE_AND_ASSETS.json): Q data is
**67,321,365 bytes**; shared `/data/mw_finetune` is **82,505,646,387 bytes**.
Both fit the 5GB Q addition / 120GB shared limits, preserving 13GB for a
Fast save. Live H100 admission reserves that save headroom and enforces a 4GB
Q operating cap with archive/cleanup headroom. No checkpoint weight trees or
optimizer states were copied into Q space.

Dependency provisioning added only the Q h5py3.14.0 overlay (no training-venv
change) and restored two absent published Fast LIBERO normalizer files from
the identical local checkpoint, **1,632 bytes total**, before sealing. No
existing checkpoint file was overwritten. Q files/artifacts live only in the
qualification directory, its `/data/mw_finetune/qualification` trees, temporary
Q deployment stages and the three explicitly requested Q→P request files.
No protected tree, foreign tmux/process, git commit or push was touched.

Each preflight verifies `runtime_code_snapshot.json` before loading assets or
issuing a GPU command. Owner source changes after this handoff return PARTIAL
with the changed paths. A CPU refresh, from the coordinator host, is:

```bash
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:src:packages/openpi-client/src
stamp=$(date +%Y%m%d_%H%M%S)
.venv/bin/python -m exp.r17_integration.qualification.deploy_cpu --output "exp/r17_integration/qualification/preflight/refresh_${stamp}"
.venv/bin/python -m exp.r17_integration.qualification.gen_slots
.venv/bin/python -m exp.r17_integration.qualification.deploy_cpu --runtime-only --output "exp/r17_integration/qualification/preflight/refresh_specs_${stamp}"
.venv/bin/python -m exp.r17_integration.qualification.cpu_preflight --output "/data/mw_finetune/qualification/preflight/refresh_${stamp}"
```

If a model owner's code changes its sealed model identity, prepare a new asset
directory with that model's island and regenerate specs; retain old seals.
Checkpoint publication alone needs no code refresh: the explicit future A
wrapper can seal it with `--preflight` (CPU) or a bounded 180s preparation phase
inside its live slot. No waiting or automatic GPU launch is scheduled.
JSON/NPZ/pickle receipts are ignored by the repository's existing gitignore;
they are present in the workspace and must be retained explicitly by the
coordinator if packaging the final integration change.
## Fix round 2 preparation history (superseded)

Status as of 2026-10-09 14:45 CDT: **Q code and regressions pass; final CPU
resealing, all-slot regeneration and deployment wait for the owner handbacks.**
No GPU command, training/tmux manipulation, commit or push was performed.

| Finding | Change | Regression / evidence | Current status |
|---|---|---|---|
| B2 N1(a), LIBERO480 inputs | Extract original paired RGB256 dataset episode1272 frames0/1/2 with corresponding recorded states and original prompts; seal encoded/RGB/NPZ hashes; preflight verifies real256 pairs. Live original-kernel shim sends256. | `test_n1_original_256_pairs_and_wrist_passthrough`: byte-exact source decoding, both real DP/Fast CPU preprocessors accept256 and reject480; [H100 file proof](preflight/fix2_h100_original_pairs.json). | CPU PASS; final sealed assets pending owners |
| B2 N1(b), DP12/11 hard count | Require status pass, skipped0, nonempty all-pass checks, passed==len(checks), and exact sealed weight/ModelSpec. | `test_n1_dp_receipt_uses_real_probe_checks_not_old_count`: valid13/14 accepted; skipped/failure/count/seal disagreement rejected. | CPU PASS; real pinned probe awaits P4 seam |
| B2 N1(c), disabled Fast B2 | Convert both C_B2 wrappers to retired CPU-only phase, zero memory budget, PARTIAL before source/model/GPU inspection; no --batch2 native command. | `test_n1_retired_fast_batch_slot_never_starts_process_or_gpu`, both domains; subprocess launch is forbidden by test. | CPU PASS; final generated catalog pending owners |
| B2 N1(d), fabricated DP wrist | Consume the recorded wrist key unchanged; missing wrist fails closed. | Same real-pair regression compares each wrist byte to source PNG and rejects missing wrist; neither camera is derived from the other. | CPU PASS |
| A2 usage, scheduling/task vocabulary | π G1 explicitly periodic_gap2/target_irNone. Exact complete MT50 ordered table and task-table identity; official40 LIBERO canonical suite/task names, language only as prompt. | `test_a2_g1_periodic_recipe_uses_exact_canonical_mt50_order` accepts canonical table and rejects reordered table; original LIBERO plan-only PASS at `/data/mw_finetune/qualification/preflight/fix2_libero_plan_01`. | CPU PASS |
| Owned seams | Attest real per-K samplers to MW worker debug; exclude DP K1 from live YAML/recipe/G3 augmentation; bind Fast load to P2 benchmark documents. | Real Q WebSocket/hot-bundle test verifies K2/K10 metadata; full Q suite17 passed. | CPU PASS; final manifests pending owners |

Executed CPU checks: [Q17 tests](preflight/fix2_q_initial.xml),
[legacy457](preflight/fix2_legacy.xml), [six socket checks](preflight/fix2_legacy_proxy.xml),
[default-off2](preflight/fix2_default_off.xml), [R16 seal](preflight/fix2_r16_verify.log).
Six ordinary legacy skips were all executed successfully as socket checks;
one real EGL test was deselected to respect zero GPU.

Cross-package requests for coordinator dispatch:
[Q_to_P4](../REQUESTS/Q_to_P4.md) asks native probe to consume Q's pinned load
descriptor and recorded paired256 NPZ so its ModelSpec equals the CPU seal.
[Q_to_P6](../REQUESTS/Q_to_P6.md) originally reported a worker whitelist
conflict; P2 removed its client registry dependency, resolving it without a
P6 patch. [Failing historical repro](preflight/fix2_builder_repro.log) and
[passing current builder](preflight/fix2_builder_recheck.json) are retained.
P-owned code was not edited by Q.



## Fix round 2 — B3/A3 refresh before completion confirmation (superseded)

All assigned Q changes are complete. P1–P4 Fix round 2 handback SHA proofs are
recorded in [SLOTS.json](SLOTS.json). B3's DP protocol parser, protocol checks,
new code identities and Fast-WAM lazy hash changes are consumed. The A3 final
report/code present at deployment is also consumed; the final 609-file source
check reports no drift. Existing sealed stores were retained, and new stores
are explicitly labelled SYNTHETIC. These are qualification mechanics, not
formal 500-episode quality or isolated pricing certificates.

| Finding | Change | Test / evidence | Status |
|---|---|---|---|
| REVIEW_B2 N1: LIBERO480 frames | Use original dataset episode1272 frames0/1/2, matching recorded states/language, original256 front and wrist views; validate encoded/RGB/NPZ hashes. Live shim keeps256. | `test_n1_original_256_pairs_and_wrist_passthrough`: compares original decoded PNG bytes and states; real Fast/DP CPU validators accept256, reject480; missing wrist rejected. Both fresh LIBERO seals and original-kernel plans PASS. | Q fixed; CPU PASS |
| REVIEW_B2 N1: DP12/11 checks | Accept nonempty all-pass probe checks with status pass, skipped0, passed==len(checks), exact sealed weight and ModelSpec. No fixed count. | `test_n1_dp_receipt_uses_real_probe_checks_not_old_count`: valid13/14 accepted; skips, failed check, count or seal mismatch rejected. Current real stock CPU probes13/0 in both domains. | Q fixed; pinned native A gate PARTIAL pending P4 API |
| REVIEW_B2 N1: Fast B2 disabled | Both C_B2 commands are retired CPU-only slots: zero memory budget, no native --batch2, receipt PARTIAL before model/source/GPU work. | `test_n1_retired_fast_batch_slot_never_starts_process_or_gpu` forbids launch in both domains; regenerated-catalog regression; both wrapper receipts `gpu_executed=false`. | Q fixed; intentionally retired |
| REVIEW_A2 usage: π schedule | Explicit periodic_gap2/target_irNone for retrieval recipe. P1 IR arithmetic is fixed, but MW cost coefficients remain unmeasured, so no IR tier claim. | `test_a2_g1_periodic_recipe_uses_exact_canonical_mt50_order`; native/full path keeps recipes disabled, retrieval enables the explicit periodic recipe. | CPU PASS |
| REVIEW_A2 usage: ordered_tasks | Complete canonical MT50 table and task-table identity in every bound YAML; complete official40 LIBERO table for LIBERO. | Same regression verifies exact canonical order and rejects swapping tasks; all generated YAML validations and eight driver plans PASS. | CPU PASS |
| Latest user update/B3: formal DP-MW/current identities | All DP-MW variants select `/data/mw_finetune/dp/mt50_policy_100000`, derived from immutable scratch checkpoint100000. CPU discover pins then reload diagnostic=False under owner formal checks. LIBERO load consumes actual P2 manifest, including B3 parser seals. | `test_b3_dp_load_uses_p2_manifest_and_formal_pins`, `test_formal_dp_export_checks_step_mt50_and_pins`, catalog assertion; formal real CPU seal PASS. | CPU PASS; new generation sealed |
| Owned audit fixture after A3 | Keep queue/service/cost in native diagnostics so writer backpressure cannot overwrite policy timing; retain physical/cursor/terminal disagreement rejection. | `test_closed_mechanics_audit_and_physical_disagreement` exercises real closed capture and negative physical joins. | CPU PASS |

Final validation:

| Evidence | Result |
|---|---|
| [Q/default-off suite](preflight/fix2_q_b3a3_final.xml) | 23 PASS: 21 Q tests, two default-off checks; no skip/failure |
| [Current R17 composition](preflight/fix2_composition_refreshed.xml) | 648 PASS across current common/benchmarks/conductor/models/debug/search/ops, including B3/A3 regressions |
| [Legacy CPU rerun](preflight/fix2_legacy_cpu_final.xml), [socket rerun](preflight/fix2_proxy_final.xml) | 457 PASS, six socket skips resolved by separate6 PASS, one real EGL test deliberately deselected |
| [R16 verification](preflight/fix2_r16_final.log) | DESIGN_VERIFIED; no outcome reads. Shared/frozen source diff remains empty |
| [CPU_SLOTS](receipts/CPU_SLOTS.json) | Every34 wrapper executed with --preflight:25 PASS/9 PARTIAL/no FAIL; runs `/data/mw_finetune/qualification/preflight/fix2_slots_b3a3_final` on relevant host |
| [ASSET_SEALS](receipts/ASSET_SEALS.json), [DRIVER_PLANS](receipts/DRIVER_PLANS.json) | Eight fresh CPU seals and eight plan-only drivers PASS; LIBERO plans create no environment |
| [DP_NATIVE_CPU](receipts/DP_NATIVE_CPU.json) | Current real DP stock CPU probes13 checks each/skipped0; explicitly diagnostic and not substitutes for Q's pinned native A gate |
| [STATIC](receipts/STATIC.json), [SOURCE_SNAPSHOT](receipts/SOURCE_SNAPSHOT.json) | 29 Python files parse;34 shells bash -n PASS;609 source/spec files match deployed snapshot |
| [Routes local](preflight/fix2_routes_local_final.json), [routes H100](preflight/fix2_routes_h100_final.json), [relay](preflight/fix2_relay_cpu_final.log) | Actual timan107/108 policy/driver echo and H100→Q relay→timan108 payload import PASS |
| [PORT_CENSUS](receipts/PORT_CENSUS.json) | Q lanes free; protected H100 DP23254/55/58/59 still listening; no process modification |
| [VERIFICATION](receipts/VERIFICATION.json) | Owned findings PASS; qualification readiness PARTIAL only for listed dependencies/scope |

Final payload deployment is [fix2_deployment_06](preflight/fix2_deployment_06/WORKER_DEPLOYMENT.json),
using P6's CPU/file-only installer and both real tether import probes. Both
nodes use `/scratch/zixuans8/mw_r17/qualification/payload_acbe45dfbf523fc21d851ea51f2e1c9d5f191ca2b3eb6eb57dc34ed14dbc286b`.
H100 runtime [fix2_deployment_07](preflight/fix2_deployment_07/RUNTIME_DEPLOYMENT.json)
is `/data/mw_finetune/qualification/runtime_q1`, SHA
`3f9fd02012156c65c5f23be3dd6d2f0d29922865bcd5b872bd20f0fb8eef3972`.
Each slot pins this worker destination and uses only its reserved Q lanes.
The r17_srv_dp1..4 servers and timan108 r17_dp1..4 drivers were untouched.

Fresh assets use suffix `_fix2_b3a3_v1`: local π, GR K4/K2/K1 and DP-LIBERO;
H100 DP-MW, Fast-MW and Fast-LIBERO. DP final100000 intentionally reuses the
same formal DP-MW seal. No model weights were copied into Q space. Current
[H100 space receipt](receipts/H100_SPACE_AND_ASSETS.json): Q144,686,007 bytes;
shared85,158,830,326 bytes, below107GB while preserving13GB transient save
headroom and well within Q's5GB allowance.

Pending cross-package work is [Q_to_P4](../REQUESTS/Q_to_P4.md): optional probe
--load-spec/--observations, preserving actual diagnostic flag and paired256
inputs. Three DP A slots defer before GPU admission until its help/API checks
pass; run DP B only after matching A passes. Earlier P2 default-config and P6
reach task-id fixture requests are resolved by A3 and the648-test green run.
Distinct real replan1 tail qualification and full advanced G3 remain historical
P1/P2/P6 scope requests; their slots retain explicit PARTIAL interpretation.

**A3 refresh still required after all reviewer A edits finish.** Regenerate
with gen_slots.py, refresh full worker payload via deploy_cpu, regenerate to
pin new payload paths, then refresh runtime-only and re-run every CPU preflight.
If a sealed model identity changes, create a new asset generation using its
CPU island first. Runtime drift defers before any GPU command. The refreshed
PLAN records this requirement even though A3's final report is now on disk.

Safety deviation: the initial legacy refresh used an incorrect -k exclusion,
so `test_same_identity_same_initial_state_across_processes` ran its real EGL
kernel with CUDA hidden. It completed; GPU rendering cannot be ruled out.
[That run](preflight/fix2_legacy_refreshed.xml) is excluded from CPU-only totals.
The corrected legacy_cpu_final run explicitly deselects it. No GPU policy,
training/eval server, tmux, protected tree, commit or push was touched.


## Final refresh after confirmed A3 completion

A3 completion refresh is **complete**. gen_slots.py regenerated all34 wrappers
and SLOTS/spec files against current code. [SLOTS.json](SLOTS.json) records
SHA256 proofs for REVIEW_A3.md/REVIEW_B3.md and the P1–P4 Fix round2 handbacks.
No model source changed since the previous B3/A3 seals: the checked source
snapshot was identical. The eight `_fix2_b3a3_v1` sealed asset sets were safely
retained, revalidated and used by the new plan/preflight runs.

| Final evidence | Result |
|---|---|
| [CPU_SLOTS](receipts/CPU_SLOTS.json), [all-slot log](preflight/a3_slots_sealed_final.log) | Every34 wrapper:25 PASS/9 expected PARTIAL/no FAIL; all gpu_executed=false |
| [Q/default-off](preflight/a3_q_deployed_final.xml) | 23 PASS, no skips/failures |
| [Current integration](preflight/a3_composition_final.xml) | 648 PASS, including final A3 GR timing/default table and P6 service/IR/writer regressions |
| [Legacy CPU](preflight/a3_legacy_final.xml), [sockets](preflight/a3_proxy_final.xml) | 457 PASS; six socket skips covered by separate6 PASS; real EGL test explicitly deselected |
| [Asset seals](receipts/ASSET_SEALS.json), [driver plans](receipts/DRIVER_PLANS.json) | Eight each PASS; current plan runs under `/data/mw_finetune/qualification/preflight/a3_plans_sealed_final` |
| [STATIC](receipts/STATIC.json), [source snapshot](receipts/SOURCE_SNAPSHOT.json) | 29 Python files parse,34 shells bash -n;609 source/spec files match deployment |
| [Local route](preflight/a3_routes_local_final.json), [H100 route](preflight/a3_routes_h100_final.json), [relay](preflight/a3_relay_cpu_final.log) | Actual timan107/108 CPU echo and H100→relay→timan108 import PASS |
| [R16 verify](preflight/a3_r16_final.log), [port census](receipts/PORT_CENSUS.json) | DESIGN_VERIFIED/no outcome reads; Q ports free; protected DP23254/55/58/59 still listening |
| [Aggregate](receipts/VERIFICATION.json) | Q prep PASS; GPU readiness retains named PARTIAL dependencies/scopes; A3 dependency removed |

Final worker installation/import probes:
[a3_deployment_09](preflight/a3_deployment_09/WORKER_DEPLOYMENT.json), both
nodes at `/scratch/zixuans8/mw_r17/qualification/payload_acbe45dfbf523fc21d851ea51f2e1c9d5f191ca2b3eb6eb57dc34ed14dbc286b`.
Final H100 runtime:
[a3_deployment_11](preflight/a3_deployment_11/RUNTIME_DEPLOYMENT.json),
`/data/mw_finetune/qualification/runtime_q1`, SHA
`51ef0b7651a85e7b0dd9b6dfb755116f1b50afc175bfa1560513063de91c29ea`.
Repeated deployment originally collided with an unchanged staged archive;
Q's deploy_cpu.py now gives each run a fresh stage while P6 verifies the
same immutable installed payload. The full repeated deployment passes.
No --force overwrite or P-owned code edit was needed.

Final operating order (the full34-row duration/memory/prerequisite tables are
in [PLAN.md](PLAN.md)):

- **4090/timan107:** π G1 A→B; GR G2 A→B→A_k2→A_k1→C_B2→R; π G3→GR G3→π diagnostic tail. Conditional DP-LIBERO A→B→R→G3 follows P4's API fix. GR60000 A→B follows immutable checkpoint publication.
- **H100/timan108:** Fast-MW A→B, Fast-LIBERO A→B; Fast-MW G3→Fast-LIBERO G3→Fast-MW R. Conditional DP final100000 A→B, current DP-MW A→B→R→G3 follows P4's API fix. Fast final16001 A→B follows immutable publication. Both retired Fast C_B2 wrappers stay CPU-only.

A must PASS before its B; A+B must PASS before replicas/G3. GPU admission
still waits for actual training exit, locks, Q ports and free memory. Memory
sharing remains explicit, estimated and subject to live admission. Every G3
and π tail retains the documented PARTIAL scope. Current/final100000 DP-MW
A/B use the same immutable formal export and seal; the current-name A/B are
repeat runs of that same checkpoint, not a distinct qualification cohort.

Nine CPU partials are exactly three DP A/API dependencies, four unpublished
GR60000/Fast16001 variants and two retired Fast B2 commands. All dependent DP
GPU slots wait for matching A even where their CPU preflight passes. The sole
native-gate request is [Q_to_P4](../REQUESTS/Q_to_P4.md); A3 is no longer a
pending request. Any later source edit still requires a fresh deployment and
resealing where identity changes, with the source guard deferring before GPU.

All MW retrieval recipes use explicit periodic_gap2 and target_irNone with
the complete canonical MT50 table; no measured IR profile is fabricated.
P6's final service-only pricing and owner-IR analysis code is in the deployed
runtime, but debug/shared smoke results carry no isolated pricing certificate.
DP-MW final slots retain `/data/mw_finetune/dp/mt50_policy_100000`,
diagnostic=False and pinned weight/norm/model identities. LIBERO retains the
original recorded256 paired views and actual P2 benchmark manifests.

[H100 space](receipts/H100_SPACE_AND_ASSETS.json): Q144,765,367 bytes;
shared85,158,913,232 bytes; both within limits with13GB save headroom.
This final refresh used zero GPU operations and excluded the real EGL test.
Running DP eval servers/ports/drivers and all trainings/tmux were untouched;
no commit/push, protected-tree edit or process-name kill occurred.


## GPU fix 1

All requested Q changes are implemented and CPU/tether checks completed.
No Q GPU command/query, simulator environment creation, foreign process/tmux
operation, P-owned code edit, commit/push or protected-tree edit occurred.
P2 subsequently supplied the explicit [P2_to_Q](../REQUESTS/P2_to_Q.md) patch;
Q adopted it only in its owned pi_native.py (SHA
`db2dfbd03286fe0a02749e2008a231fe49c6c463b3fd4823110edca30ae0ed7c`).

### Probe diagnosis and request

Both deployed workers use `/scratch/zixuans8/mw_r17/envs/sim/bin/python`
3.11.15. Their installed distribution versions exactly match the validator:
metaworld3.0.0/mujoco3.14.0/gymnasium1.3.0. Full benchmark probe generation
without renderer/environment creation passes package/protocol/camera/table
checks on both nodes. CPU diagnostics explicitly omit renderer and task-data
checks; those remain required for real GPU B qualification.

The real failed runs' census.json already contains valid native renderer and
task-data probes. Stock WorkerLoop sends probe only on the first pull of each
TCP connection. P2 MetaWorldConductorDriver validates payload.get('probe') or
{} on every pull. Subsequent omitted probes therefore fail the package check,
including final shutdown/idle pulls. Deployment versions/interpreter are correct.

[CPU replay](preflight/gpu_fix1_probe_pull_repro.json): valid first pull PASS,
probe-less same-peer second pull REJECTED with package mismatch. The strict
connection-aware patch proposal is in [Q_to_P2](../REQUESTS/Q_to_P2.md), GPU fix1.
Reuse only an accepted probe on the same attested TCP peer/endpoint; a first
pull/new peer missing a probe, explicit empty/mismatching probe and endpoint
change must still be rejected. Q has not patched the P-owned driver/validator.

Q's owned ProbeRecordingDriver now retains full pull_payload, probe presence,
raw received probe, previously accepted probe, TCP peer and exact error. It
writes rejected_worker_probes.json immediately and includes the rows in driver
and top-level slot receipts. A rejection stops the owned driver promptly;
mechanical slot PARTIAL names the rejection instead of waiting out the phase.

Evidence: [timan107 import](preflight/gpu_fix1_timan107_import.log),
[packages107](preflight/gpu_fix1_timan107_packages.json),
[full probe107](preflight/gpu_fix1_timan107_actual_probe.json),
[timan108 import](preflight/gpu_fix1_timan108_import.log),
[packages108](preflight/gpu_fix1_timan108_packages.json),
[full probe108](preflight/gpu_fix1_timan108_actual_probe.json).
The refreshed deployment import-probe now additionally records executable,
package versions, full CPU-checkable benchmark probe and validation scope.

### Port release and regressions

Every generated job includes port_wait_seconds60. Live Q admission waits for
its policy/driver/backend ports; H100 dispatch also waits for local relay23158.
SO_REUSEADDR plus bind/listen checks permit normal closed-socket reuse while
still refusing active listeners. Wait is clamped by the owned slot deadline,
records initial busy ports/elapsed/status in receipt.json, and never signals a
port holder. A timeout prints PARTIAL with gpu_executed=false. The reserved-lane
allowlist excludes formal23151, I01823180 and protected H100 eval23254/55/58/59.

[test_gpu_fix1.py](test_gpu_fix1.py) proves delayed listener release, bounded
live-listener timeout without termination, closed-socket reuse, protected-lane
rejection, and raw missing/changed-probe preservation. Existing checks for
owned PID isolation, deadlines, canonical tables, byte transport and capture
joins remain green. [Q/default-off](preflight/gpu_fix1_q_sealed_pi.xml):32 PASS;
[current P2 conductor](preflight/gpu_fix1_p2_current.xml):46 PASS. The separate
CPU replay still exposes P2's missing subsequent-pull test/implementation gap.

### Refreshed CPU readiness and deployments

[CPU_SLOTS](receipts/CPU_SLOTS.json): every34 wrapper rerun under
`/data/mw_finetune/qualification/preflight/gpu_fix1_slots_pi`,30 PASS/four expected
PARTIAL/no FAIL, all gpu_executed=false. Four partials are retired Fast B2×2
and unpublished Fast16001×2. [GPU_FIX1](receipts/GPU_FIX1.json) is the final index.
Eight core asset seals and eight driver plans PASS; published GR60000's existing
seal and both native-A GPU receipts remain available. Source/spec612 hashes
match;30 Python files parse and34 wrappers pass bash -n.

P4's pinned --load-spec/--observations API landed during this fix. DP's source
identity changed, so Q sealed fresh `dp_libero_gpu_fix1_v1` locally and
`dp_mw_gpu_fix1_v1` on H100. Both actual CPU probes pass13/skipped0 and Q's
exact weight/ModelSpec gate. LIBERO uses original recorded paired256; MW uses
formal100k diagnostic=False. [DP_NATIVE_CPU](receipts/DP_NATIVE_CPU.json) records
the exact pinned proofs. DP current/final100000 slots remain on
`/data/mw_finetune/dp/mt50_policy_100000`. No checkpoint weight copy was made.

P6 CPU/file installer and rich import checks passed on both nodes:
[gpu_fix1_deployment_pi14](preflight/gpu_fix1_deployment_pi14/WORKER_DEPLOYMENT.json).
Both use `/scratch/zixuans8/mw_r17/qualification/payload_e67cdb2cdf26ef6cc3c644449819842ba92ca1f345ee783c19898a157f698df7`.
Final H100 runtime [gpu_fix1_deployment_pi15](preflight/gpu_fix1_deployment_pi15/RUNTIME_DEPLOYMENT.json)
SHA `ba3020c40031e155842d67e8f620a69f99595d3a3ebf137377592b63e38ec26c`.
Formal GR eval/timan107 r17_groot and H100 trainings/evals were untouched.

### P2-authorized π parity integration

[P2_to_Q](../REQUESTS/P2_to_Q.md) landed during final Q verification, satisfying
the coordinator's condition for touching pi_native.py. P2's one actual GPU
diagnostic identified whole-call compilation differences and exact parity
against R16's eager serving path. Q ran no GPU diagnostic and has not rewritten
the historical G1 A FAIL or claimed a corrected GPU PASS.

Q consumes P2's independent r16_reference and assert_native_parity. Normalized
and wire bytes, debug bytes and native suffixes stay exact; native NFE/shape/
finite gates and the common framework probe remain. Receipts explicitly record
r17.pi05.parity.r16_eager.v1, baseline_path, exact_r16, exact_debug and the alternate
comparison. K10 stock timing alternate stays exact. K2/K1 compiled-only diagnostic
max/mean limits are .08/.02 and .13/.03, rtol0, restricted to the exact pinned
checkpoint/norm/archive cohort. These bounds are not policy-quality tolerances.
The CPU preflight now checks profile admission before any GPU load.

Three new Q entry-point regressions exercise successful compiled discrepancy,
one-bit R16 reference rejection and out-of-bound compiled rejection. Existing
fixture composition still runs the actual framework probe. P2's independent
poison/profile tests were rerun with Q's regressions:
[gpu_fix1_pi_q.xml](preflight/gpu_fix1_pi_q.xml),18 PASS,0 skipped. The final Q/default-off
suite is32 PASS,0 skipped. All34 slot preflights were regenerated and rerun against
the refreshed612-file seal; no drift. π G1 A is first in the ready retry order.

### Ready to retry

```bash
bash exp/r17_integration/qualification/slot_G1_pi05_mw_A.sh --share-eval
bash exp/r17_integration/qualification/slot_G2_groot_mw_A_k2.sh --share-eval
bash exp/r17_integration/qualification/slot_G2_groot_mw_A_k1.sh --share-eval
bash exp/r17_integration/qualification/slot_G2_dp_libero_A.sh --share-eval
```

After DP-LIBERO A PASS, its B can run; LIBERO's standalone CPU-renderer shim
does not use the broken MW pull path. GR B/current+60000 remain conditional
on P2's follow-up-pull fix plus a fresh source snapshot/deployment and CPU
replay PASS. Their native A GPU gates already passed, so no redundant A retry
is requested while native model identities remain unchanged. GR C/G3/replicas
wait for matching B PASS. π G1 A is ready with the P2-authorized parity reference above;
π B also waits for the P2 pull fix; tail/G3 wait for A+B. H100 native A slots are prepared but wait for actual
training exit; all MW mechanics inherit the P2 pull-path prerequisite.

## GPU fix 2

2026-10-09 19:32 CDT. Current code consumed after P2's connection-aware
worker-probe delivery. Q completed CPU/tether preparation; only the coordinator
runs GPU slots. No GPU command/query, simulator environment construction,
foreign PID/tmux operation, P-owned source edit, checkpoint weight copy,
commit/push or protected-tree edit occurred.

### Finding → change → test → status

| Finding | Q change / handling | Actual check | Status |
|---|---|---|---|
| Worker subsequent-pull rejection | Consume P2 GPU fix2 driver; strict worker/transport peer/endpoint reuse. Q rejection logger remains active. | Fresh deployed timan107 and timan108 probes: first/full, second/absent, terminal/absent PASS; seven strict negative cases per node rejected with raw evidence. P2 conductor suite includes real stock TCP/WorkerLoop shutdown tests. | CPU PASS; historical B PARTIAL remains preserved for retry. |
| Runtime drift after owner edit | Full worker reinstall/reverification and H100 runtime refresh, then gen_slots.py and final runtime reseal. Source-drift guard remains enabled. |614 source/spec hashes match,33 Python files parse,34 wrappers bash -n; all34 CPU preflights rerun. | PASS; previously deferred K2/K1/DP-LIBERO A can retry. |
| Native code-identity changes | Audit exact native dependency hashes for nine unique published asset stores, including GR60000. Driver is outside those model dependency sets. | Local6/H1003 source-identity audits PASS; no weight load for audit. Existing manifest/ModelSpec/library seals retained. Both fresh pinned DP CPU probes13/0 match exact sealed weight/ModelSpec; MW remains diagnostic=False. | PASS; only runtime source seal renewal required. |
| Legacy/default-off and Q integration | New Q probe-pull replay regression; retain port wait, owned PID/deadline, parity/capture/config checks. | Q/default-off33 PASS; current conductor73 PASS; zero failed/skipped. Disabled-validator stock equality and real TCP witnesses are included. | PASS, CPU only. |

Replay sources and evidence:
[probe_pull_replay.py](probe_pull_replay.py), [test_gpu_fix2.py](test_gpu_fix2.py),
[timan107 replay](preflight/gpu_fix2_probe_pull_repro107.json),
[timan108 replay](preflight/gpu_fix2_probe_pull_repro108.json),
[import/deployment](preflight/gpu_fix2_deployment1/WORKER_DEPLOYMENT.json).
CPU metadata probes explicitly omit renderer/task-data checks; actual B slots
still require their full native worker attestation. First missing probe, new
peer, new worker, explicit empty/null/modified probe and changed endpoint remain
rejected; no package or renderer check is weakened.

### Current deployments and seals

Worker payload on both nodes remains the exact client-only archive:
`/scratch/zixuans8/mw_r17/qualification/payload_e67cdb2cdf26ef6cc3c644449819842ba92ca1f345ee783c19898a157f698df7`.
P2's changed driver is server-side and outside the worker allowlist, so its SHA
does not change. P6 deployment was nevertheless rerun for timan107/timan108;
the installed files and both fresh rich import probes PASS with the required
metaworld3.0.0/mujoco3.14.0/gymnasium1.3.0/runtime3.11.15.

Final H100 runtime: `/data/mw_finetune/qualification/runtime_q1`,
[deployment2](preflight/gpu_fix2_deployment2/RUNTIME_DEPLOYMENT.json), SHA
`6bf27f857c78a981e98d11fc687677810e9b6e1adcf8ad9e3968e0cca1037847`.
The local runtime uses the same renewed614-file source/spec snapshot. No drift.
[ASSET_IDENTITY_REVIEW](receipts/ASSET_IDENTITY_REVIEW.json) records why the native
stores need no replacement. [DP_NATIVE_CPU](receipts/DP_NATIVE_CPU.json) contains
the renewed pinned13/0 checks; DP current/final100000 remain fixed at
`/data/mw_finetune/dp/mt50_policy_100000`. DP-LIBERO keeps brandonyang30k for
mechanics and original recorded256 paired views. All teacher/library stores
remain sealed and explicitly synthetic. All MW recipes use periodic_gap and
canonical MT50 ordered_tasks.

[CPU_SLOTS](receipts/CPU_SLOTS.json):30 PASS/four expected PARTIAL/no FAIL under
`/data/mw_finetune/qualification/preflight/gpu_fix2_slots`. Expected partials are
two retired Fast C/B2 and two unpublished Fast016001 variants. Native GPU work
is not inferred from CPU preparation PASS. Eight core seals and eight plan-only
driver plans PASS; the ninth published GR60000 seal passes source audit and its
individual slot preflight. [Q/default-off](preflight/gpu_fix2_q.xml):33 PASS;
[conductor](preflight/gpu_fix2_p2.xml):73 PASS; counts overlap with owner suites
and should not be added to prior round counts.

### Ordered exclusive4090 queue

The first six commands are ready now from the coordinator checkout. Use fresh
wrapper-generated run directories; preserve the prior receipts. Every live
wrapper retains the890s hard bound and60s owned-port wait. No --share-eval is
needed while this lane is exclusive.

```bash
bash exp/r17_integration/qualification/slot_G1_pi05_mw_B.sh
bash exp/r17_integration/qualification/slot_G2_groot_mw_B.sh
bash exp/r17_integration/qualification/slot_G2_groot_mw_final60000_B.sh
bash exp/r17_integration/qualification/slot_G2_groot_mw_A_k2.sh
bash exp/r17_integration/qualification/slot_G2_groot_mw_A_k1.sh
bash exp/r17_integration/qualification/slot_G2_dp_libero_A.sh
```

π G1 A now has a real coordinator PASS at
`/data/mw_finetune/qualification/runs/slot_G1_pi05_mw_A_20261009_191908_601423/receipt.json`.
Current GR A and GR60000 A also retain real PASS. None is rerun by Q.

| Follow-up order | Slot | Dependency |
|---:|---|---|
| 7 | `slot_G2_dp_libero_B.sh` | After DP-LIBERO A PASS |
| 8 | `slot_G2_groot_mw_C_B2.sh` | After current GR B PASS |
| 9 | `slot_G2_groot_mw_R_replicas.sh` | After current GR B PASS |
| 10 | `slot_G2_dp_libero_R_replicas.sh` | After DP-LIBERO A+B PASS |
| 11 | `slot_G3_pi05_mw.sh` | After π B PASS; scoped PARTIAL |
| 12 | `slot_G3_groot_mw.sh` | After current GR B PASS; scoped PARTIAL |
| 13 | `slot_G3_dp_libero.sh` | After DP-LIBERO A+B PASS; scoped PARTIAL |
| 14 | `slot_G1_pi05_mw_C_tail.sh` | After π B PASS; scoped PARTIAL |


A failure stops its dependent branch. The six ready slots are independent of
one another's pending B results because their required A receipts already PASS
or they are independent native A probes. G3/tail intentionally remain scoped
PARTIAL rather than full advanced/dev/test/semantic qualification.

### H100 wait and later ordered queue

[CPU census](preflight/gpu_fix2_h100_census.json) confirms Fast-WAM and DP-LIBERO
training processes active. All H100 GPU candidates wait for both to actually
exit; DP-LIBERO's ~20:30 finish alone cannot admit Q while Fast-WAM runs toward
~22:15. The live CPU process gate rejects either training before any GPU query
or model load. No training overlap is authorized by this handback.

1. `slot_G2_fastwam_mw_A.sh` — After both trainings exit; sanity004000
2. `slot_G2_fastwam_mw_B.sh` — After matching A PASS
3. `slot_G2_fastwam_libero_A.sh` — After both trainings exit; exclusive two-view probe
4. `slot_G2_fastwam_libero_B.sh` — After matching A PASS
5. `slot_G2_dp_mw_final100000_A.sh` — After both trainings exit; formal mt50_policy_100000
6. `slot_G2_dp_mw_final100000_B.sh` — After matching A PASS
7. `slot_G2_fastwam_mw_R_replicas.sh` — After Fast-MW A+B PASS; exclusive
8. `slot_G2_dp_mw_R_replicas.sh` — After DP-MW A+B PASS
9. `slot_G3_fastwam_mw.sh` — After Fast-MW A+B PASS; scoped PARTIAL
10. `slot_G3_fastwam_libero.sh` — After Fast-LIBERO A+B PASS; scoped PARTIAL
11. `slot_G3_dp_mw.sh` — After DP-MW A+B PASS; scoped PARTIAL
12. `slot_G2_fastwam_mw_final16001_A.sh` — After both trainings exit, immutable publication and fresh CPU seal
13. `slot_G2_fastwam_mw_final16001_B.sh` — After final16001 A PASS


DP-MW current A/B names are compatibility aliases of the same formal100k
checkpoint; avoid duplicating passed final100000 native/mechanical work.
Fast final016001 waits for immutable publication and CPU sealing; never use a
mutable trainer root. Fast C/B2 slots remain retired CPU-only PARTIAL.

H100 shared storage measured95.836GB, Q directory0.146GB;13GB save reserve
remains within the120GB shared limit and Q is below its5GB allocation.
No weights were copied or save/train directories modified. Existing training
PIDs, eval ports and timan driver sessions were untouched. Future owner source
edits will correctly defer slots until another source refresh.

Resolved seam: [Q_to_P2](../REQUESTS/Q_to_P2.md), GPU fix1, consumed via
[P2 GPU fix2](../handbacks/HANDBACK_P2.md) and [P2_to_Q](../REQUESTS/P2_to_Q.md).
[PLAN.md](PLAN.md) and [GPU_FIX2](receipts/GPU_FIX2.json) are the current release index.

## GPU fix 3

**CPU preparation/deployment PASS; actual GPU retries pending coordinator.**
This section and [GPU_FIX3](receipts/GPU_FIX3.json) supersede prior readiness.
All changes are Q-owned qualification code, generated specs/docs/receipts,
private Q assets/runtime deployments, and an appended Q-to-P4 request.
No frozen stock transport, P-owned source, training/eval/tmux process or
pi_native.py was changed by Q in this fix. No GPU slot was run by Q.

The historical GR B run
`/data/mw_finetune/qualification/runs/slot_G2_groot_mw_B_20261009_193927_724271/`
completed both accepted episodes, but its remote saved receipt shows worker0
exit0 and worker1 exit-15. Worker1's log records connection-refused retries after
the driver closes its listener. Supervisor2287 remained waiting for that worker;
it was correctly terminated by its owned cleanup. The missing final shutdown
pull is reproduced with real stock TCP handlers/WorkerLoop and a fake CPU
episode runner. The small closed physical tree (263 files, about2.7MB) does not
explain the timeout. Evidence: [remote repro](preflight/gpu_fix3_remote_repro.json).
Historical PARTIAL receipts remain unchanged.

| Finding | Change | Regression / verification | Status |
|---|---|---|---|
| MW B accepted episodes but tether_workers overruns | Q driver keeps accepted sockets alive for at most35s after scheduler completion, lets each stock final pull receive MSG_SHUTDOWN, records both shutdowns, and closes only its own unfinished sockets on a failed drain | Actual socket/stock WorkerLoop test delays the final pull; both workers exit normally. A missing worker bounds the drain and remains PARTIAL | CPU PASS; retry π B and GR B/current+60000 |
| Supervisor/receipt must distinguish clean exit from deadlines | Explicit normal_exit requires both own child codes0. Owned P6 snapshot child has a recorded PID/birth/argv, a20s bound within the500s supervisor budget, and bounded own-PID cleanup. Closed locks/logs precede the archive | Actual supervisor main with two fake CPU children; actual P6 archive installer; hanging children remain PARTIAL. Same fake-child proof on both deployed timan nodes exits in about0.2s | CPU/deployment PASS |
| Accepted journals cannot salvage interrupted workers | MW B validates driver drain PASS, remote normal_exit/two0 codes/no forced child, verified snapshot with no partials, and exactly two expected accepted/error-free journal rows before existing physical/capture checks | Clean fixture PASS; worker timeout/nonzero exit, missing drain or missing journal row each remain PARTIAL | CPU PASS; real deadlines still PARTIAL |
| P4's new B1 release changes DP native source/ModelSpec identity | Fresh DP LIBERO and formal MW assets, load pins, synthetic stores/libraries/fits; require real_production_B1_only, max_batch_size1 and exact B1 queue bytes. All batch capabilities must remain unsupported; labelled B2 diagnostics confer no batch/parity certificate | Both actual pinned CPU probes strict475,13 checks/zero skips; dynamic-count criterion and damaged B1 gate regression | CPU PASS; new-release GPU A required |
| Refreshing runtime alone could leave stale native assets | Preflight audits the selected native source seal before GPU admission, and DP also compares dependencies from the actual serving Python interpreter with the manifest | Changed native code still returns PARTIAL after runtime snapshot refresh; mismatched dependency versions reject | CPU PASS |

[test_gpu_fix3.py](test_gpu_fix3.py) supplies12 cases for the drain, supervisor,
closed journal and identity/dependency checks. The supervisor fixtures substitute
two short-lived CPU children; they do not instantiate a simulator, policy or
GPU renderer. Normal workers receive no shutdown signal from the supervisor.
The stock MSG_SHUTDOWN response remains responsible for their normal exit.
Original port-release waits, all phase deadlines and the outer≤890s watchdog
remain enforced. A real deadline is never promoted to PASS by completed rows.

### Current deployment and CPU evidence

- Full payload refresh: [worker deployment](preflight/gpu_fix3_deployment1/WORKER_DEPLOYMENT.json).
  Both timan107/timan108 install and --import-probe PASS using the deployed
  `/scratch/zixuans8/mw_r17/envs/sim/bin/python`. Both report MetaWorld3.0.0,
  MuJoCo3.14.0 and Gymnasium1.3.0. The new Q supervisor is in payload SHA
  `46d82392042773e16cb45f3ac635506e6b90453829841105b45ed158fa89d8ae`;
  destination on each node is
  `/scratch/zixuans8/mw_r17/qualification/payload_46d82392042773e16cb45f3ac635506e6b90453829841105b45ed158fa89d8ae`.
- Both actual deployed supervisor CPU fixtures PASS:
  [timan107](preflight/gpu_fix3_timan107_supervisor_cpu.log),
  [timan108](preflight/gpu_fix3_timan108_supervisor_cpu.log).
  Fresh probe-pull replay PASS on both nodes for first/second/terminal pulls
  plus seven strict rejection cases/node:
  [107](preflight/gpu_fix3_probe_pull_repro107.json),
  [108](preflight/gpu_fix3_probe_pull_repro108.json).
- Final H100 runtime `/data/mw_finetune/qualification/runtime_q1`,
  SHA `7c48284f86e11d72f0ed033d0d46f068978228c77467ab01f90af7d1e20ee628`:
  [runtime deployment5](preflight/gpu_fix3_deployment5/RUNTIME_DEPLOYMENT.json).
  All617 archived source/spec files match current code, including P2's driver
  and P4's completed B1 release. Every wrapper was regenerated with gen_slots.py.
- All34 CPU preflights: **30 PASS, four expected PARTIAL, zero FAIL**;
  [slot receipts](receipts/CPU_SLOTS.json),
  [log](preflight/gpu_fix3_slots_release.log). The four PARTIALs are retired
  Fast-WAM C/B2×2 and unpublished Fast016001 A/B×2. Every receipt records
  gpu_executed=False. G3/tail CPU preflight PASS is admission evidence only;
  their actual slots retain the documented scoped PARTIAL result.
- Eight current asset inventories/config validations/plan-only drivers PASS;
  [refresh](preflight/gpu_fix3_plans_release.log),
  [asset seals](receipts/ASSET_SEALS.json), [driver plans](receipts/DRIVER_PLANS.json).
  Nine unique published native stores (including GR60000) pass the
  [identity audit](receipts/ASSET_IDENTITY_REVIEW.json); final source drift is zero.
- **45 Q/default-off PASS,73 conductor PASS, zero failures/errors/skips**:
  [Q XML](preflight/gpu_fix3_q_complete.xml),
  [conductor XML](preflight/gpu_fix3_p2.xml).
  Native paired256/wrist provenance, canonical MT50 ordered_tasks, explicit
  periodic_gap/target_ir=None, dynamic DP check counts and retired Fast B2
  regressions remain passing. Native model equality tolerances were not widened.
- Fresh sealed DP LIBERO asset is
  `/data/mw_finetune/qualification/assets/dp_libero_gpu_fix3_v1`;
  fresh H100 DP MW asset is
  `/data/mw_finetune/qualification/assets/dp_mw_gpu_fix3_v3`.
  LIBERO preserves the original paired256 NPZ and P4's current model pin
  `sha256:67331d0af13c19fc3e993e5a89f81a1c89469e68d66049f0ec26122de203a022`.
  DP-MW current/final100000 stay on `/data/mw_finetune/dp/mt50_policy_100000`,
  diagnostic=False, with formal checkpoint100000 provenance. Each actual
  pinned CPU native probe passes13/0 and exact ModelSpec/weight gates:
  [DP receipts](receipts/DP_NATIVE_CPU.json). P4's owner-authorized B1-only
  choice resolves the appended Q_to_P4 batch-failure request; no universal
  batch tolerance was added and no new DP GPU PASS is claimed.

P4 finished while the first refresh was in progress. Those intermediate source
drift results are retained; only the fresh release run above is the final index.
H100 DP sealing first lacked h5py, then a seal made with the main interpreter
failed the actual serving interpreter's model pin. Both are retained as
intermediate generations. The final v3 seal uses the actual DP Python island
with existing Q-private python_deps312 for h5py, and its pinned probe passes.
The new dependency-version preflight makes that mismatch visible before GPU use.
All synthetic teacher/library products remain explicitly labelled synthetic;
no real teacher, quality/SR or MW IR-price certificate is inferred.

### Exact 4090 retry order

These four independent first rows are ready now, serially on the exclusive4090:

```bash
bash exp/r17_integration/qualification/slot_G1_pi05_mw_B.sh
bash exp/r17_integration/qualification/slot_G2_groot_mw_B.sh
bash exp/r17_integration/qualification/slot_G2_groot_mw_final60000_B.sh
bash exp/r17_integration/qualification/slot_G2_dp_libero_A.sh
```

Then follow [PLAN.md](PLAN.md)'s conditional table: DP-LIBERO B after its fresh
A PASS; GR C/B2 and replicas after current GR B PASS; DP replicas after A+B;
π/GR/DP G3 and π tail after their named prerequisites, preserving their scoped
PARTIAL limits. Stop only the dependent branch when its prerequisite fails.
π A, GR A/current+60000, GR A_k2 and GR A_k1 already passed on GPU; their native
identities were unchanged, so those five are excluded from the new retry queue.
The historical DP-LIBERO A FAIL and three B PARTIALs are not reclassified.

### H100 waiting order and budget

All H100 GPU work waits for BOTH actual training exits. Read-only
[CPU census](preflight/gpu_fix3_h100_census.json) still sees Fast-WAM and
DP-LIBERO training; estimates remain22:15/20:30 CDT, not admission grants.
The ready-checkpoint order is Fast-MW A→B, Fast-LIBERO A→B, formal
DP-MW final100000 A→B, then their conditional replicas/G3. Fast016001 A→B
also requires immutable publication and fresh CPU sealing. DP current A/B
are aliases of final100000 and need not duplicate a passed identical checkpoint;
Fast C/B2 remains retired. Exact13-row queue is in PLAN.md and GPU_FIX3.json.
No H100 training/eval server or timan108 driver session was touched.

H100 shared tree measured96.944GB, Q space0.147GB; the13GB Fast-WAM save reserve
fits within120GB. This refresh added about1.46MB to Q's directory, below5GB;
no checkpoint weights were copied. Future source changes correctly defer the
affected slots until fresh CPU deployment/native resealing where required.


## GPU fix 4

**CPU preparation/deployment PASS; Q ran no GPU slot.** This section and
[GPU_FIX4](receipts/GPU_FIX4.json) supersede earlier readiness. Source changes
are confined to Q's qualification directory. Running evals/training and their
workers, tmux sessions, locks and ports were untouched. No P-owned/frozen
transport, model math, pi_native.py, commit or push was changed by Q.

The third-pass MW B failures came from tether's file-transfer allow roots
`[/home /tmp /srv]`: the worker archives were correctly closed, but the pull
source was under /scratch. The read-only repro is
`slot_G1_pi05_mw_B_20261009_201506_944954/worker_pull.log`. The audit also found
H100 relay pushes to /data and remote LIBERO pulls from /data would be refused.
`tether exec` CPU reads/copies remain valid; file-transfer restrictions apply
at the actual push/pull endpoints.

| Finding | Change | Regression / real CPU proof | Status |
|---|---|---|---|
| Both timan MW tar sources were outside allow_roots | Closed producer archives are copied into unique `/tmp/r17_Q_qualification/<run>/workers_<sha>_<uuid>.tar` stages. Published transfer receipt carries exact path/SHA/run/kind. Launcher pulls that attested path | Actual deployed fake CPU worker supervisor → real tether pull → actual P6 install on timan107 and timan108 lanes | CPU PASS |
| Stage cleanup must follow verification | Exact source stage removed only after SHA/P6 installation and closed worker/journal checks. Ownership marker, path, run, kind and SHA must agree; original archive retained | Corrupt/timed-out pulls never call cleanup; wrong SHA/run/kind/root reject and retain archive; successful actual transfers remove only their stages | CPU PASS |
| H100 relay pushed into denied /data | Coordinator checks pulled SHA, pushes to allowed H100 /tmp stage, then bounded CPU SHA-copy into the exact durable Q run. Incoming stage removed only after verified durable copy; producer stage after P6 verification | Actual H100 HTTP request → coordinator broker → timan108 → allowed H100 push/CPU install/P6 verification → source cleanup | CPU PASS |
| H100 remote LIBERO tar was also under /data | H100's remote worker uses `--transfer-stage` and a unique allowed libero tar stage. Local LIBERO path remains direct. Remote source cleaned after P6 install | Actual H100 HTTP relay transfer from weilandserver with a labelled synthetic CPU LIBERO archive; real push/install/P6/cleanup | CPU PASS |
| Added relay hops need a sufficient bounded phase | H100 pulls get125s (40 pull +15 mkdir +40 push +25 install +5 margin); local pull remains85s. Cleanup15s tether/20s phase; existing840s work and≤890s hard watchdog unchanged | Local/H100 clean, corrupt and timeout regression cases exercise the phase bound; real timeouts remain PARTIAL | CPU PASS |

New [archive_stage.py](archive_stage.py) is stdlib/Python3.8 compatible for the
LIBERO island and included in both worker payloads. It caps archives at512MiB,
streams copies/hashes, rejects symlinked or mismatched stages, never overwrites
an existing archive and uses a fresh UUID for every stage. Stage markers remain
as cleanup receipts. Failure retains the exact stage for recovery; no broad
file/process deletion is used. Original source archives remain with their runs.
[Test GPU fix4](test_gpu_fix4.py) has17 cases, including SHA corruption,
failed/timeout retention, exact-run ownership and H100 relay path validation.
No tether pull from /scratch or /data, or push to H100 /data, remains in Q's
archive path. Deployment pushes already use allowed /tmp stages and passed.
[Transfer path audit](receipts/TRANSFER_PATH_AUDIT.json) covers every Q transfer.

### Final CPU/deployment evidence

- **62 Q/default-off PASS,73 conductor PASS**, zero failures/errors/skips:
  [Q XML](preflight/gpu_fix4_q_final.xml), [conductor XML](preflight/gpu_fix4_p2.xml).
  Existing shutdown-drain, default-off, original256 paired views/wrist,
  canonical MT50, periodic_gap/target_ir=None, DP dynamic-count/B1 and retired
  Fast-WAM B2 regressions remain passing.
- **Four real tether CPU transfers PASS**, all SHA/P6 checked with exact stage
  cleanup: [TRANSFER_CPU](receipts/TRANSFER_CPU.json),
  [log](preflight/gpu_fix4_transfer_cpu.log). Timan proofs substitute two short
  CPU children and instantiate no simulator/policy/renderer. H100 LIBERO proof
  uses a synthetic archive; it is a transfer certificate, not a new episode.
- **34 final preflights:30 PASS/four expected PARTIAL, zero FAIL**, all
  gpu_executed=False: [CPU_SLOTS](receipts/CPU_SLOTS.json),
  [final log](preflight/gpu_fix4_slots_final.log). The four are retired Fast
  C/B2×2 and unpublished Fast016001 A/B×2. All wrappers regenerated by gen_slots.py.
- Eight inventories/config validations/plan-only drivers PASS:
  [final refresh](preflight/gpu_fix4_plans_final.log),
  [asset seals](receipts/ASSET_SEALS.json), [driver plans](receipts/DRIVER_PLANS.json).
  Nine native source identities match; no native reseal needed this round:
  [identity review](receipts/ASSET_IDENTITY_REVIEW.json). DP pinned CPU probes
  rerun13/0 each, exact weight/ModelSpec and B1 gate:
  [DP_NATIVE_CPU](receipts/DP_NATIVE_CPU.json). Formal DP-MW current/final100000
  still select `/data/mw_finetune/dp/mt50_policy_100000`, diagnostic=False.
- Full worker refresh/import probes on both nodes PASS:
  [deployment1](preflight/gpu_fix4_deployment1/WORKER_DEPLOYMENT.json).
  Payload SHA `b753918aa5cae303d3cbe6aa17a2aaa28f8e4d67438f9fc0ceab7f0afecf6121`;
  destination on each node:
  `/scratch/zixuans8/mw_r17/qualification/payload_b753918aa5cae303d3cbe6aa17a2aaa28f8e4d67438f9fc0ceab7f0afecf6121`.
  Both report MW3.0.0/MuJoCo3.14.0/Gymnasium1.3.0. Probe-pull replays PASS for
  first/second/terminal plus seven strict rejection cases/node:
  [107](preflight/gpu_fix4_probe_pull_repro107.json),
  [108](preflight/gpu_fix4_probe_pull_repro108.json).
- Final H100 runtime `/data/mw_finetune/qualification/runtime_q1`,
  SHA `81fc77efca3fdadc935a7e93a17afc9dcad6843b7ac676f67d2fd44933f48b2b`:
  [deployment4](preflight/gpu_fix4_deployment4/RUNTIME_DEPLOYMENT.json).
  All619 archived source/spec files match current code. An owner edit to
  exp/mw_finetune/dp/serve_libero.py after the first seal was caught; runtime
  was refreshed and every final preflight repeated. That helper is outside
  native DP math identities, so existing A/B native model seals remain valid.

### CPU recovery of the three completed MetaWorld B runs

The original closed tar from each third-pass run was staged/pulled with the new
allowed-root helper. Local artifacts were copied into fresh Q preflight paths;
P6 installed the verified tar there. Closed worker/shutdown/journal checks and
**the full native capture, accepted/control identity join, execution ACK, cost
ledger and analysis audit all PASS** for all three, two accepted episodes each:
[replay receipt](preflight/gpu_fix4_real_replays.json),
[log](preflight/gpu_fix4_real_replays.log).

| Historical run | CPU recovered audit |
|---|---|
| slot_G1_pi05_mw_B_20261009_201506_944954 | PASS |
| slot_G2_groot_mw_B_20261009_201910_969563 | PASS |
| slot_G2_groot_mw_final60000_B_20261009_202027_977884 | PASS |

These are CPU recovery receipts for existing GPU-produced evidence. No new
policy call/episode/GPU workload ran, and none of the coordinator's original
PARTIAL receipts was rewritten. Original archive sources remain intact.
Synthetic retrieval donors still confer mechanics evidence only; no500-episode
quality/SR or measured MW IR-price claim is introduced.

### Ready order and held lane

CPU-ready serial queue, **when GPU admission permits**; preserve all current
evals and their workers. Use the default GPU-lock behavior:

```bash
bash exp/r17_integration/qualification/slot_G1_pi05_mw_B.sh
bash exp/r17_integration/qualification/slot_G2_groot_mw_B.sh
bash exp/r17_integration/qualification/slot_G2_groot_mw_final60000_B.sh
bash exp/r17_integration/qualification/slot_G2_dp_libero_R_replicas.sh
```

GR C/B2 and GR replicas follow current GR B PASS. π/GR G3 and π tail follow
their named B prerequisites and retain scoped PARTIAL outcomes. DP-LIBERO A
(`20261009_202155_987088`) and B (`20261009_202215_989238`) already GPU PASS
on the B1 release; do not repeat them. G3_dp_libero's scoped PARTIAL
(`20261009_202751_1023021`) was observed as intended and is excluded from this
retry queue. The earlier five native π/GR passes also remain valid. The complete
nine-row 4090 pending table is in [PLAN.md](PLAN.md).

H100 order remains Fast-MW A→B, Fast-LIBERO A→B, formal DP-MW final100000 A→B,
then conditional replicas/G3, and Fast016001 A→B after immutable publication
and CPU seal. DP current A/B are compatibility aliases; Fast C/B2 remains
retired. [Latest CPU census](preflight/gpu_fix4_h100_census.json) sees Fast-WAM
training and no DP-LIBERO training rows; Fast-WAM's actual exit and normal GPU
admission are still required. No H100 GPU work was started.

H100 shared data measured98.783GB, Q space0.147GB;13GB transient save reserve
fits within120GB. Q growth since GPU fix3 is about0.359MB, below5GB. No weights
were copied. Current release index: [GPU_FIX4](receipts/GPU_FIX4.json), mirrored
in [VERIFICATION](receipts/VERIFICATION.json). No cross-package fix is pending
from this transfer issue; future source edits correctly require another refresh.

## Formal launcher

Ready: [ops/formal_eval.sh](../ops/formal_eval.sh) and
[formal_eval.py](../ops/formal_eval.py), with the full operator description in
[FORMAL_EVAL.md](../ops/FORMAL_EVAL.md). This uses the R17 conductor for individual
episodes and N resident R17 model processes behind the unchanged stock replica
proxy. Each resident executes B1; independent resident processes serve together.
DP replicas receive `--cuda-graph` and verify both E's native and R17 physical
actions against eager execution with `torch.equal` before listening. Cache,
retrieval, recipes, debug capture and videos are off. Default local DP is N12,
W24 and14 GiB; startup records and checks owned physical GPU memory.

Exact coordinator commands, executed from `/home/weiland/projects/openpi`:

1. DP-LIBERO formal400, original paired256 views,40 tasks ×original init0..9:

```bash
exp/r17_integration/ops/formal_eval.sh --model dp_libero --checkpoint /data/mw_finetune/dp/libero_policy_100000 --replicas 12 --workers 24 --gpu cuda:0 --memory-budget-gib 14 --run
```

2. DP-LIBERO official2000, a separately sealed protocol using original init0..49:

```bash
exp/r17_integration/ops/formal_eval.sh --model dp_libero --checkpoint /data/mw_finetune/dp/libero_policy_100000 --replicas 12 --workers 24 --gpu cuda:0 --memory-budget-gib 14 --official-50 --run
```

Choose one local LIBERO campaign at a time. Both default to public23160,
backends23161–23172 and conductor23179 on `ziyanglin.com`.

3. DP-MW canonical MT50 500, final100000 H100 export, timan108 workers:

```bash
exp/r17_integration/ops/formal_eval.sh --model dp_mw --checkpoint /data/mw_finetune/dp/mt50_policy_100000 --server-node h100 --worker-node timan108 --replicas 12 --workers 24 --gpu cuda:0 --memory-budget-gib 14 --driver-port 23178 --run
```

4. Fast-WAM MT50 500, H100 +timan108, after training ends and final016001 is published:

```bash
exp/r17_integration/ops/formal_eval.sh --model fastwam_mw --checkpoint /data/mw_finetune/fastwam/mt50_frozen_b128/policy_016001 --server-node h100 --worker-node timan108 --replicas 2 --workers 24 --gpu cuda:0 --memory-budget-gib 72 --port 23174 --driver-port 23177 --run
```

H100 policy endpoints use `149.165.153.233`; conductor endpoints remain on
`ziyanglin.com`. The separate conductor ports let the local and H100 campaigns
coexist. The two Fast residents budget72 GiB, so close the DP H100 pool before
starting that pool; Fast N1 with36 GiB can share when the coordinator admits it.
No H100 GPU job was started. H100 GPU admission remains the coordinator's
decision after the running training finishes. Fast final016001 appeared at the
end of preparation and its CPU seal/import/500-episode plan passed. Each
`--run` command regenerates its CPU seal and payload before starting processes.

The launcher also supports `pi05_mw`, `groot_mw`, and `fastwam_libero`, with
explicit replica/GPU/worker/checkpoint/budget parameters. Native full samplers
are DP100, π10, GR4 and Fast10. `--extra-workers N` adds the other timan node.
Workers use EGL device0, `HOME=/home/zixuans8`, and R16's client/src/root PYTHONPATH
order. LIBERO uses an isolated Python3.11 dependency copy of the original R16
LIBERO/robosuite source and assets; the original environments are untouched.
The trusted original NumPy init pools retain their original `torch.load`
semantics with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in the private worker env.
CPU preparation imports the complete worker and checks all40 pools have50 states.

CPU validation passed for every integrated bundle and both LIBERO trial counts:

| Bundle / cohort | CPU receipt under `/data/mw_finetune/eval/` | Episodes |
|---|---|---:|
| DP-LIBERO formal400 | `formal_dp_libero_formal400_preflight_final/PREFLIGHT.json` |400|
| DP-LIBERO official2000 | `formal_dp_libero_official2000_release/PREFLIGHT.json` |2000|
| DP-MW final100000 | `formal_dp_mw_mt50_release/PREFLIGHT.json` |500|
| Fast-MW sanity004000 | `formal_fastwam_mw_sanity_release/PREFLIGHT.json` |500|
| Fast-MW final016001 N2 | `formal_fastwam_mw_final016001_replicas2_release/PREFLIGHT.json` |500|
| GR00T-MW final60000 | `formal_groot_mw_preflight_final/PREFLIGHT.json` |500|
| π0.5-MW RLinf | `formal_pi05_mw_preflight_final/PREFLIGHT.json` |500|
| Fast-LIBERO released | `formal_fastwam_libero_preflight_final/PREFLIGHT.json` |400|

The current planner revalidated the plans with exact model, norm, feature,
layout, sampler, benchmark and canonical table bindings. Regression checks cover
400/2000/500 cohort enumeration, separate official50 partitions, reserved-port
collisions, selected-benchmark worker admission, stale-result SR fencing,
retry/error counts, PID reuse and all24 workers sharing the requested EGL device.
`qualification/formal_cpu_pytest_release_final.log` under `/data/mw_finetune/`
records20 passing CPU/qualification/legacy-default-off tests. The actual rejected
LIBERO probe from the initial smoke now passes CPU replay at
`/data/mw_finetune/eval/formal_dp_libero_smoke4_v4/PROBE_REPLAY.json`.

Final GPU smoke: `/data/mw_finetune/eval/formal_dp_libero_smoke4_final/`.
It ran4 complete original episodes (one per suite), accepted both workers and
exercised distinct backends23161/23162. Both replicas passed native +R17 CUDA-graph
self-checks and ModelSpec attestation; the proxy acknowledged the full bundle.
All177 accepted decisions had capture off, policy source, NFE100 and
stage calls(1,1,100). SR was spatial1/1, object1/1, goal1/1, libero_10 0/1; this
four-episode smoke is a mechanics check, not a formal40-task SR estimate.

Evidence files: `SERVER_ATTESTATIONS.json`, `PROXY_METADATA.json`,
`BUNDLE_ACK.json`, `GPU_MEMORY.json`, `BACKEND_EVIDENCE.json`,
`driver/worker_dispatch.jsonl`, `driver/journal.jsonl`, `driver/per_step.jsonl`,
`driver/DRIVER_RECEIPT.json`, `workers/timan107/TRANSFER_SHA256.json`,
`SUMMARY.json`, and `RECEIPT.json`. The remote archive was pulled from the exact
allowed `/tmp` stage, P6/SHA verified, then that stage was removed. Driver shutdown
drained both workers; receipts confirm every owned PID/pgid closed and all owned
ports released. The existing π0.5 port23150 remained live. Failed setup attempts
remain separate evidence directories; none accepted evaluation episodes.

Measured N2/W2 throughput was177 inferences /122.421s =1.446 infer/s including
dispatch, EGL simulation and shutdown drain. The two resident processes used
1798 MiB total. N12/W24 planning throughput from a linear extrapolation is about
8.7 infer/s, subject to shared GPU/simulator contention; N12 throughput is not
measured. Formal400/official2000/MT50 campaigns were not launched. Their outputs
go to `/data/mw_finetune/eval/<unique_run>/SUMMARY.json`, including per-suite SR,
the40-task mean, and errors/retries; incomplete cohorts remain PARTIAL.

Lifecycle ownership is recorded before work, checked against PID start ticks,
and cleaned by PID/pgid only. Remote supervisors expire unrenewed leases in120s.
Closed worker and H100 server snapshots use the SHA-checked `/tmp` transfer path.
The default full campaign bound is12h, each episode is bounded to30min with two
retries, and both limits are explicit flags. No `src/`, P-owned model, frozen
R16/harness, training, live eval or foreign tmux/process was modified; no commit,
push or broad process kill was used.

## Retried-error completion

Source confirmed: stock `src/openpi/conductor/driver.py:325–403` requeues retriable
errors and journals accepted terminal outcomes; rc=2 is defined by
`ops/formal_driver.py:145` for any PARTIAL receipt, including an incomplete shutdown
census. The launcher now accepts rc=2 only with every planned UID accepted exactly
once, no terminal errors, `all_done=true`, no rejected probes and error attempts
resolved by later terminal attempts. Other exit codes/incomplete journals remain
FAIL. Automatic aggregation retains retry/error counts and the raw driver receipt;
clean worker closure, SHA transfer and owned cleanup gates still apply. The generic
child monitor defers driver exits to this check, including the polling race.

CPU regression: 8 formal tests plus 2 default-off checks and 16 subtests pass.
Read-only replay of
`formal_dp_libero_formal400_20261009_220843_8dca4316` aggregates to PASS: 400/400,
zero terminal errors, 2 error attempts, 5 retries, full-arm evidence verified.
Evidence: `preflight/formal_retried_error_completion.json`. No campaign files,
running campaigns, processes or GPU state were changed.

## Payload refresh 10-09 night

Local refresh PASS: 651 current source/spec hashes match. Added and CPU-tested
`deploy_cpu --local-only` (optionally `--runtime-only`) so this refresh deploys
only timan107 and the local runtime seal; H100/runtime_q1 and timan108 are untouched.
GR00T's existing native seal still matches. DP required a fresh native manifest,
configs and explicitly synthetic stores at
`/data/mw_finetune/qualification/assets/dp_libero_night1009_234426`; its existing
immutable mechanics checkpoint is unchanged. Both replica specifications now
reference the current local release and timan107 SHA payload.

Verification: 25 CPU regression/default-off tests PASS; both final wrapper CPU
preflights, native source audits and plan-only drivers PASS; timan107 deployed
import/package probe PASS. Evidence is under
[payload_refresh_1009_night_20261009_234426](preflight/payload_refresh_1009_night_20261009_234426/).
GPU DP-LIBERO replicas PASS in 29.16s, with both resident servers exiting0 and
the owned proxy/proof processes closed:
`/data/mw_finetune/qualification/runs/slot_G2_dp_libero_R_replicas_night_refresh_20261009_234713/receipt.json`.
GR00T first caught D's concurrent Cosmos source change; the local runtime was
refreshed again and CPU preflight passed. Its GPU retry returned PARTIAL before
GPU admission (`gpu_executed=false`, no owned children): another library teacher
smoke had acquired the GPU/eval lock. That run and D/P6's work were left untouched.

Exact coordinator command, from `/home/weiland/projects/openpi`, once the 4090
and eval lock are exclusive again (no `--share-eval`):

```bash
bash exp/r17_integration/qualification/slot_G2_dp_libero_R_replicas.sh && bash exp/r17_integration/qualification/slot_G2_groot_mw_R_replicas.sh
```

DP already passed; to finish only the remaining GPU proof, run the second command.
Final CPU receipts and unchanged source census are `CPU_PREFLIGHTS_FINAL.json`
and `SOURCE_SNAPSHOT_FINAL.json` in the evidence directory above. Historical
PARTIAL receipts are preserved; subsequent source edits still require refreshing.

## G2 receipts bound for P6

Published separate, evidence-bound PASS receipts; both original coordinator and
H100 execution receipts are unchanged. P6 should use:

```bash
--g2-a-receipt /data/mw_finetune/qualification/bound_receipts/g2_bound_p6_20261010_000040/G2_A.json
--g2-b-receipt /data/mw_finetune/qualification/bound_receipts/g2_bound_p6_20261010_000040/G2_B.json
```

A's preserved native GPU PASS is tied to owned PID80365, its exact command/run
and clean exit. Its full ModelManifest has57 pins (weights/base, Wan, tokenizer,
norm and code); recomputed complete-map and code-submap digests match its spec.
The native sidecar SHA is
`d94143a3e0aec0bebe30214f3c2a479b737953bb377ded96f99b52f15973c60a`.
B's entire recorded resident spec equals that manifest's spec; its ready PID80420
matches the owned server command/run and clean exit, and both capture writers
closed/drained under that PID. Thus B's attested checkpoint identity binds the
same complete pin map, without substituting current assets. B's ready SHA is
`13fdb4985ab47c43e45f475ffa6a05083a133587256f17c0f5ad27d2d8077107`.
Each published wrapper carries `remote_receipt.model_manifest` plus original
receipt/evidence hashes and the explicit historical binding proof.

Going forward, `model_runtime` exports the loaded Fast-WAM adapter's actual
manifest in SERVER_READY/WebSocket metadata. `run_slot` attaches A's native
manifest and B's owned resident attestation (PID/spec/sidecar SHA verified) to
the PASS record. Missing or mismatched bindings fail closed. The A/B specs use
the separately deployed H100 runtime
`/data/mw_finetune/qualification/runtime_g2_bound_p6_20261010_000040_v2`;
existing runtime_q1, running campaigns and the4090 were not modified or used.

Verification:74 CPU regression/default-off/P6 gate tests PASS; both future A/B
execution-host CPU preflights PASS with GPU hidden. P6's actual G2 gate accepts
the two published receipts; its fresh teacher-pin/geometry/sampler comparison
still runs at campaign preparation. No GPU rerun was needed or executed.
[Evidence, hashes, gate and publication](preflight/g2_bound_p6_20261010_000040/)
and [release index](receipts/G2_BOUND_P6.json) preserve all checks and originals.

## Concurrent fleets on one node

Cause is proven by the timan107 kernel log: global OOM killed DP worker7,
PID419359, at **2026-10-10 10:20:15 CDT** (exit-9). The old fleet supervisor
treated any worker failure as terminal and SIGTERMed its other23 workers.
Its process receipt exited2 without an expired-lease reason; the launcher's
later cleanup found the supervisor already closed. The driver had167 accepted
episodes, zero terminal errors and zero rejected probes. No deployment overwrite,
shared lock failure or cross-campaign cleanup was evidenced. The two24-worker
fleets overlapped without a node RAM admission check.
[Preserved remote receipts and exact OOM lines](preflight/concurrent_fleets_20261010/DP_REMOTE_EVIDENCE.json)
and [local receipt/TETHER_CALLS/driver evidence index](preflight/concurrent_fleets_20261010/DIAGNOSIS.json)
record the diagnosis; original campaign files remain unchanged.

`formal_eval.py` now checks node RAM during CPU preparation, before any server
launch. `formal_workers.py` atomically rechecks under a node admission lock before
forking. Accounting reads only framework receipts and PID/birth-matched RSS,
including legacy live fleets and reservations for workers still starting.
Defaults reserve4096MiB per LIBERO worker,2048MiB per MW worker, plus
max(8GiB,10% of node/cgroup capacity) headroom; actual RSS is already charged in
MemAvailable. `--worker-memory-mib` exposes the reservation. Refusal records its
reason and spawns no workers. Shared admission/install locks are short preparation
locks; each fleet's evaluation lock, control receipts, leases and health checks
remain bound to that campaign.

A crashed worker now gets at most2 replacements (`--worker-restart-limit`),
with up to120s waiting for replacement memory and the campaign deadline still
enforced. Healthy siblings continue; the conductor's existing connection-drop
handling requeues the interrupted episode with its attempt fence. All worker
generations and exit codes are retained. Exhausted retries or real deadlines
remain PARTIAL at fleet level and FAIL at campaign level. Failure receipts now
include the exact owned remote fleet receipt and worker failure details instead
of just `remote worker fleet failed: timan107`. Cleanup signals only owned
Popen children and the existing receipt/birth-bound process groups.

Future deployments publish client payloads and simulator dependencies by content
SHA, under per-release installation locks, atomically and read-only. Reuse verifies
bytes without rewriting files or PAYLOAD markers. A changed dependency source
creates a new release. Config copies and B inputs live under
`formal/campaigns/<campaign>/<node>/`; worker cwd, TMPDIR, XDG_CACHE_HOME and
HF_HOME live under that campaign's run and worker generation. Original A pools
and assets are read-only references; LIBERO resolves its existing assets locally,
avoiding shared-home cache/download writes. Existing `libero311_v1`, SHA payloads,
Fast-WAM timan107 and Cosmos timan108 campaigns were not redeployed or signalled.
The next campaign's normal prepare path builds and installs the new payload.

Verification: **96 CPU tests and27 subtests PASS**, including two simultaneous
fleets with injected SIGKILL and successful replacement, fatal cleanup leaving
the other fleet alive, deadline/refusal cleanup, legacy/PID-reuse RAM accounting,
atomic immutable install reuse/source changes, private B/config/scratch paths,
actual client bootstrap/imports, tether retry and legacy/default-off checks.
Regression: [test_concurrent_fleets.py](../ops/tests/test_concurrent_fleets.py).
[CPU log](preflight/concurrent_fleets_20261010/cpu_suite.log),
[JUnit](preflight/concurrent_fleets_20261010/cpu_suite.xml), and
[verification/source hashes](preflight/concurrent_fleets_20261010/CPU_VERIFICATION.json).
No GPU work was performed.

Read-only live timan107 admission preview found about64GiB MemAvailable and
40GiB reserved future growth for Fast-WAM's24 workers, plus22GiB headroom.
Additional24-,12- and8-worker fleets were all refused; don't launch another
fleet there until capacity is available. The failed DP run is still closed at
167/2000; this fix does not resume it or change the running campaigns.
[Live preview](preflight/concurrent_fleets_20261010/timan107_memory_preview.json).

## timan108 LIBERO assets

Fixed timan108's missing configured assets without editing its shared LIBERO
config or existing runtimes. Read-only inventories of the existing timan107 and
timan108 asset caches match for all586 original simulation files (422,327,965
bytes, about403MiB). Their canonical SHA256 manifest identity is
`3eb6e3e7ee3a03663dba84ccc6d6cf9dbb4ae21fb20558ca4097abc066be57dc`.
[Sealed original file pins](specs/libero_original_assets.json) are now included
in formal worker payloads.

Provisioned and verified this immutable release on timan108:

```text
/scratch/zixuans8/mw_r17/formal/assets/libero_3eb6e3e7ee3a03663dba84ccc6d6cf9dbb4ae21fb20558ca4097abc066be57dc/assets
```

`worker_installs` verifies every original file before and after an atomic local
copy, excludes Hub cache metadata, and freezes the release read-only. Existing
releases are reverified and reused without rewriting; missing/changed/unlisted
assets fail closed. There is no downloader. Every LIBERO bundle using the formal
launcher selects this release, links its private simulator package to it and
sets only the new campaign's private config to the verified assets path.
The shared `/home/zixuans8/.libero/config.yaml` remains byte-for-byte unchanged
(SHA256 `f3c42b0384a7c1dd2183ee7841e9c3b0d9799ea8640abd3893db1654f29371d0`).
Future timan107 preparations use the same sealed originals and release process.

Proof: the real `prepare_workers` deployment for DP-LIBERO on timan108 passed
client-only imports and native simulator/pool validation. The additional CPU
admission probe, bound to failed campaign104712's official-50 benchmark seal,
passed the stock worker-probe validator, checked its canonical40-task table,
verified all40 native pools have50 states, and recorded the canonical pool file
and init0 hashes. Both `get_assets_path()` and `get_libero_path('assets')` resolve
to the release above. CUDA was hidden and uninitialized; no renderer, resident,
driver or worker fleet was launched. A read-only24-worker RAM preview also
passed, subject to the launcher's normal admission recheck.

Verification: **102 CPU tests and29 subtests PASS**, including missing configured
assets, immutable reuse, SHA/extra-file/corrupt-release rejection, private config
isolation, both worker lanes, concurrent fleets, tether retries and legacy/default-off.
[Regression](../ops/tests/test_worker_assets.py),
[worker deployment/imports](preflight/timan108_libero_assets_20261010/WORKER_DEPLOYMENT.json),
[DP native admission proof](preflight/timan108_libero_assets_20261010/DP_LIBERO_NATIVE_PROBE.json),
[release/config validation](preflight/timan108_libero_assets_20261010/RELEASE_VALIDATION.json),
and [CPU verification index](preflight/timan108_libero_assets_20261010/CPU_VERIFICATION.json).
The failed campaign and running Cosmos on timan108 and Fast-WAM/DP on timan107
were left untouched. Retry through a new campaign's normal prepare path to use
the updated immutable payload; no GPU work was performed.

## Fast-WAM baseline recovery

Root cause: baseline `formal_fastwam_libero_official2000_20261010_101044_b0c4e68b`
first lost **timan107 worker18, PID423719, SIGKILL(-9)** at **11:18:43 CDT**.
The kernel explicitly records `global_oom` / `Out of memory: Killed process423719`;
normal-zone free memory was below its minimum and all swap was exhausted.
The24 workers accounted for53.01GiB RSS; the victim had the largest combined
RSS+swap footprint in that fleet. This proves a node-wide OOM victim, rather
than an EGL exception, deployment collision, stale health check, or expired
lease. The historical record does not identify the allocator responsible for
the rest of the node's memory pressure. The old fail-fast supervisor then
terminated its other23 children with SIGTERM(-15), aborting the fleet. Original
driver evidence has391 distinct accepted terminal episodes,391 matching
attempts,9278 full K10/C5 policy traces, and zero rejected probes or error
attempts. Original receipt remains FAIL and unchanged.

Evidence: [diagnosis](preflight/fastwam_baseline_recovery_20261010/DIAGNOSIS.json),
[remote fleet receipt and all24 log tails/SHA](preflight/fastwam_baseline_recovery_20261010/REMOTE_EVIDENCE.json),
[kernel victim/process/memory evidence](preflight/fastwam_baseline_recovery_20261010/OOM_MEMORY.json),
[OOM backtrace](preflight/fastwam_baseline_recovery_20261010/OOM_BACKTRACE.json).

C10 was exposed to the same node-wide failure:6144MiB is a cooperative
reservation, not an enforced RSS/native-allocation ceiling; bounded replacement
still requires available RAM. Read-only checks saw RUNNING/no worker failures
and MemAvailable fall from216.5GiB at admission to about89.4GiB. The final
[read-only status](preflight/fastwam_baseline_recovery_20261010/C10_AFTER.json)
now reports **PARTIAL at439 accepted episodes, campaign deadline**, all owned
processes closed, and no newer kernel OOM for its workers. Its fleet's on-disk
RUNNING receipt is stale after group cleanup; PID-birth/liveness admission
correctly reserves nothing for those closed processes. Q did not signal,
restart, or modify C10, DP, or Cosmos.

Future LIBERO formal launches now default to `--worker-max-episodes8` (MW0;
explicit0 disables). [worker_lifetime.py](../ops/worker_lifetime.py) uses the
stock WorkerLoop and runner, reports the completed attempt first, closes its
runner/socket, and emits a PID/birth-bound planned-recycle marker. The owned
[fleet supervisor](../ops/formal_workers.py) replaces that worker under the RAM
admission lock. Planned recycling has its own ledger and does not consume the
fault restart limit; generation paths remain unique across both kinds of
replacement. No mid-episode signal, task/seed change, or broad process matching
is used. This bounds long-lived worker state; it is not a guarantee against
external node-wide pressure. C10's previously deployed immutable payload did
not receive this change.

[formal_eval.py](../ops/formal_eval.py) now accepts `--resume-from`, with the
selection/merge checks in [formal_resume.py](../ops/formal_resume.py) and the
missing-UID dispatch filter in [formal_driver.py](../ops/formal_driver.py).
It requires a closed source, identical model/domain/checkpoint/execution host,
device, benchmark, sampler/horizon, retry/episode protocol, sealed ModelManifest,
norm/code pins, batching treatment, full YAML and every original episode identity. Native worker
runtime/probe and episode-code bindings must also match. Changes to worker
count, resident count, transport ports or RAM/lifetime controls are resource
changes; the original semantic campaign, cell IDs and paired policy seeds stay
fixed. Successes and legitimate failed episodes are both retained. Duplicate,
unresolved, missing-attempt or incomplete/mismatched trace evidence is refused.
Unmerged continuation runs are refused as resume sources; select their original
`resume_from` source instead, rather than silently losing inherited results.

Current P1 code has four legitimate Fast-WAM pin changes (`batcher.py`,
`provider.py`, `serve.py`, `staged.py`), producing model identity
`123b2d8c...` versus the baseline's `9f8304f7...`. Fresh-current sealing was
**correctly refused**; see [pin diff](preflight/fastwam_baseline_recovery_20261010/MODEL_PIN_DIFF.json).
H100 resume therefore makes a SHA-checked **new private copy** of the source's
preserved runtime archive (`fe420ba49c502e509a233e9ca3e99805cfef40e829d963b2b0331fe0adfd827b`),
CPU-seals with its original native code, and still checks current external
weights/base/Wan/tokenizer/norm bytes. It never extracts over the original or
another live runtime. Current Q worker lifecycle runs in a separately sealed
client payload. No P-owned repository source was changed or reverted.

CPU preflight **PASS**:
`/data/mw_finetune/eval/fastwam_baseline_resume_final_cpu_20261010/`.
Its `RESUME.json` retains391, `EPISODE_PLAN.json` dispatches only1609, and
`FULL_EPISODE_PLAN.json` preserves the full2000 denominator. ModelManifest,
full YAML, all task identities, and the timan107 native CPU probe match the
original. `PREFLIGHT.json` previews24 workers /6144MiB, N2 replicas, ports
23180/23181/23182/23189; RAM admission and these ports were free at the last
check, and actual launch rechecks them. The private recycling wrapper's
[deployed import probe](preflight/fastwam_baseline_recovery_20261010/RECYCLE_WRAPPER_IMPORT_LATEST.json)
also passes client-only imports. Prepare-only records resource admission as a
preview and can finish CPU validation while a node is busy; `--run` continues
to refuse failed admission before launching residents.

After the missing-cell driver and SHA-checked fleets close successfully,
resume preserves the raw segment under `resume_segment/driver/`, checks source
SHA pins again, and constructs a union journal containing every original cell
exactly once. The new run's `RECEIPT.json` and `SUMMARY.json` can become PASS
only for the verified full2000 union. `RESUME_MERGE.json` discloses the source
FAIL receipt,391 retained /1609 dispatched, segment rc/retry recovery, and any
abandoned prior attempts; summary retains suite/task SR,40-task mean and
error/retry counts. All original evidence stays untouched. No synthetic result
was written into a production campaign.

Verification: **141 CPU tests plus29 subtests PASS**, including a391+1609 merge
with successes and legitimate failures, strict checkpoint/bundle/code/norm/
benchmark/seed/cell mismatch refusal, source mutation and live-source refusal,
incomplete/error/duplicate evidence, stock-wire result-before-recycle, separate
recycle/fault budgets, unique generations, two-fleet SIGKILL recovery and owned
cleanup, immutable assets, tether retries, and legacy/default-off checks.
[Resume regressions](../ops/tests/test_formal_resume.py),
[worker lifetime regressions](../ops/tests/test_worker_lifetime.py),
[CPU verification index](preflight/fastwam_baseline_recovery_20261010/CPU_VERIFICATION.json),
[suite log](preflight/fastwam_baseline_recovery_20261010/cpu_suite_latest.log).
The final run preserves concurrently added default-off batching options;
baseline resume explicitly remains B1 and rejects B1/B8 or DP batch mixing.
No GPU resident, proxy, conductor or simulator worker was started by Q.

Coordinator command (new run directory; original baseline remains immutable;
H100 policy23180, backends23181/23182, local driver23189):

```bash
bash /home/weiland/projects/openpi/exp/r17_integration/ops/formal_eval.sh \
  --model fastwam_libero --checkpoint /data/mw_finetune/models/fastwam_libero \
  --server-node h100 --worker-node timan107 --replicas 2 --workers 24 --max-batch 1 \
  --worker-memory-mib 6144 --worker-restart-limit 2 --worker-max-episodes 8 \
  --official-50 --seconds 86400 --episode-seconds 3600 --retries 2 \
  --memory-budget-gib 64 --port 23180 --driver-port 23189 \
  --resume-from /data/mw_finetune/eval/formal_fastwam_libero_official2000_20261010_101044_b0c4e68b \
  --run
```

## Formal prestaging and queue

The owner superseded the earlier B1 Fast-WAM recovery command: **do not resume
that baseline**. The queued example starts a fresh B8/5 ms control and then its
C10 arm. Additional arms are more argument arrays. Cosmos resume still uses
its retained-cell/source identity proof and archived native runtime; D's new
explicit unreachable-source closure exception was preserved and retested.

Startup reconstruction is in
[STARTUP_BREAKDOWN.json](preflight/formal_prestage_20261010/STARTUP_BREAKDOWN.json).
The exact named runs contradict the rough launch-to-journal examples: DP first
dispatch was 11:24:00.777, first accepted 11:25:16.786; Fast-WAM first dispatch
11:23:45.025, first accepted 11:25:19.691 CDT. Neither named run waited until
11:30/11:36 for initial dispatch. First acceptance includes episode execution,
queueing and reporting; it is not a GPU-idle measurement.

| Historical startup phase (seconds) | DP12 / timan108 | Fast-WAM2 / timan107 |
| --- | ---: | ---: |
| Runtime/worker tar construction, transfers/install, server asset snapshot | 1.89 | 6.42 |
| CPU model seal and plan | 20.72 | 29.16 |
| Immutable dependency/original asset verification | 1.93 | 20.59 |
| Worker import + all original init-pool validation | 4.26 | 10.01 |
| RAM admission preview | 0.28 | 0.94 |
| Private config and campaign/benchmark pushes | 0.99 | 2.80 |
| **CPU preparation total** | **30.08** | **69.92** |
| Sequential replica load/startup self-check | 266.98 | 85.58 |
| Proxy startup + bundle acknowledgement | 0.66 | 1.91 |
| Worker startup/renderer attestation to first dispatch | 17.06 | 24.62 |
| First dispatch to first accepted episode | 76.01 | 94.67 |
| **Launch to first accepted** | **390.79** | **276.69** |

Durations use invocation timestamps in TETHER_CALLS, local artifact mtimes,
stock driver dispatch/journal records and native resident ready receipts.
Historical subphase boundaries are inferred from the contemporaneous prepare
call order; run-name launch timestamps have one-second precision. Individual
resident ready durations were DP 18.55–22.02 seconds and Fast-WAM 38.90/39.05.
No evidence supports an additional ten-minute transfer phase in these runs.
Future runs write named PHASE_TIMINGS.jsonl records for sealing, release
publication, dependency/assets checks, native probes, admission and readiness.

Changed Q-owned ops:
- `formal_prestage.py` / `immutable_release.py`: immutable server releases
  addressed by file-content hashes, verified before reuse; tar timestamps do
  not force new deployments. Existing releases are never overwritten. Worker
  payload cache hits verify every pinned byte and skip both pushes/install.
  Original LIBERO assets/dependencies remain verified immutable shared releases.
- `formal_eval.py`: `--prepare-only` writes a bound PREPARED.json without GPU
  replicas, fleets, EGL construction or Cosmos tunnels. `--run --prepared-run`
  consumes it once, rejects mutations/overrides, revalidates immutable payloads,
  private configs, benchmark bytes, RAM and resume pins, then binds services.
  DP startup defaults to four concurrent loads per wave, Fast-WAM two; all
  resident self-checks, DP CUDA graphs and VRAM accounting remain required.
  Workers bootstrap while replicas load; stock reconnects wait until the
  driver starts **after** every resident attests and the proxy acknowledges.
  Startup fleet failures fail immediately; native-parity-only preparation
  records the empty worker deployment.
- `formal_queue.py` / `.sh`: one GPU, one CPU preparation ahead, argv-array
  input, paired-control references, owned PID/pgid receipts and stop on any
  preparation/campaign/cleanup failure. Arms can stage while a control runs;
  the closed PASS control/batching audit gate is checked again before GPU use.
- `worker_installs.py`: readonly verification of prepared dependency and asset
  releases, without re-importing the native simulator at the switch.

M's final LIBERO defaults are retained: **four episodes / 6144 MiB**, with
bounded fault restarts separate from planned recycling; MW remains zero/off.
D's renderer distribution hooks, VRAM reservation, device/UUID receipts and
protected timan108 GPU1/2 exclusion were preserved. P3/P4 batching options and
D's concurrent resume-closure/tunnel changes were preserved. Q did not edit
formal_workers.py, worker_gpu.py, formal_process.py, worker_lifetime.py,
formal_resume.py or src. D's newer resume selection is included in PREPARED's
file pins; its source-closure proof refresh continues through open_tunnels.

CPU prepare proof:
[PREPARE_SMOKE.json](preflight/formal_prestage_20261010/PREPARE_SMOKE.json),
with runs `/data/mw_finetune/eval/q_prestage_{cold,release,warm}_cpu_20261010`.
The warm run reused **both** server runtime and timan107 payload; runtime
build/verification took **0.54 seconds**. Both release/warm runs passed native
CPU validation of all 40 original init pools and the client-only import probe.
No simulator fleet/renderer or GPU resident was launched by Q.

Expected prepared switches: **DP12 approximately80–110 seconds;
Fast-WAM2 approximately60–100 seconds**, including warm validation, owned
cleanup/control overhead and stock worker reconnect. These estimates derive
from historical individual load times plus overlapping startup; no GPU test
of the new concurrency was authorized/run. The remaining warmup is measured
by future receipts. Cosmos parity/other unmeasured compilation can exceed two
minutes; no universal model-load bound is claimed. First accepted episodes
add actual episode execution to these switch times.

Validation:
- Final broad CPU suite: **175 passed +31 subtests**, including legacy/default
  off, resume, driver/proxy qualification, renderer admission, concurrent
  fleets, native recycling and tether retries. Evidence:
  [cpu_suite_final.log](preflight/formal_prestage_20261010/cpu_suite_final.log).
- After D's concurrent closure-waiver hooks landed, current-source targeted
  rerun: **92 passed +29 subtests**, including D's waiver tests, prepared
  launch, formal eval/resume and default-off. Evidence:
  [concurrent_edits.log](preflight/formal_prestage_20261010/concurrent_edits.log).
- New regressions verify actual SHA cache hits/no pushes, unchanged immutable
  inode/mtime, refusal of corrupt releases and changed assets, one-shot launch,
  no Cosmos prepare tunnels, one-ahead overlap, refusal after failure/open
  cleanup, closed-control dispatch gate, GPU-target isolation, and four owned
  residents spawned before the first readiness wait with owned cleanup.
  The latter also asserts worker bootstrap precedes resident spawn and that
  every DP resident keeps `--cuda-graph` and its attestation.
- Shell syntax and the supplied B8/5 ms queue argument sets validated on CPU.
  [FINAL_SOURCE_PINS.json](preflight/formal_prestage_20261010/FINAL_SOURCE_PINS.json)
  records the integrated sources. Historical campaigns/evidence are untouched.

Coordinator commands (do not use the old B1 resume command):

```bash
# Edit/add argv arrays and unique run names for the desired control/all arms.
exp/r17_integration/ops/formal_queue.sh \
  --jobs exp/r17_integration/ops/formal_queue_fastwam.json --run

# Stage a single next campaign while the current campaign runs:
exp/r17_integration/ops/formal_eval.sh <campaign arguments> --prepare-only
# Consume that exact preparation later:
exp/r17_integration/ops/formal_eval.sh \
  --run --prepared-run /data/mw_finetune/eval/<prepared_run>

# CPU-only staging of a whole argument list, with no dispatch:
exp/r17_integration/ops/formal_queue.sh --jobs <jobs.json> --prepare-only
```

[FORMAL_QUEUE.md](../ops/FORMAL_QUEUE.md) documents the argv-array schema and
`--paired-control @earlier_run_name`. The supplied example uses H100/timan107,
23173/74/75 plus driver23177; the coordinator must own those ports/GPU at run
launch. CPU preparation does not bind them. All results stay under the existing
formal eval root; no live campaign or other user's process was signalled.

Final integrated verification after the model-interpreter compatibility fix:
**204 passed +31 subtests** in137.24 seconds, including D's closure-waiver tests
in the broad suite and legacy/default-off checks. Evidence:
[cpu_suite_release.log](preflight/formal_prestage_20261010/cpu_suite_release.log).
Cosmos's actual H100 Python3.10.21 lacks `hashlib.file_digest`; Q's standalone
runtime hashing/bootstrap and prepared remote verification now work on3.10.
The actual Cosmos interpreter installed and then reused a tiny SHA-bound
CPU-only archive under a new owned /tmp stage; no model import/GPU work.
[PY310_NATIVE.json](preflight/formal_prestage_20261010/PY310_NATIVE.json) records
both results. The additional regression removes the3.11 hashing API and
checks installation plus byte verification. FINAL_SOURCE_PINS reflects these
final helpers and the preserved concurrent D hooks.


## Frozen paired-study runtime

Cause: between the hs3 Fast-WAM B8/5 ms control seal and C20 preparation, P3 added `exp/r17_integration/models/fastwam/mps.py`. Fast-WAM's loader pins every top-level `models/fastwam/*.py`, so the new `code:r17:mps.py` pin (`0485fdd8c47c0fe363ffc97f08280efe2ceb5ca3af95c622c8e9bb6b5d3630f2`) changed the ModelManifest code digest and checkpoint identity even with MPS disabled. The control identity was `fd5264922c5a7aebfdfab056bd89e172b8c7e90931d355f6783fc7b8b88c9528`; C20 instead sealed `b63a5e7a58cf4537faa41bd0d34136661dcb8424090b048c26b74107123d815b`. `preflight/frozen_runtime_20261010/IDENTITY_DIFF.json` records the exact manifest differences and the other archive-only edits (`models/dp/samples/dynamic_batch_report.py`, `ops/formal_eval.py`, `formal_prestage.py`, `immutable_release.py`, new `perf_recorder.py`). Only the new Fast-WAM MPS file changed this ModelManifest.

Changes: new Q-owned `ops/formal_runtime.py` verifies source archive SHA, release content identity, immutable execution-host release, preserved client payload/deployment, and original ModelManifest. `formal_eval.py` accepts `--runtime-from <baseline_run>`; paired-control arms inherit it automatically. The native CPU seal must reproduce the original model/benchmark manifests exactly, including weight/vendor/norm/code pins. Frozen local sealing and resident/proxy imports use the release cwd; the conductor driver uses a verified local mirror of that release. Workers reuse the control's exact preserved payload. Missing/corrupt archives or releases and checkpoint, bundle, batching, topology (Fast-WAM) or MPS changes fail before GPU dispatch. Source hashes are rechecked at prepared launch, and `FROZEN_RUNTIME.json` is part of the preparation seal. The DP model-owned shim is unchanged: binding occurs inside the Q prepare entry after its batching hook populates the campaign. `formal_prestage.py`, `formal_queue.py`, and `FORMAL_QUEUE.md` were updated narrowly; P3's MPS edits are preserved.

`formal_queue.py` explicitly adds `--runtime-from` to each paired job. Its new `--prepared-queue` launch mode validates all existing preparation seals and uses their settings without interpreting them through later model CLI edits or rebuilding occupied run directories. It retains the closed-PASS paired-control gate, one-shot claims, current launch RAM admission, and stop-on-failure/unclean-exit behavior.

CPU tests: final suite `preflight/frozen_runtime_20261010/cpu_suite_final.log` / `.xml`: **217 passed, 40 subtests passed**, including legacy/default-off, worker recycling/renderer admission, fleet isolation, source resume/closure waiver, and tether retry checks. New `ops/tests/test_formal_runtime.py` covers checkout edits/new MPS files, missing/tampered/unbound source archives, immutable release absence, model/checkpoint/batching/MPS mismatches, changed source pins, unchanged live-control receipts, frozen cwd imports, preserved Fast-WAM implementation treatment with topology rejection, DP hook ordering/flag forwarding, paired-source mismatch, prepared-queue validation/GPU isolation and failure stopping.

All six Fast-WAM arms passed CPU-only preparation in 277.7 s, ordered **C10 → C16 → C20 → C32 → K2 → K1**. Each plans 1,000 original-init episodes for the requested spatial/10 suites. Every server archive SHA is exactly the control's `2e474ca33d561cfa2e24617e12fa217e4f323575f603d78e3ec61e0b1751b6b9`; every release identity is `5a041e420a878fe9be249e04d3c7c6b31518eab21009aa933ec59cf8dc9967c3`; all ModelManifests and worker archive SHAs equal the control's. Existing immutable server and worker releases were reused, no GPU/fleet/dispatch was started, and all owned CPU children closed. Busy-node RAM admission remains a preview and is checked again at launch. Evidence: `preflight/frozen_runtime_20261010/PREPARE_RESULTS.json`, and `/data/mw_finetune/eval/formal_queue_hs3_fast_frozen_20261010_132040/QUEUE_RECEIPT.json`. Original failed queue/results and both running controls were untouched.

Fast-WAM: the original jobs file is retained. The new `/home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs3_fast.frozen.json` contains all six jobs with unique `_frozen_132040` run-name suffixes and explicit `--runtime-from /data/mw_finetune/eval/hs3_fast_b8w5_h100_k10_c5_control_20261010_125303`. CPU preparation was:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 exp/r17_integration/ops/formal_queue.sh --jobs /home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs3_fast.frozen.json --prepare-only --output /data/mw_finetune/eval/formal_queue_hs3_fast_frozen_20261010_132040
```

That preparation is complete; its run directories are one-shot. **After the Fast-WAM control completes with a closed PASS**, the coordinator launches the six existing preparations using:

```bash
exp/r17_integration/ops/formal_queue.sh --prepared-queue /data/mw_finetune/eval/formal_queue_hs3_fast_frozen_20261010_132040 --run
```

DP: `/home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs3_dp.sh` now adds `--runtime-from "$B"` to its shared arm invocation, where `B=/data/mw_finetune/eval/hs3_dp_b16w5_4090_k100_c5_control_20261010_125303`. Its baseline wait, batching, checkpoint, suites, GPU lock, ports and arm order are retained. The real model-owned wrapper CPU preflight passed at `/data/mw_finetune/eval/q_dp_frozen_cpu_20261010_132330/` with B16/5 ms, C10, the requested worker treatment and exactly the DP baseline's ModelManifest; `preflight/frozen_runtime_20261010/dp_prepare.log` and the run's `FROZEN_RUNTIME.json` / `PREPARED.json` contain proof. `bash -n` passed. The coordinator's command is:

```bash
bash /home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs3_dp.sh
```

No GPU use; no running campaigns, workers or their process configuration were changed or signalled. No P-owned package or `src/` file was edited. Do not start a duplicate queue if the coordinator already has one active; this work did not start either GPU queue.


## Grouped driver orchestration

2026-10-10, CPU only. S's evidence (`REQUESTS/S_to_R.md`, `/data/mw_finetune/perf/2026-10-10/headroom.json`) matches the reproduced hot path: `ops/formal_driver.py:114–117` opens/flushes/fsyncs each evidence row; its `handle_result` holds the scheduler lock at line128 across the stock result handler and every step append at line145. `conductor/queue.py:64–67` separately fsyncs each assignment under that same lock. Each pull also calls `MetaWorldStrategy.ensure_bundle` (`conductor/strategy.py:53–58`), serializing a redundant control WebSocket round trip for a fixed already-published bundle. There is no tether call inside the dispatch loop; the launcher’s batched fleet lease/status probes are outside the driver process.

Added `ops/formal_driver_fast.py` and `ops/group_commit.py`, opt-in with `--driver-mode group-commit`. The qualified queue selection, endpoint compatibility, original task/init/seed metadata, dispatch generations, retry classification/budget, native transport/probe admission and stage semantics are retained. Pull/report handlers enqueue bounded in-memory evidence; no filesystem I/O runs under the scheduler lock. A single writer groups at one second or64 events, writes/fsyncs the supporting streams, then writes/fsyncs terminal journal rows and invokes scheduler completion callbacks. Results awaiting commit reserve their exact generation, so disconnects/timeouts/duplicate reports cannot steal it. After the launcher and inherited preload validate the fixed `formal_full_v1` bundle, pulls skip the redundant driver-side select; workers retain their original selection behavior. Grouping is restricted to pure inference; library/capture and parity-only campaigns are refused. Legacy is still the default, with `formal_driver.py` and all P-owned packages/`src/` unchanged.

**Durability guarantee:** an episode is counted by the scheduler and completion monitor only after its step/attempt/dispatch evidence AND its terminal journal row have completed successful fsyncs. Evidence-file creation and the driver/campaign directory entries are also synced before any acceptance. A surviving complete terminal frame therefore always has durable prerequisite evidence. Pending assignments/nonterminal diagnostics may be lost at a crash; they do not authorize skipping an episode. Recovery uses the existing `--resume-from` into a new segment after the source fleet is closed or explicitly isolated; in-place restart is refused, avoiding reuse of an uncommitted dispatch generation. Retries become schedulable after their evidence barrier. A writer failure/overflow or unclosed drain fails the campaign, including the rc2 completed-worker-shutdown recovery path. Resumed grouped runs additionally fsync a SHA-bound original segment under `durable_segment/` before the stock merger replaces files, and fsync the merged evidence/plan/receipt before aggregation. The source run remains untouched.

`ops/driver_release.py` installs a separately recorded immutable orchestration release: clone the baseline runtime, overlay ONLY the two driver/writer files, verify the exact delta, and freeze the result under its full content hash. All native model/server/worker/benchmark code stays byte-identical. `ORCHESTRATION.json`, `campaign.json`, the launcher receipt and summary record both the source baseline driver and the selected driver maps/identities. Source controls that predate explicit pins derive their legacy pin from the verified sealed runtime. Existing explicitly pinned controls are verified and inherit their actual recorded selected pin. Driver cwd/PYTHONPATH and endpoint replanning use the recorded release, avoiding working-tree import precedence. The framework modifications in `formal_eval.py`, `formal_prestage.py` and `formal_queue.py` are narrow; P3’s incoming MPS validation and D’s closure/recovery changes are retained.

Real `/data` replay:24 concurrent fake workers,32 episodes,24 step rows/episode, real JSON appends and fsync, **no injected I/O delay**, no GPU work. Legacy handlers were extracted verbatim from the DP control's sealed `runtime.tar` (formal driver SHA `ce49d2987fcf819219e5f0a6b0ba0d88d0d1d9e15f94c1cb696a8e399e4e8b0a`). Input/task and terminal outcome hashes match exactly; all24 workers received assignments and all closed. Evidence: `/data/mw_finetune/qualification/driver_replay_20261010_group_commit_final/RESULTS.json`; preserved old run: `driver_replay_20261010_group_commit_v2/legacy/`.

| CPU replay measurement | Legacy | Grouped |
|---|---:|---:|
| pull latency p50 / p95 |667.029 /1787.168 ms|0.1282 /0.1992 ms|
| report handler p50 / p95 |27,628.697 /47,134.679 ms|0.0870 /0.1523 ms|
| report → next assignment p50 / p95 |11,558.672 /16,697.673 ms|0.2161 /0.2610 ms|
|32 episodes, durably drained |59.982 s|1.563 s|
|fsync operations |864 per-row/per-dispatch|13 total, including namespace barriers|

The last-cycle metric is recorded in `RESULTS.json` and reflects workers that receive another task. Replay completion includes the final durable drain. These measurements isolate CPU orchestration; end-to-end GPU/batch-fill improvements remain to be measured by the coordinator. The first96-episode replay was stopped by its recorded owned CPU PID to reduce filesystem load; its partial evidence is retained, and it is not used for the reported comparison.

CPU regressions (`ops/tests/test_driver_group_commit.py`) cover blocked/failed fsync without lock blocking, completion after barriers, journal failure, actual hard process exits before evidence/after terminal fsync/after acceptance, duplicate/stale/timeout fences, retry exhaustion, grouped shutdown drain, fixed-bundle pulls without control traffic, immutable overlay restrictions, checkout edits after pinning, frozen endpoint planning, one-shot prepare/launch lock, refusal to refresh launched runs, refresh mode selection, partial-queue failure and rc2 recovery refusing failed commits. The broad CPU/legacy/default-off suite passed **231 tests +43 subtests** (`preflight/driver_group_commit_20261010/cpu_suite.log` / `.xml`); after subsequent merge/validation changes, focused final suites passed **47 tests +32 subtests** (`cpu_final.*`) and **97 tests +30 subtests** (`cpu_resume_final.*`), including the final grouped-resume durability regression. Latest resume durability regression preserves the original SHA-bound segment across replacement of every aggregate input and requires successful composite barriers. No GPU, worker fleet, resident, tunnel or campaign dispatch was launched for verification.

**Ready CPU preparation:** H100 **C10 → C16 → C20 → C32 → K2 → K1**,6/6 PASS at `/data/mw_finetune/eval/formal_queue_hs4_fast_frozen/QUEUE_RECEIPT.json`; DP **C10 → C15 → C16 → DDIM10 → DDIM5**,5/5 PASS (`preflight/driver_group_commit_20261010/DP_QUEUE_PREPARED.json`). Each arm plans the requested1,000 original spatial/10 cells, keeps the exact baseline's ModelManifest, runtime archive SHA and worker archive SHA, and has no LAUNCH/RECEIPT or GPU binding. `PREPARE_RESULTS.json` independently audits all11 immutable seals and equality to the source controls. `DRIVER_PREFLIGHT.json` verifies frozen plan equality, native execution-host release hashes and timan107’s CPU import probe; full native task/init/renderer admission remains required at first real pull. Previous queue/preparation seals are retained. The initial CPU refresh's omitted mode assignment was fixed and covered by a regression; its FAIL evidence was preserved before the original CPU seal was verified and refreshed successfully.

Driver pins: Fast-WAM legacy `43a75f0553c843abc9a5ddb7aa546925b51ce22a87813c42a4fdef28fc385078` → grouped `c5e1f4ca6358b1dc97bf24b281d980d3d5a33f225a8d8965cbf696b71914cab8`; DP legacy `e0b6aca18bc8e8e64c2e5c7173d8adbe7f760aaab5f88cd1cdeed3cf82aef3e9` → grouped `07fac83b543f95c83f59d03a49f3d0674588684c137779a4ba7d11c382e9d431`. All selected overlays match the current two helper SHA pins.

Both scratch queues now select `--driver-mode group-commit`. The DP script pins `/data/mw_finetune/eval/hs4_dp_b16w5_4090_k100_c5_control_20261010_133915`; `--prepare-only` skips its control wait and GPU flock. Default/`--run` waits for control PASS and launches the already prepared directories through the common launcher, preserving B16/5ms, worker recycling/spreading, ports23180/23189, exact native source and the local GPU lock. GPU execution has not been started by Q.

CPU refresh/preparation completed with:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 exp/r17_integration/ops/formal_queue.sh --prepared-queue /data/mw_finetune/eval/formal_queue_hs4_fast_frozen --prepare-only --refresh-driver group-commit
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 bash /home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs4_dp.sh --prepare-only
```

After the respective controls finish with a closed PASS, the coordinator’s launch commands are:

```bash
exp/r17_integration/ops/formal_queue.sh --prepared-queue /data/mw_finetune/eval/formal_queue_hs4_fast_frozen --run
bash /home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs4_dp.sh --run
```

The control wait remains enforced for DP; Fast-WAM’s paired-control gate remains enforced at launch. Existing prepared directories are one-shot, so do not reprepare them with the scratch JSON jobs or start a duplicate queue. A future unlaunched queue can update its orchestration with the CPU refresh command above. Source controls, all running campaigns and recorder processes were not modified, signalled or reconfigured. No commit/push, broad process matching/kill, forbidden path/node access or P-owned code edit was performed.

## Tether completion recovery (10-10 afternoon)

DP `hs5dp_ctl_142812` is **PASS**, with2000/2000 accepted and zero terminal errors.
The coordinator recovered482,928,640bytes in a reported93s. Q verified preserved
remote SHA `c1daf1be89c13bc10789e82ca1b0ec5d805120e4c0c75e9a511dfbd6d042d9bf`,
installed all7717 snapshot files, checked exact durable journal completion,
closed owned processes, resident attestations and full inference traces, and
aggregated automatically. The failed receipt is preserved as
`RECEIPT_BEFORE_RECOVERY.json`; `RECOVERY.json` records its error and recovery.
Receipt/SUMMARY are `/data/mw_finetune/eval/hs5dp_ctl_142812/{RECEIPT,SUMMARY}.json`.

Fast-WAM `hs6fw_res_152931` merged1839 retained +161 new cells into2000/2000,
then also suffered a final H100 server-evidence timeout. **After all owned
processes were confirmed closed**, Q recovered441,313,280bytes in SHA-checked
chunks, matching preserved SHA
`faee7ad4f4a2c69addd269b5b78994c84be1e9e2cd056163d9a46a81151386bd`.
The CPU batch audit from the frozen native runtime passed all10877 resident
decisions and the original treatment identity
`ec35c19a05d89550774b355718b5bb63a4680894dceadb45818a48be33435666`.
The merged receipt is now **PASS**, with its failed receipt and recovery proof
preserved. Receipt/SUMMARY:
`/data/mw_finetune/eval/hs6fw_res_152931/{RECEIPT,SUMMARY}.json`.
The original1839-cell segment, including its resume-pinned failed receipt, was
left unchanged.

All six jobs in
`/home/weiland/projects/openpi_ext/scratch/r17_ops/queue_hs6_fw.json` and their
existing preparations in `/data/mw_finetune/eval/fq_hs6fw` were updated to:

```text
--paired-control /data/mw_finetune/eval/hs6fw_res_152931
--runtime-from /data/mw_finetune/eval/hs6fw_ctl_143332
```

CPU rebinding and frozen runtime/worker import/plan preflights passed6/6
(`CONTROL_REBOUND_PREFLIGHT.log`, `QUEUE_RECEIPT.json` in that queue). The
completed native paired-control gate also passed6/6 (`CONTROL_GATE.json` below).
The coordinator subsequently launched C10; Q did not modify launched campaigns.
Preparation can defer control completion; launch requires a complete closed PASS
merge. Unrelated controls, changed native archives/source pins/cells and
incomplete or unclosed merged controls are refused. Existing preparation seals
must be rebound with the CPU-only `formal_queue --rebind-control` option;
changing a jobs file alone does not update those seals. Old seals/native inputs
are retained, and rebinding refuses launched runs.

Cause: the retry classifier missed rc70 `waiting for chunk`, `nats: timeout`,
and `read pipe: i/o timeout`, including the `pull (tier B):` envelope. The fix
adds safe retries with2/4/8/16/30s backoff within300s. Real application/checksum
errors, confirmed fleet death and sustained outages remain failures. A lost
spawn reply is recovered through that exact request's receipt without executing
spawn twice. Archive cleanup can replay its own verified cleanup receipt.

All formal evidence pulls, including worker/server archives, asset archives and
library shards, now use chunks of at most4MiB, per-chunk and whole-archive SHA
checks, independent retries,2MiB/s aggregate throttling via a bulk-only lock, and
worker-sync heartbeat callbacks. Original stages remain until complete snapshot
verification. The real H100 recovery survived five rc75 saturation blips,
recovering after about62s without restarting the archive. File-only smokes
passed on timan107 (6.0s) and H100 (7.7s), each4,250,000bytes in two chunks.
System Python3.8 compatibility was fixed and verified on timan107. timan108 was
offline; its optional smoke exhausted300s, and no process or agent there was
changed.

Q files changed: `ops/formal_eval.py`, `tether_retry.py`, `evidence_transfer.py`,
`transport_operations.py`, `formal_runtime.py`, `formal_prestage.py`,
`formal_queue.py`, `formal_recovery.py`, and
`ops/tests/test_tether_resilience.py`. RUN_CONFIGS pitfall: `REQUESTS/Q_to_R.md`.

Validation:72 focused CPU tests +9 subtests PASS. The first broad ops/default-off
run passed462 tests +43 subtests and exposed three failures: a fixture sharing
the production bulk lock, an overly strict test error-text match, and a malformed
request-less legacy spawn fixture. All three were corrected and retested in the
passing focused run. Default-off checks passed. The actual two recovery paths,
native batch audit, six control gates, six frozen preflights and two file-transfer
smokes also passed. Evidence under
`qualification/preflight/tether_baseline_recovery_20261010/`:
`RESULTS.json`, `REGRESSION_TESTS_FINAL.log`, `CPU_TESTS.log`, `CONTROL_GATE.json`,
`H100_CHUNK_SMOKE.json`, `timan107.bin.TRANSFER.json`.
No GPU work by Q; no running campaign was stopped, signalled or reconfigured;
no commit/push.


## Private MPS control and zero-acceptance queue retries

Cause: both DP C15 starts failed before model initialization/dispatch at the native
replica `get_server_list` query. A live, ready private daemon accepted two UI
connections at exactly 15:56:02.663 (c15) / 15:57:47.319 (c15r0), but only one UI
command completed. Both runs have zero accepted episodes and confirmed owned
closure; C10 used the identical frozen MPS file successfully. Exact receipts,
log excerpts and source hashes: `qualification/preflight/mps_queue_20261010/ROOT_CAUSE.json`.

Fix: Q `ops/mps_control_guard.py` shadows only the CLI, using a private-pipe lock
and a 7s bounded backoff for read-only send failures. Native mutating calls and
real errors are not retried. `formal_eval.py` installs it by value outside the
frozen runtime, records its SHA separately, and preserves startup width 2. No
server/worker/model source overlay. `formal_prestage.py` preserves sealed inputs
before its exclusive launch claim. `queue_retry.py` creates fresh, CPU-prepared
copies only after a closed zero-acceptance failure; manifests, YAML, runtime/worker
archives, episode inputs/seeds, settings and selected driver remain identical.
`formal_queue.py` enables at most two retries by default and adds safe continuation.
Original failed receipts remain unchanged. REQUESTS/Q_to_R.md has the RUN_CONFIGS pitfall.

Validation: 138 CPU tests and 41 subtests PASS, including race/retry/isolation/deadline/mutating-command regressions,
zero-acceptance and cleanup guards, bounded fresh-copy retries, one-shot claims,
continuation boundary and frozen/default-off checks. The full result is in
`qualification/preflight/mps_queue_20261010/CPU_TESTS.log`. H100 by-value deployment
and four simultaneous fake CPU CLI calls PASS (`H100_CPU_GUARD.json`); no real
MPS/GPU command was run. A real frozen C16 CPU clone is
`/data/mw_finetune/eval/q_mps_cpu_161954` (not launched); its native archive,
worker archive, manifest, YAML, episode plan and driver pin are byte-identical,
and all 8 remote asset files verify (`REAL_FROZEN_CLONE.json`, `H100_CLONE_ASSETS.json`).

Running queue: PID 2898445 starts each formal_eval child from the current project,
so C16..K1 automatically receive the MPS fix without changing their preparation
seals or interrupting C10. That already-loaded parent cannot acquire the new retry
loop. The coordinator can start this continuation now in a separate tmux window;
it waits for the old queue's natural closure before any GPU/worker dispatch:

```bash
/home/weiland/projects/openpi/exp/r17_integration/ops/formal_queue.sh \
  --continue-queue /data/mw_finetune/eval/formal_queue_20261010_155703_e26f7fd4 \
  --run --startup-retries 2 \
  --output /data/mw_finetune/eval/fq_hs6fw_mps_recovery
```

It skips closed PASS arms, retries only a fully closed zero-acceptance failure
under a fresh name, and then launches the remaining sealed preparations. If every
old arm passes it returns an empty PASS. A failure with accepted results is left
for explicit resume. No queue/campaign was launched, stopped, signalled or
reconfigured by Q; the H100 CPU clone uses separate paths and no GPU.


## Completed shutdown ACK gaps and GPU-first evidence sealing (2026-10-10)

C10 `/data/mw_finetune/eval/hs6fw_c10/RECEIPT.json` is PASS:1000/1000 accepted,
no terminal/errors attempts, all GPU/fleet owners closed, native frozen batching
audit PASS (38931 decisions). Raw driver rc2/PARTIAL is preserved. The missing
shutdown ACKs for47 of48 workers are disclosed in RECEIPT/SUMMARY, alongside
receipt-bound fleet closure. Original PARTIAL is `RECEIPT_BEFORE_SHUTDOWN_RECOVERY.json`;
`SHUTDOWN_RECOVERY.json` binds its SHA and the recovery. An additional server-stage
failure was verified:645806080 bytes exceeded the frozen512MiB helper limit.
A compressed exact-file snapshot (12545692 bytes, SHA
`3a421fa05763758fa60da061def519b2d282555c9cf65f85058bcca4df6d1808`) now verifies
all resident records and the qualified native batching treatment. Originals stay
preserved; no frozen runtime/server/worker/model code was edited.

`ops/formal_eval.py` now closes owned processes before final evidence pulls.
`ops/formal_finalize.py` gates missing-ACK recovery on exact durable completion,
resolved generations/errors, and verified child/supervisor/fleet closure. It
handles all model families. Ordinary formal campaigns then return rc0 for a
verified GPU handoff while their receipt remains PENDING_EVIDENCE. A detached,
CPU-only, PID/start-tick/pgid-recorded finalizer verifies transfers, frozen native
audits and aggregation before publishing scientific PASS. GPU lock FDs are not
inherited. Closed inputs and the CPU code archive have a separate recorded pin;
model/server/worker/driver pins remain intact. Compressed H100 snapshots avoid
the old512MiB ceiling via a bounded process-local8GiB orchestration override.
`sync_back.py` preserves uncompressed default behavior; `transport_operations.py`
restores the qualification bound after cleanup. Smoke/library/cache/parity remain
synchronous scientific closures with GPU cleanup preceding final transfer.

`formal_queue.py` / `queue_retry.py` switch on verified GPU handoff, check late
failures before new dispatch, and wait for all evidence seals before queue PASS.
`formal_resume.py` rejects pending sources; `formal_runtime.py` keeps pending
controls behind the completion gate. `formal_recovery.py --shutdown` provides
conservative CPU recovery without rewriting the raw driver receipt.

The running coordinator retry shell and DP flock shell load current code in
each future formal_eval child, so later arms pick up both fixes automatically.
A currently running arm retains its already-loaded close path. No queue was
started, stopped or signalled. Existing shell parents need the CPU-only
`formal_finalize --wait <actual completed physical run paths>` barrier after
ALL_DONE; commands and both pitfalls are in REQUESTS/Q_to_R.md. Do not treat
handoff rc0 or PENDING_EVIDENCE as paired/resume scientific completion.

CPU validation: broad ops + legacy/default-off + native MPS suite328 passed and
43 subtests; final affected suite193 passed and38 subtests; latest finalizer suite
28 passed. Regressions cover exact missing-ACK recovery across5 model families,
duplicate/missing/error/stale evidence, surviving workers, archive binding,
compression, default-bound restoration, real detached CPU finalization with
unchanged native code, pending pairing/resume refusal, and a blocked pull during
which the next job dispatches but queue PASS waits for verification. Evidence:
`qualification/preflight/close_handoff_20261010/RESULTS.json` and its CPU logs.


## Completed snapshot close and port release (2026-10-10)

`/data/mw_finetune/eval/hs6fw_c20/RECEIPT.json` is now sealed PASS,1000/1000,
with the one task34/init48 inference error resolved by retry and disclosed in
SUMMARY (`error_attempts=1`, `retry_attempts=1`, zero terminal errors). This is
valid for paired analysis. The exact original FAIL is preserved in
`RECEIPT_BEFORE_SHUTDOWN_RECOVERY.json`; CPU proof is `SHUTDOWN_RECOVERY.json`.
The worker receipt was already PARTIAL/interrupted with no worker failures.
Its native fleet PID1568507/start_ticks338798058 and supervisor1568506 were gone;
owned cleanup had forced SIGKILL after12s during snapshot building, hence exit-9
and the missing native `.transfer.json`. All288 registered worker generations
were verified gone, and the exact launch-manifest SHA matched. The compressed
quiescent rebuild preserves all4068 source files and has archive SHA
`4394e46b8101faa5419ac3a7960032bc7bc68e25135dd3b4069d4332773d4cf9`
(107360612 bytes). The compressed H100 server evidence SHA is
`4e9d10fb87ac7aa1a35bc8e73d76299c0f38567f4f52d101075783e0b56c62dc`
(6758766 bytes). The frozen native batching audit passes39098 decisions with
original treatment identity ec35c19a05d89550774b355718b5bb63a4680894dceadb45818a48be33435666.
Prepared model/spec/YAML/episode-plan pins and native runtime archive content
identity were checked unchanged; no server/worker/model/selected driver pin changed.

`ops/formal_eval.py` now drains each verified completed driver's owned fleet and
defers an interrupted snapshot exit to CPU verification. `formal_finalize.py`
records `worker_close_gap`, requires confirmed PID-bound closure, reconstructs a
missing native snapshot by value with original fleet/manifest/lock checks, and
permits only normal or interrupted native fleet receipts without unreplaced
failures. Full execution and native audits still precede PASS. The narrow
post-cleanup close exception and `formal_recovery.py --shutdown` support this
case across model families, including rc0; missing cells, live workers, actual
fleet failures, changed manifests or unresolved retries stay failures.

`ops/formal_ports.py` replaces plain startup binds with SO_REUSEADDR+bind+listen,
holds all configured test sockets together, and waits at most120s per host for
an occupied listener to release. H100 server ports and the local driver port
are checked separately; `PORT_RELEASE.json` records admission. Configured ports
remain unchanged and no port owner is signalled. A real TCP regression reproduces
plain-bind Errno98 on TIME_WAIT and proves reusable admission plus a bounded wait
for an unrelated listener without disturbing it. Existing coordinator shell
queues load these orchestration fixes in subsequent formal_eval children;
already-running children retain their loaded close path and may need CPU recovery.
No queue/campaign was stopped or reconfigured, and no GPU work was performed.

CPU evidence is
`exp/r17_integration/qualification/preflight/close_snapshot_ports_20261010/`:
expanded suite211 passed+38 subtests (including legacy/default-off and MPS),
final focused regressions58 passed. `RESULTS.json` records current Q code hashes,
C20 seals/pins and the CPU barrier. `SEAL_BARRIER.json` records every physical
finished hs5dp/hs6fw run; incomplete/running campaigns were not touched.

| Arm/source | CPU seal-barrier result |
| --- | --- |
| hs5dp_c10 | PASS1000 |
| hs5dp_c15; hs5dp_c15r0 | FAIL retained; zero accepted, MPS startup failures |
| hs5dp_c15s0 | PASS1000; valid C15 replacement |
| hs5dp_c16s0 | PASS1000 |
| hs5dp_ddim10s0 | PASS1000; its existing CPU finalizer completed normally |
| hs5dp_ddim5s0 | Running; skipped |
| hs6fw_c10 | PASS1000 |
| hs6fw_c16 | PASS1000 |
| hs6fw_c20 | Recovered/sealed PASS1000, one resolved error/retry |
| hs6fw_c32 | FAIL retained; zero accepted, port admission failure |
| hs6fw_c32s1 | Running; skipped |
| hs6fw_k2; hs6fw_k1 | Prepared only; no finished receipt yet |
| hs5dp_ctl_142626 | FAIL retained; zero accepted, port startup failure |
| hs5dp_ctl_142812 | PASS2000 |
| hs6fw_ctl_143332 | FAIL retained; incomplete1839/2000 source segment |
| hs6fw_res_152931 | PASS merged2000; use this paired control, with original runtime pin |

No finished complete study arm remains unsealed. Failed startup attempts and the
incomplete original baseline are excluded from standalone paired pricing. The
pitfalls and future seal-barrier instructions are appended to `REQUESTS/Q_to_R.md`.

## Local LIBERO workers on weilandserver (2026-10-10 owner exception)

Implemented in `ops/local_workers.py`, `local_worker_entry.py`, `formal_eval.py`, `formal_workers.py`, `formal_prestage.py`, `worker_memory.py`, `worker_gpu.py`, `cache_control.py`, `cache_lane.py`, and the CPU-finalizer overlay. `formal_eval --worker-node local --workers N` and explicit `--worker-counts timan107=8,local=16` are supported. `cache_lane --worker-counts ...` freezes the topology at preparation; one fleet is reused across every size/group. Local creation/launch is limited to today's LIBERO cache study (or its diagnostic smoke); running fleets are not changed.

Local process supervision/lease/status/cleanup and SHA-checked atomic file copies execute directly, with PID/pgid/birth-tick ownership, without tether. Each worker generation records node, EGL device/UUID, renderer reservation, and physical policy connection. Local DP connections use localhost while preserving the canonical EpisodeTask server key; H100 connections use the same campaign route as timan107. Driver connections use loopback. Local ops changes have a separately sealed `LOCAL_WORKER_ORCHESTRATION.json`; native worker-entry, client and simulator bytes stay unchanged. The explicit control exception waives only worker counts/topology and discloses both sides; model, checkpoint, sampler, resident server, batching/MPS and global control fields remain strict.

Private local Python: `/data/mw_finetune/qualification/local_workers/env311/bin/python` (3.11.15, existing binaries reused read-only). LIBERO/robosuite 1.4.0 dependencies are copied into a verified immutable release; NumPy 1.26.4, Torch 2.7.1+cu126, MuJoCo 3.6.0 match timan107. Original asset release digest is `3eb6e3e7ee3a03663dba84ccc6d6cf9dbb4ae21fb20558ca4097abc066be57dc`. No downloads, system package changes, or shared .libero config edits. All 40 tasks' original 50-state pools passed CPU validation. Final current-source CPU deployment: `/data/mw_finetune/eval/q_local_worker_final_preflight_20261010/WORKER_DEPLOYMENT.json`.

RAM admission reserves 4096 MiB per local worker and at least 32768 MiB for the host/server. EGL admission reserves 1024 MiB per local worker, accounts for other local fleets, and preserves 16384 MiB additional server growth plus 10% physical VRAM headroom. Recycling retains each already-admitted slot; use N4 recycling and restart limit2. Recommend initially DP16 local + H1008 local, total24 across both lanes. The CPU capacity probe admitted24 with 88 CPUs, load7.8 and 234.5 GiB available RAM. Actual server/other-fleet allocation can reduce admission, so these are starting counts rather than a bypass.

Successful bounded smoke: `/data/mw_finetune/eval/q_local_worker_smoke3_1791681965/SMOKE_RECEIPT.json` PASS, 57.05 seconds including startup/drain, four local workers, 12/12 clean original-init episodes with 20 controls each, four planned worker recycles. All owned parents and worker generations are gone (`FINAL_CLOSURE.json`); the fixture used a private loopback no-op endpoint and never contacted DP/H100 lane endpoints. The first two synthetic fixture attempts exposed callback/decision-metadata omissions, were stopped only through their recorded own PIDs, and remain as FAIL evidence. The corrected fixture is `qualification/local_worker_smoke.py`; it is explicitly a synthetic transport/renderer smoke, not model SR evidence.

`PIXEL_COMPARISON.json` and SHA-verified `timan107_frames.npz` compare original spatial task0/init0, object task0/init48, goal task0/init1 (seed7, ten settle controls, original 256-square agent and wrist views). Every image channel and full simulator state is byte-identical: max/mean pixel delta0, changed fraction0; original state hashes and prompts equal. OpenGL reports NVIDIA RTX4090 / driver595.91.07 locally versus NVIDIA GTX1080 / driver535.183.01 remotely. The reference renderer used its own home directory after timan107 /tmp and /scratch reported zero free blocks, with the existing node admission namespace checked/locked; no active run files were removed.

`THROUGHPUT.json`: 240 controls / 131.19 active worker-seconds = 1.829 controls/s/worker, or 0.366 H5 inference calls/s/worker. Including startup, N2 recycling in this short smoke and drain gives 1.118 controls/s/worker. This includes expensive short-episode environment creation and is not a prediction of real model throughput.

Final CPU verification: 142 tests +31 subtests PASS, including all10 new local regressions, background CPU evidence sealing, and legacy/default-off checks (`preflight/local_workers_cpu_20261010.log`). The concurrent storage fixes landed and the existing fleet-config test now mocks its new disk-admission dependency. The smaller focused suite also passed85 tests +4 subtests. P6b size-grid validation separately reported YAML seal changes while its artifacts were being written; those artifacts were not overwritten. Explicit details/coordination and commands are in `REQUESTS/Q_to_P6b.md` (thread01a12882-35d3-7201-85b3-3b62a4b417d2). Machine-readable result: `preflight/local_workers_RESULTS_20261010.json`.

For the NEXT campaigns append `--worker-counts local=16` on DP and `--worker-counts local=8` on the H100 lane. These local-only choices avoid timan107 saturation/full disk. After its disk admission passes, mixed alternatives are `timan107=8,local=16` and `timan107=8,local=8`. Keep all existing server/model/cache settings and `--worker-max-episodes 4 --worker-restart-limit 2`. Use new preparation directories/prefixes; do not alter already prepared/running fleets. Exact new size-lane preparation commands are in Q_to_P6b.md. No existing lane, queue, GPU model, or other user's process was launched, stopped or reconfigured.
## Cache continuation RAM / T1 GPU3 redistribution (2026-10-10)

OOM cause:48 Fast-WAM +32 DP cache workers were packed onto timan107 with
2048MiB reservations, although cache workers reach3.1–3.5GiB RSS. Admission
now reserves at least4096MiB for cache workers and the future footprint of
historical live cache fleets; it never alters those running fleets. T1 leases
at most16 workers,6144MiB RAM /1024MiB VRAM each, N4 recycling, physical GPU3
only. UUID GPU-ba276422-a0b1-e371-8327-f744be942f15 /PCI0000:c1:00.0 maps to
minor0; the pinned namespace wrapper masks protected GPUs0–2 before native
imports. Admission queries only that UUID. Its4MiB root GDM display is verified
and accounted for; foreign compute allocations still refuse admission.

`worker_topology.py`, `timan1_workers.py`, `cache_resume.py`, `cache_lane.py`,
`queue_retry.py`, and the formal prepare/launch/worker/transfer helpers support
same-total redistribution into a NEW segment. Original source receipts and
scientific runtime/model/client/simulator bytes remain untouched. Canonical
`worker_nodes`, total workers and the Fast-WAM batch-treatment identity stay
exactly sealed; `execution_worker_nodes` separately records the actual fleet.
WORKER_REDISTRIBUTION.json binds the change; per-worker node/device/UUID is
in the native fleet receipts. Existing immutable dependency/config releases
are revalidated and reused, including when timan107's mutable legacy LIBERO
config has disappeared. All40 original pools still pass50-state validation.

Final CPU gate:143 tests +29 subtests PASS, including9 new regressions for
T1 guard/mapping, protected-device rejection, root-display accounting, historical
cache RAM, same-total/scientific and batch pins, model-scoped lane overrides,
resume selection, allowed-root evidence and immutable dependency/config reuse.
Log: qualification/preflight/timan1_redistribution_final_cpu_20261010.log.
T1's wrapper import probe passed; all three deployment RAM/VRAM gates passed.

H100 lane relaunched and LEFT RUNNING in tmux `r17_cache_h100_ramfix2`, owned
PID729176 /birth12632676; log
`/home/weiland/projects/openpi_ext/scratch/r17_ops/cache_lane_h100_ramfix2_20261010.log`.
Fast-WAM run `/data/mw_finetune/eval/fastwamc7615306896`:
48 workers =timan10716 +local16 +timan116; local/timan107 reserve4096MiB,
T1 reserves6144MiB. Two resident replicas attested and the conductor accepted
new episodes.4311/8000 complete final-ACK cells were retained;3689 missing
cells were dispatched. Latest counts are in
`/data/mw_finetune/qualification/disk_recovery_20261010/PROGRESS.json` and
the run's driver/journal.jsonl. T1 fleet RUNNING with16 workers, GPU3/UUID
only, no worker failures; local and timan107 fleets also RUNNING without failures.

Following Cosmos is already prepared at
`/data/mw_finetune/eval/cosmosp6223294666`: its SAME8 workers all on T1 GPU3,
4000 cells, original scientific runtime/assets and settings. The lane launches
it automatically after Fast-WAM releases its owned residents/fleet.
Tim107 had97.4GiB MemAvailable with both fleets running; admission leaves
about34GiB after requested/other fleet reservations, including22GiB mandatory
headroom. No DP continuation, DP server, other-user process or protected GPU
was stopped, signalled or reconfigured.

Disk guard remains8GiB. Previously sealed Q run copies were already absent.
Recovered9.55GB by removing ONLY the closed redundant T1 transfer tar on
Tim107 after source SHA, ownership and two no-open-handle checks; its identical
SHA-verified copy on T1 is preserved and simulator installs/source receipts
are untouched. Free space rose from6.11GB to15.66GB. Audit:
`/data/mw_finetune/qualification/disk_recovery_20261010/T1_TRANSFER_RECLAIM.json`.
The first refused startup exposed the canonical batch-topology gate and exited
without starting a resident, driver or fleet; its evidence was preserved.

Future closed-source redistribution command (do not issue against this live lane):

```bash
.venv/bin/python -m exp.r17_integration.ops.cache_lane --lane h100 --directory /data/mw_finetune/eval/csl10h --redistribute-workers fastwam_libero:timan107=16,local=16,timan1=16 --redistribute-workers cosmos_libero:timan1=8 --run
```

Pitfall added to REQUESTS/Q_to_R.md: never pack48+32 cache workers on timan107;
use measured RAM reservations, canonical scientific pins and separate execution
maps. All run/progress/deployment/cleanup audits are preserved under the directory
above. No commit/push or broad cleanup commands were used.

## T1 planned recycle through the GPU3 namespace (2026-10-10)

Cause: every33 exit75 record in fastwamc7615306896_timan1 carried a valid
original N4 episode-boundary marker. The original classifier compared the
marker's native Python PID/birth against Popen's bubblewrap PID/birth, so every
planned replacement consumed the crash-restart budget. Example worker5/gen0:
wrapper4124143/birth218720933 vs native4124156/birth218720937. The third planned
cycle exhausted its restart limit. Logs/receipts/markers are copied into
qualification/preflight/timan1_recycle_20261010/FAILED_T1_EVIDENCE.json.

Fix: formal_workers launches the Q timan1_workers orchestration relay INSIDE
T1's unchanged, SHA-pinned GPU3-only wrapper. Before entering the unchanged
worker_lifetime, the relay writes NATIVE_PROCESS.json: native PID/birth,
wrapper ancestry and a unique random token assigned by the owning fleet to
this worker generation. Exit75 is a planned recycle only when this identity
record, generation token, wrapper PID/birth and original N4 marker all match.
It then uses the existing admitted-slot recycle path and spends no restart
budget. Missing/malformed/mismatched markers, other exit codes and genuine
deaths retain their bounded failure handling. Wrapper/sitecustomize/config
pins, native worker_entry, worker_lifetime, simulator/client and scientific
server runtime remain unchanged. The helper hashes are separately recorded
in LOCAL_WORKER_ORCHESTRATION.json.

CPU tests:106 passed +29 subtests (legacy/default-off and cache lane included),
CPU_TESTS_FINAL_REFRESHED.log. New test_timan1_recycle.py reproduces the real
wrapper/child PID split and four75 exits with restart-limit0; verifies the
original classifier fails and the new fleet reaches PASS; rejects borrowed
generation tokens, reused births, wrong ancestry/PIDs/N4 counts, missing
markers and unguarded/wrong-device launches. T1's actual pinned namespace
wrapper CPU proof also passed:75 forwarded unchanged, original classifier
false, bound classifier true, guard SHA checks all true; no EGL or model load.
Proof: T1_WRAPPER_CPU_PROOF.json. A concurrently added capacity authorization
expanded only the parser's upper bound to40; test_local_workers now tests41
as invalid, while the ordinary T1 lease still rejects17 without authorization.

Clean recovery: only the verified own formal_eval launcher818315/birth12708549
of fastwamc5232306668 received SIGTERM. Its normal finally path closed every
owned resident/fleet/driver/MPS process, verified by SOURCE_CLOSURE.json; the
lane parent was left running. Its cache continuation kept all5055 validated
terminal-ACK cells, including all347 accepted in that stopped segment, and
dispatched only2945 missing cells. Failed source receipts/evidence are kept.
OWNED_CAMPAIGN_STOP.json records the exact UID, PID/birth and command check.

Now LEFT RUNNING: /data/mw_finetune/eval/fastwamc5481985003, H100 two replicas,
48 workers=timan10716+local16+timan116. At the recorded progress snapshot:
5240/8000 valid cells =5055 retained +185 new.
T1 has10 observed real planned75 recycles, generation1 replacements, zero
worker failures, only EGL3/UUID GPU-ba276422-a0b1-e371-8327-f744be942f15.
T1_LIVE_RECYCLES.json and PROGRESS.json hold the live proof. Runtime archive
SHA is byte-identical to the source: RESUME_PIN_PROOF.json. Next Cosmos remains
cosmosp6223294666 (8 T1 GPU3 workers); launch-time scoped helper refresh is
required and enabled, so it also receives this relay without changing science
pins. DP continuation dpc3464789869, the 4090 residents/fleet and other users
were not stopped, signalled or reconfigured. No replacement queue was started.
The new pitfall is appended to REQUESTS/Q_to_R.md.

Final live check: 5332/8000 valid cells (277 in the new segment); T1 16 planned recycles, generations[0, 1], zero failures. Lane left running.


## Size-500 Cosmos capacity handoff

Fresh CPU-prepared run: `/data/mw_finetune/eval/cosmos500w85_220813`.
Actual execution fleet is timan107=45 + timan1=40 (85 total), external nodes
only, T1 GPU3 through the original SHA-pinned namespace wrapper. Both fleets
recycle after4 episodes; timan107 reserves4096MiB RAM/768MiB renderer VRAM per
worker and spreads EGL over0..7; T1 reserves6144MiB RAM/1024MiB renderer VRAM,
EGL3 only. H100 retains exactly2 replicas, B1, unchanged server settings.

Initially prepared57 workers (17+40) while DP32 still occupied timan107,
preserving its full128GiB future RAM reservation plus node headroom and an
extra4GiB margin. A subsequent read-only check found dpc3464789869 closed,
with zero live timan107 workers, and the DP lane's active retry
`dpc8713063796_r1_7fdb` using local33. This allowed the authorized maximum45
on timan107. DP_EXTERNAL_CAPACITY_RELEASE.json records the transition; no DP
or Fast-WAM process was stopped, signalled or reconfigured by this task.

Every4000 planned cell, size500 library input, YAML, benchmark, checkpoint and
native runtime file remains SHA-identical to the original8-worker preparation.
The original canonical scientific topology remains sealed; only execution
capacity, renderer spread and separately recorded orchestration change.
Native runtime identity5318b07bfd914c17e284569600167aa3284ba3329aa9941333eef7a1f1b90b9e
is unchanged. Scientific pins and all source seals are in CAPACITY_HANDOFF_FINAL.json.

The existing lane parent729176/birth12632676 continues Fast-WAM5481985003 and
keeps its logical Cosmos path `/data/mw_finetune/eval/cosmosp6223294666`.
That path atomically resolves to the fresh85-worker run. No parent restart or
new queue is needed. The8-worker directory is preserved at
`cosmosp6223294666_PREPARED_ORIGINAL_215910`; the intermediate57-worker run and
its alias are also preserved. PREPARED_HANDOFF_cosmos_libero.json in csl10h
records the exact active parent, predecessor, atomic exchange and seals.
CPU preclaim admission waits for RAM, renderer VRAM and disk before binding
any residents or claiming launch; the predecessor's live leases stay counted.
The T1 wrapper/recycle CPU proof passed again. Admission while Fast-WAM is
live may defer and is expected to clear after its fleets close.

H100 CPU-only lookup benchmark:800 real sealed500 feature queries across
spatial/long-horizon and corrector off/on, actual frozen retrieval, query memo
and calibrated-stall code. Worst p95=0.644ms; pessimistic mean including cold
verification=2.80ms, about715 lookup/s across2 CPU cores. This leaves lookup
CPU headroom at85 workers; retain2 B1 replicas. This measurement excludes
VAE/GPU inference, network transport and rendering. LOOKUP_CPU.json and the
fresh run's LOOKUP_CAPACITY_ASSESSMENT.json retain scope and estimates.

CPU validation:101 tests passed +29 subtests, including cache lane,
legacy/default-off, T1 wrapped75 recycling, renderer authorization,
preclaim admission, scientific-change rejection, and atomic re-handoff of
an existing prepared alias. No GPU smoke or model load was run. Logs and
probes are in qualification/preflight/cosmos_capacity_20261010/.
Worker-only scaling of the owner's5h/8-worker estimate is28min at85 workers;
plan roughly35–60min with launch/drain overhead, subject to native GPU misses.
The split is static; there is no unrecorded live fleet growth.


## Model order and incremental sync, 2026-10-10 night

Ordered execution: Fast-WAM size500 continuation fastwamc5481985003 -> Fast-WAM sizes50,100,200,300,400 (csznf) -> Cosmos500 (cosmos500w85_220813) -> Cosmos sizes50..400 (cosmossizes85_222001). Upcoming fleets are external85=timan10745+T140, original GPU3 namespace guard retained. H100 retains two replicas; all original scientific YAMLs, init cells, library sizes, model runtime and server treatments stay sealed.

Configuration staged at /data/mw_finetune/eval/h100_model_order_20261010. fast_sizes has one prepared five-size Fast-WAM job; cosmos_all has the size500 job then one prepared five-size Cosmos job. Each five-size job hot-switches all library VIEW/YAMLs on one resident fleet. Existing500 launchers close their own residents normally; safely adopting them across campaign ownership requires a separate server-lifetime contract. No live model was modified or adopted.

Q executed CPU preparation/config writes/tests only. Q did not run hold-next, cancel-waiter, any model queue, lane or sync daemon. Read-only status later showed coordinator had installed the hold and cancelled the old waiter, and the Fast-size queue child1568208/csznf was live. Former waiter was1167414/birth12977350; the cancellation helper requires WAITING_PREDECESSOR, own UID/PID birth, one exact argv-matched parent and no live owned children. It never signals Fast-WAM or its lane. The old csl10h parent naturally refuses its held Cosmos launch before LAUNCH/GPU claim; its intentional failure is not a failed Fast scientific receipt. Replacement ordering depends directly on the Fast campaign/queue. Cosmos's hold is released only by its explicit ordered queue, after the Fast-size PASS barrier and old parent closure, with the original prepared-seal binding checked.

Next coordinator commands are in COORDINATOR_COMMANDS.txt. Do not repeat the already-running Fast-size launch. Cosmos queue waits for Fast-size sealed PASS. For the already completed Fast500 campaign, ordering may hand off at PENDING_EVIDENCE only with verified driver completion, confirmed owned closure and GPU_RELEASED witness; pending evidence is never used as a control or resume source. The standalone queues keep all baseline/control checks.

Disk implementation: incremental_sync.py has a private authenticated rsync receiver, port23197, config under /data/mw_finetune/eval/incremental_sync_20261010 (prepared, not launched). Bulk chunks go directly from timan nodes to rsync://ziyanglin.com:23197/r17q/ with4096KiB/s per sequential transfer, every180s, max16 chunks/node/cycle. Tether carries only small control calls and this daemon's own password file; no evidence bytes travel through tether. Active claims are deduplicated and stale/offline historical claims are skipped.

Only exited, PID/birth-verified worker generations are prunable. Native T1 child identity is checked too. Accepted journals, all JSON/JSONL markers, receipts, manifests, locks, active workspaces, immutable installs/config/assets stay remote. Root metadata is versioned locally. Every chunk/member SHA is verified and files, receipt and directories fsynced before ACK. The remote manifest-bound INCREMENTAL_OFFLOAD index is durable before source unlink. A transport loss keeps source data and reuses one pending chunk/run. If a fleet starts closing, the helper retains originals and drops only its verified staging archive. No process is signalled by sync operations.

Finalization restores and verifies the offloaded union from INCREMENTAL_SYNC/node before sealing, including root metadata versions; tampering prevents PASS. New closed snapshots share an incremental I/O lock with pruning. Launch-time scoped worker helper refresh adds only sync_back's snapshot interlock; qualified simulator/client/model code stays exact. Older frozen worker snapshot writers do not share the new lock; any detected close/snapshot race is recoverable via the existing verified-gone snapshot rebuild, and preserved SHA-bound offload data is required for final sealing. Resume journals and accepted-cell accounting remain untouched. Data stays under local /data, not H100 weight storage.

Verification: ORDER_SYNC_TESTS_FINAL.log records129 CPU tests +2 subtests, including legacy/default-off and cache lane. ORDER_SYNC_FOCUSED_FINAL.log adds active-claim dedup/stale filtering and verifies closed-generation offload, rsync-only bulk/bandwidth limit, missing ACK/transport retention, source-change rejection, native live child protection, final snapshot union restoration, tampering barriers, private configuration and model-boundary holds. No GPU smoke or receiver/network service was started by Q. Coordinator startup is the external rsync endpoint test.

Final CPU check:18 focused tests passed. rsync exists at /usr/bin/rsync on both timan107 and timan1. Receiver config remains prepared only; Q started no campaign, lane, fleet or sync daemon.
