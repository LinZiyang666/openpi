# R8 E5 — one observational debug mode

Explorer E5; 2026-09-30. Specification and bounded read-only analysis only; no implementation or rollout.

## 1. Summary and checked inventory

**Recommendation:** add one method-independent debug observer around the existing decision/control boundaries.
Collect exact observations, decisions, issued controls and physical successors for all 500 episodes of every arm.
Complete shadow-policy and diagnostic-retrieval products from those frozen inputs, preferably after collection.
Keep experiment assignment in the serving method. Debug must never become an alternative action controller.

I read the R8 brief in full; P3 `SCHEMA_V2.md`, handback/streaming/branch contracts and collector/engine/reader code;
ordinary `closed_loop/{plugin,blind,stage_overrides}.py`, README and collection code; trace configuration,
interceptors, RNG and HDF5 sink; `profile/` and C4 tools; R7 selection/stage contracts; LIBERO and target adapters.
The bounded audit in §5 inspected four legacy HDF5 episodes, four accepted P3 episodes and eight R7 input NPZs.

| Existing component | What is actually present | Reuse / change for R8 |
|---|---|---|
| Legacy trace, `src/openpi/cache/trace/`, both interceptors | Exact served/cache/policy chunks, keys, images, embeddings, denoising states; private HIT noise under `verdict_aware`, normal MISS draw | Reuse detached-copy, action provenance and atomic-finalization ideas. Do not enable its twin/all-module serving path for R8. |
| `closed_loop/plugin.py:_BlindAdapter` | Explicitly rejects trace twins and traced policies; `_ConnPolicy` owns blind/full/camera dispatch | Add observation hooks here and at native model boundaries; leave normal `query`, `blind_step`, judge and history order intact. |
| Ordinary `after_infer`, `_write_inputs` | Rows/weights, state, served head, vision/source/reasons, timing; scalar extras capped at 24 or 40; NPZ retains scalar extras | Retain compact logs; debug adds full typed diagnostics and exact arrays before `after_infer` clears them. |
| P3 `hooks.py`, `v2_engine.py` | Private policy draws, blind diagnostic clone, full chunks and measured request-stage counts | Extract model adapters and field semantics. Do not reuse assignment/injection or replacement of live MISS noise/output. |
| P3 `telemetry.py`, `snapshots.py`, `worker_v2.py` | Per-control physical state, object observations, predicates, contacts, actuator substeps, pre-infer snapshots | Reuse runner dependency injection and read-only capture; replace P3-marker dependence, repeated JSON arrays and duplicated state. |
| P3 `stream_*`, `dispatch_fence.py` | Offset ACK, conflict detection, durable receipts, bounded queue/spill, persistent dispatch generations | Reuse tested transport semantics and latest attempt fence; extend explicit filename/schema allowlists for new block files. |
| P3 `read_v2.py`; C4 `common`, `profile_report`, audits | Strict accepted-attempt/control joins, cost accounting, camera/follow/stage/failure analyses | Factor generic joins out of P3 treatment checks; add a schema adapter. Existing P3 reader cannot directly read ordinary arms. |
| `profile/{coverage,explain,breakdown,compare,timeline,repr,runprof}` | Library/harness tools with useful metrics, bootstrap and provenance views | Reuse functions; their harness-format CLIs are not universal rollout readers. Explicit CPU limits, never whole-machine defaults. |

**Surprises that affect the design:**

- Contrary to the brief's image assumption, **0/8 sampled ordinary R7 input NPZs contain image arrays**.
  The current `_write_inputs` implementation saves keys/state/actions, not images. The two inspected L10 arm trees
  also expose no image payload. Other possible remote artifacts were not audited; do not claim they exist.
- Legacy trace has no simulator-ground-truth writer or fields in the inspected first-step trees. Its images and
  actions remain valuable historical evidence, but cannot supply missing object/contact labels for R8.
- P3 is a research controller as well as a collector: a selected CALL can execute its private shadow sample.
  CPU cache parity with `p=0` does not prove stochastic policy parity with ordinary CU/CT or pure inference.
- P3's `event_labels.grasp_slip_release` is explicitly **unannotated**, not a ready-made grasp/slip ground truth.
- Repeated before/after state and entity-name JSON is expensive; raw keys, rather than images, dominate the
  inspected R7 NPZs. Lossless collection needs deduplication and binary arrays before it needs selective episodes.

## 2. Profile tools wanted

Costs below are engineering estimates on one CPU core, excluding GPU shadow inference; D≈60 decisions/episode,
300 controls/episode. Report actual runtime/bytes after pilot. No tool below needs an active server or simulator.

| Tool | Question | Inputs: existing / new | Outputs | Stage decision informed | Priority; rough CPU cost |
|---|---|---|---|---|---|
| `debug validate` | Is the apparent stage effect a missing/joined-wrong record? | Existing journals, P3/C4 joins; new IDs, manifests, payload receipts and capability masks | Coverage matrix, conflicts, accepted/error prefixes, per-arm PASS/INCOMPLETE and exact missing fields | Admit datasets before any boundary/value estimate | P0; metadata 1–3 min/arm; payload hashing is I/O-limited |
| `debug parity` | Does instrumentation change any chosen/executed action or live state? | Recorded observations/noise/order; paired debug-on/off outputs and state digests | First differing decision/control, dtype/shape/bytes, RNG/history/camera/cost differences | Admit debug mode, independent of segmentation quality | P0; CPU method replay 1–10 min/arm; real-model verification is separate |
| `debug provenance` | Which demonstrations/stages support each continuation? | Library row maps, frozen StageTable; new full member provenance, continuation offsets | Stage/mode mass, unknown mass, effective episode count, cross-stage kernels, unsupported tails | Distinguish bad segmentation from sparse support or interpolation across boundaries | P0; 0.5–3 min/arm; O(16D) |
| `debug divergence` | Where does cache disagree with policy, and when does blind drift start? | P3 chunks where available; new same-observation shadows, live/diagnostic keys and actual control mask | Valid-dimension RMS by future offset, gripper disagreement, noise floor, blind-age and stage curves | Test whether proposed hard stages predict error on held-out tasks/models | P0; 1–5 min/arm; no pairwise library scan |
| `debug timeline` | What physically happened before/after each decision? | C4 failure_clock; new per-control poses/contacts/predicates and decision images | One episode HTML/JSON timeline: frames, motion/contact/predicate changes, proposals, decisions, termination | Validate stage boundaries against manipulation events; support E3 diagnosis | P0; numeric pass 1–5 min/arm, lazy frame rendering ≈1–5 s/episode |
| `debug budget` | Did a stage gain calls or merely cost more? | C4 accounting; new live/profiling dispatch scopes, camera/tail reasons and actual controls | N/V/M, camera completions, partial tails, owner IR, profiling cost and coverage by stage | Fair CU/CT and look-lever comparisons | P0; <1 min/arm after normalization |
| `debug value` | Is a stage's randomized call associated with improved continuation? | C4 stage_value; new exact pre-draw probabilities, overrides and eligibility | Positivity/ESS, stage excursion and occupancy estimands, paired task/init intervals | Retain or reject allocation signals without confusing divergence with call benefit | P1; 5–20 min/cell at 2,000 cluster bootstraps |
| `debug camera_follow` | Is wrist selection or continuation causing the error? | C4 camera/follow audits; new full/wrist diagnostic keys, fixed-fit proposals, physical drift | Same-observation camera contrasts, valve lead time, support and gripper transitions | Separate view loss, blind duration and stage-gate contributions | P1; 5–30 min/cell with ≤2,000 sampled diagnostic retrievals |
| `debug boundary_export` | Can E1 compare segmenters without writing another loader? | New canonical decisions/controls, versioned library labels, offline event annotations | Aligned feature/event tables plus train/validation/task/model split IDs | Compare event, motion, object-relative and change-point segmenters fairly | P1; 1–5 min/arm; no new serving labels |
| `debug snapshot_audit` | What blocks trustworthy branching? | P3 skipped-field inventories; new linked client/server state and restore manifests | Coverage, selected-point propensities, missing restore components, certification prerequisites | Decide whether E2 can request a later branch experiment | P1; seconds/episode; actual restore test is future simulator work |
| `debug capacity` | Can the remaining campaign finish within disk/time limits? | Receipts, stage dispatch totals, queue/spill/serialization counters | Per-field bytes, mean/p95 per episode, remaining-byte/time forecast | Protect full trajectories across stages instead of losing long failures | P0; <1 min/arm |

All tools consume the same accepted-attempt view and export machine-readable tables plus a short report.
Do not join different policies by their decision number and call that the same physical state: pair task/init,
then expose each arm's own control clock and optional event alignment. Keep raw and aligned times together.
Derived failure/stage labels carry detector version, evidence and uncertainty; privileged labels never enter serving.

## 3. Debug mode contract and required fields

### 3.1 One switch, independent of the method

Proposed interface: `--debug-profile <frozen-config>` on either model entry point, plus a matching generic client
adapter selected by the campaign manifest. It enables a common schema, never `--os-method ...:Profile`.
The same observer supports native/pure policy, A, CU/CT, blind follow and wrist arms. It does not require retrieval
to exist: nonapplicable retrieval fields have a status and reason. A method-specific diagnostic adapter is optional;
the generic observation/response/control record is mandatory. Unsupported diagnostics never call a live method twice.

The client generates `decision_id` and observation digest before `infer`; the server echoes a small `__debug__`
envelope with schema, ID, source, anchor/tail relation and response digest. No tensors or privileged state ride in
this envelope. Client collection must work on ordinary arms and never search recursively for a P3 marker.
Older servers may be captured with explicit `server_join=unverified`; those runs cannot pass R8 completeness.

Canonical key: `(campaign_uuid, arm_id, task_uid, dispatch_generation, decision_id)`; controls append `control_id`.
Also record driver incarnation, server process/connection incarnation, suite/task/init, environment seed and reset
hash. Persist the dispatch fence **in both stream and file mode** before dispatch; P3 currently installs it only
in stream mode. Pair scientific comparisons by `(suite, task, init-pool hash, original init, replicate)`, not UID text.

At server entry copy exact wire inputs before transforms. After normal decision selection, copy proposal/result
and structured diagnostics before cleanup. Tap the actual normalized policy/cache output and final wire response.
At the client, capture reset/settling, every issued control and its resulting observation/physical state, including
the last partial chunk. A returned H-row chunk is a proposal; only the client proves which rows were applied.

### 3.2 Fields and retention

Sizes are decimal uncompressed estimates unless marked compressed; dynamic entity/contact counts are explicit.
`status={available,not_applicable,not_sampled,unsupported,error}` plus reason replaces ambiguous null/zero values.

| Field / contract | Side and cadence | Why | Size estimate | Priority |
|---|---|---|---|---|
| IDs, clocks, control interval, source and parent/anchor IDs; accepted outcome, termination reason | Both/decision/control; episode | Exact joins, partial commits, retries and wait controls | 0.3–1 KB/record before column encoding | P0 |
| Frozen code/config/checkpoint, normalization/output-adapter, fits/library/stage-table hashes; RNG algorithm, runtime/dtype/batching versions | Episode references to campaign catalog | Reproduce proposals; protect against changed libraries/labels | 20–200 KB/catalog; <2 KB/episode references | P0 |
| Interface: camera names/order/masks, state/action dimensions, valid indices, units/frames, horizon, control rate, gripper convention, preprocessing | Episode/catalog | Prevent LIBERO or model-specific assumptions | 5–20 KB/catalog | P0 |
| Exact wire RGB for every decision, including blind/wrist calls; shape/dtype/hash; acquisition time and encoded/used mask | Server/decision, or client-owned identical payload | Later full-look shadow and visual diagnosis without rerendering | Pair: 301 KB π0.5, 393 KB GR00T raw; budget 100–200 KB compressed | P0 |
| Exact original wire state/prompt and normalized state; preprocessing fingerprint | Server/decision; prompt once unless changed | Old GR00T float32 raw-state logging loses input precision | <1 KB/decision; preserve original dtype, including float64 | P0 |
| Live raw keys, actual camera mask; separate diagnostic full-look keys and compact PCA keys | Server/decision where computed; augmentation/decision | Replay actual retrieval; analyze blind/wrist alternatives without contaminating live history | Full pair 262 KB float32; compact keys <1 KB; one payload when identical | P0 |
| Cache proposal, actual normalized chunk, wire chunk, initial live policy noise when sampled, valid execution mask | Server/decision + client verification | Compare semantics and reproduce actual policy outputs | ≈5–12 KB/decision for H=10/16, 32 padded channels | P0 |
| Policy alternative at every decision; noise/seed/domain, observation/weight hashes, output transform, origin=live/deferred/inline | Server or augmentation/decision | Same-observation disagreement across every stage and blind age | ≈3–6 KB/decision including noise; never label it an executed trajectory | P0 |
| Every member's library hash/row/episode/task/step/success, score, weight, source chunk/control offset, next-row chain, frozen stage/mode/event/unknown labels | Server/decision referencing catalog row map | Prove what a blind continuation means and recover neighbour chunks exactly | ≈1–3 KB/decision for 16 members; library chunks stored once | P0 |
| Full result extras and versioned method diagnostics: raw tests, thresholds, verdict/priority, all vetoes, selected reason, latch/budget state | Server/decision | Uncapped gate/stall/stage/camera decisions, including failed checks | Budget 1–8 KB/decision; arrays in payload blocks | P0 |
| Randomization: eligible, nominal/effective conditional propensity, coin/domain/seed, overrides/forced source, remaining budget, continuation package | Server/decision before draw plus realized choice | Causal support and correct event-stage comparisons | <1 KB/decision | P0 for CU/CT/randomized arms |
| Actual live vs profiling invocation counts for each camera/stage, completion and resampling; timing scopes/queue wait | Server/decision; receiver/client metrics | Preserve deployment IR; quantify real collection cost | <1 KB/decision | P0 |
| Action sent to env, transformed/clipped action where adapter exposes it, reward/done/termination, before-state reference and after-state | Client/every control, including settling | Applied dynamics and partial-tail proof | 0.2–1 KB/control, physical arrays below | P0 |
| qpos/qvel/act/ctrl, EE/finger state, all movable object/articulation poses and velocities; numeric observations; explicit units/frames | Client/every control; initial state once | Object-relative geometry, slip/drop/stall diagnosis | ≈4–12 KB/control raw; preserve simulator float64 | P0 |
| Contact geom IDs, position/frame/gap, constraint address; body external wrench/constraint forces; predicate values/info | Client/every control; entity/geom/body/joint/actuator maps once | Evidence for contact/alignment/goal changes; not automatic intent labels | ≈2–12 KB/control, ragged contacts/constraints | P0 |
| Actuator commands at each exposed physics substep, count/time and capability status | Client/every control | Distinguish issued commands from low-level actuation | Sample: 25×9×8 = 1.8 KB/control raw | P0 when exposed |
| Physics/controller/wrapper/RNG snapshot; method/controller export or replay-prefix reference; remaining horizon, action queue/tail, server RNG context | Both/pre-decision, stable sample plus initial/final state | Preserve branch feasibility without claiming restoration | ≈20–100 KB/selected point after image dedup; measure in pilot | P1; `restore_certified=false` |
| Two additional private policy draws, selection probability and noise domain | Augmentation/1 in 32 decisions, preselected by identity | Noise floor for E1/E2; distinguish sampler variance from cache error | ≈3–8 KB/selected point plus head compute | P1 |
| Full/wrist/other-view fixed-fit diagnostic proposals; separate feature masks and provenance | Offline augmentation/selected observations | E4 camera mechanism, including policy-input completion | Keys referenced above; ≈1–4 KB/proposal | P1 |

Retain full padded chunks for reproduction, but all action metrics use declared valid channels and executed masks.
For current LIBERO that means seven channels; GR00T padding is noise-like. Do not infer gripper polarity from sign.
For library provenance, freeze a row metadata table once and retain the hashed action arrays/fit artifacts needed
to resolve every reference. Do not copy 16 identical library chunks into every decision or reinterpret old rows
through a newly fitted stage table. Fresh diagnostic retrieval gets its own provenance and `observed_by_method=false`.

Not worth default collection: every transformer token/attention tensor, all denoising iterates, duplicate raw/model/
input images, repeated prompts/entity names, duplicate before/after arrays, per-control RGB/video, and full neighbour
chunks already recoverable from a frozen library. These dominated legacy costs without answering E5's stage questions.
Per-control contact/pose data remain complete. Event-triggered existing frames and intermediate tensors are P2
extensions with declared sampling; do not add extra renders or simulator steps to obtain them.

### 3.3 Shadow policy, snapshots and behaviour invariance

**Default shadow scheduling: deferred augmentation, within the same debug data contract.** Store exact observation,
model provenance and diagnostic seed for every decision. After rollout collection, run a dedicated read-only model
adapter on those observations, with explicit private generators and isolated caches. Actual policy calls already
supply the live policy sample; blind/policy-tail/cache decisions need fresh full-policy alternatives. Deferred
outputs are labelled replay-origin, not claimed numerically identical to hypothetical concurrent batched calls.
Never send them back to the method. Freeze weights/adapter artifacts before starting the campaign.

Inline shadows are an optional execution setting only after model/camera/concurrency parity passes. On a real
MISS preserve its existing generator, noise draw, call order and output; do not replace it with P3's seeded sample.
On HIT/blind use a named private generator, never global seed/reset/save-restore across concurrent connections.
Mutable key builders, deferred-camera objects, method state and model-side caches must be isolated; share only
verified immutable weights/results. Wrist diagnostics must not complete/mutate a live deferred prefix or its counters.
If an adapter cannot perform a pure diagnostic query, capture data and mark that probe unsupported.

Private seed domains include campaign seed, paired episode identity, decision and draw index. Exclude wall-clock,
worker and arrival order; keep attempt identity for auditing. Declare whether retries reuse the diagnostic seed.
Use the same seed convention in paired arms, while acknowledging diverged observations and control histories.

Snapshot policy: save reset plus first active pre-infer point; thereafter select each pre-decision point with a
stable hash probability 1/16, independent of action, stage and outcome. Store selection/probability on every point;
initial selection has probability 1. Include blind decisions. Save terminal state, but do not call it pre-treatment.
Capture image references rather than image copies. Do not serialize whole Python objects/pickle controller state.
Export supported numeric/typed state and an explicit skipped inventory. Method exports must include anchor, tail,
histories, counters and latches; a certified prefix replay is an alternative, not an assumed restoration method.

**Required proof, to be executed by implementers/coordinator after development:**

1. Freeze a corpus spanning both models, both suites, policy/cache, random/forced MISS, blind/tail, full/wrist,
   stage/valve LOOK, reset/reconnect, duplicate request and early terminal controls. At least 20 episodes/cell.
2. Replay identical observations and supplied live noise through debug off, capture only, and capture+shadow;
   preserve request/batch order. Require byte-identical normalized and wire actions, member weights, verdicts,
   camera choices, method histories/latches and deployment counters. Verify RNG states/call consumption unchanged.
3. Exercise multi-connection interleavings, queue pressure, injected diagnostic failure and sink spill. Response
   contents and method state stay identical; collector failure changes a telemetry status, never a chosen action.
4. Coordinator repeats closed-loop non-test episodes with identical reset state, seeds and controlled dispatch;
   compare issued action bytes and physical successor sequences. CPU method replay alone is insufficient.
5. Separately validate practical concurrent runs. Current global policy RNG, shared quantile controllers/global
   decision clocks and batch-size-dependent numerics can turn timing changes into action changes. Record dispatch
   and batch identity; a tolerance-only replay is not a byte-parity certificate. If parity fails, use deferred
   shadows and controlled scheduling, or fix request-local serving randomness in both on/off configurations and
   rebaseline that explicit serving change. Do not silently reseed ordinary policies only when debug is enabled.

This is a conditional, testable invariance contract, not a proof that arbitrary timing-sensitive methods are
unaffected by instrumentation. Low overhead and isolated RNG alone cannot establish that stronger claim.
No-intervention snapshot restoration must reproduce controller/wrapper state, solver continuation, RNG, history,
remaining horizon, actions, rewards, predicates and images across contact-rich continuations before branching.
Until then, retain `restore_certified=false`; same-state action alternatives are not causal outcome counterfactuals.

### 3.4 Storage, streaming and acceptance

Use one versioned logical schema with independently written server/client/augmentation partitions. On producers,
write compact JSONL event/offset metadata and immutable compressed NPZ array blocks (e.g. 16 decisions / 64 controls,
up to 8 MiB raw); no object arrays. Pack image tensors, contacts with offsets and float64 physics arrays losslessly.
This avoids a new client dependency and millions of per-decision files. Keep a campaign entity/library catalog once.
After verification, derive columnar analysis tables; avoid generating giant CSVs with repeated chunks/images.
Retain original raw payloads; derived tables and videos are disposable, independently versioned products.

Reuse P3 offset-ACK/overlap/SHA receipts and sticky spill, with a new debug schema and permitted block filenames.
The collector must learn to validate these blocks; pointing its old filename parser at new NPZs is insufficient.
Server local writer and client sender need separate 8 MiB bounded queues; 88 client queues cap at ≈738 MB.
The receiver writes `/home/.../r08/runs/<arm>/debug/<attempt>/`, atomically publishes files and per-side receipts.
Track campaign+arm+attempt IDs, queue high-water, serialization time, spill bytes, durable offsets and last sequence.
Spill/receipt work is outside action selection; unresolved capture failures retain episode outcomes plus explicit
data incompleteness. Do not silently discard a failed robot episode or reinterpret it as an infrastructure failure.

Scientific journal acceptance, capture-complete, augmentation-complete and analysis-ready are separate states.
R8 arm completion requires exactly 500 accepted task/init pairs with complete required capture **and** shadow
coverage, reconciled N/V/M and actual controls, content hashes and matching reset/outcome identities. Retain
unaccepted/error prefixes. A missing camera/object capability must fail that arm's required-field contract before
the full run. Optional unsupported fields remain explicit. Debug does not mutate the journal's success definition.
An emergency capture loss must be reported with its original outcome; recollections are labelled additional
attempts and selection/missingness audited, never substituted according to success.

At 50 MB/episode, 56×500 episodes = **1.40 TB**; at the brief's 60 MB = **1.68 TB**. Four optional arms add
0.10/0.12 TB. Reserve 0.20 TB for receipts/derived products/spills/artifacts and 0.30 TB free-space headroom.
Including those reserves, 56 arms at 60 MB need **2.18 TB**, 0.08 TB above the stated 2.1 TB; 60 arms need
**2.30 TB**, 0.20 TB above it. At 50 MB, the totals are 1.90/2.00 TB. Use archival or measured size reduction
before admitting an over-budget plan. Observed free space was 2.206 TB, not a reservation.
Budget ≤50 MB mean/episode; long L10 episodes may reach 80–100 MB. Never truncate long/failing episodes to hit a cap.
For 104 decisions, uncompressed GR00T images+two full keys alone are ≈68 MB; compression estimates need a pilot.
Keep one canonical copy, avoid automatic full-run tar duplicates, and pin referenced libraries before any archival.
If forecast exceeds the envelope, defer P2 arms or have the coordinator inventory/move old unused data to `/archive`;
this explorer authorizes no moves and performs none. Remove no raw field from an in-progress arm.

At the R7 5,000 episodes/hour rate, 50–60 MB/episode requires **69–83 MB/s** end-to-end aggregate capture capacity
(server local plus client network), ≈0.8–0.95 MB/s per 88-worker equivalent. Network load is the client portion.
P3's reported 3.42 MB/s paced and 21.88 MB/s burst synthetic tests do not certify that load. Require a coordinator
pilot/load test at ≥100 MB/s aggregate production+ingest and at measured client peak traffic, with final hashing,
compression, reconnect/spill and no silent loss. Batching/fsync overhead, not disk capacity alone, may bottleneck.

### 3.5 Portability

Separate `ModelAdapter` (wire transforms, noise, action mask, stages/cameras), `MethodObserver` (opaque versioned
diagnostics) and `EnvironmentAdapter` (control/physics/capabilities). The common schema has named cameras/joints,
arbitrary action structure, actual timestamps and optional contacts/predicates/snapshots. Privileged fields remain
client-side diagnostic channels and are never appended to `QueryView` or policy inputs.

Local evidence: `src/openpi/policies/robocasa_policy.py` requires three real π0.5 camera inputs and emits 12 action
channels; `exp/robocasa365/episode_runner.py` includes base motion and model-specific action conversion. Record
those conversions and base/world/EE frames; do not transplant LIBERO's two-camera, 8-state, 7-action constants.
`src/openpi/policies/metaworld_policy.py` has one real camera and four action/state channels, with masked wrist
slots. Its four-state payload does not establish that this checkpoint reads state. Preserve input/usage masks.
Each simulator version needs a capability/restore adapter and verification; P3's `env.env.sim` path is not universal.
Real robots retain timestamps, commanded/measured motion and available sensors, but report object/contact truth
and restore support unavailable unless independently measured. Do not infer real grasp success from a gripper command.

## 4. Arms requested — all 500 episodes, standard paired 10×50 test set

Cells: **all8** = {π0.5, GR00T} × {L10, Spatial} × {50, 500 demo episodes}; **sparse4** restricts library=50;
**pi4** = π0.5 × both suites × both libraries. Pure policy needs no library axis; diagnostic cache fits can be
evaluated against both libraries on the same policy trajectory without creating duplicate policy arms.
Use R7 frozen A/stall/stage/camera fits; CU/CT use ρ=.30 sparse and .18 dense with their frozen budget solve.
No stage labels or thresholds may be selected from these test outcomes. New stage candidates are offline research.

| Owner group | Cells and variant | New arms | Why | Priority |
|---|---|---:|---|---|
| 1: pure inference | Both models × both suites; fresh full policy every 10 controls, client decisions every 5 with explicit policy tail | 4 | Current-cadence physical reference; old L5 traces lack ground truth and are a different cadence | P0 |
| 2: without stages | all8: A, 10-control commit | 8 | Isolate cache error from random/stall calls and stage gating | P0 |
| 2: without stages | all8: CU (A + calibrated stall + uniform hashed-coin calls), fixed ρ above | 8 | Owner's non-stage system and natural randomized opportunities | P0 |
| 3: R7 stages | all8: CT, same fits/ρ/stall and R7 event+first-deviation-entry tilt, λ recalibrated as prescribed | 8 | Measure what the actual crude stage system misses; preserve all raw labels | P0 |
| 4: look less | all8: UF1, E=1 extra five controls, structural support only | 8 | Uniform 15-control opportunity control | P1 |
| 4: look less | all8: SF1, same E=1 plus stage and state valve | 8 | Existing gated extension with complete physical traces | P1 |
| 4: gate decomposition | sparse4: V1, E=1 plus state valve only, no stage veto | 4 | UF1→V1 isolates valve; V1→SF1 isolates stage at the same extension cap | P1 |
| 4: look half | pi4: wrist retrieval at every eligible anchor after mandatory full first/reset look; standard 10-control cadence | 4 | Camera lever without R7 stage selection; full completion on any actual MISS | P1 |
| 4: look half | pi4: R7 SW, identical cadence with stage-gated wrist eligibility | 4 | Uniform-wrist→SW estimates effect of stage selection; both compare with A | P1 |
| 5: interaction | pi4: SF1+SW with the exact constituent gates | 4 optional | With A/SF1/SW, completes a 2×2 lever comparison under the same gates | P2 |

**28 P0 + 28 P1 = 56 arms / 28,000 episodes; optional group 5 gives 60 / 30,000.** This is a prioritized
integration envelope for the coordinator/E4, not an additional parallel campaign. The valve-only comparison is
deliberately sparse-only; it does not identify dense gate interactions. E=2 and GR00T wrist execution are deferred
(GR00T has no validated one-camera serving path). Retain old L5 policy traces as historical evidence only; do not
claim their SR or missing telemetry belongs to the fresh L10 reference. No mandatory new branch arms before certification.

Every arm gets decision images, complete control ground truth and a policy alternative at every decision. Do not
confound intervention groups with different telemetry sampling. P1 snapshot/noise samples use the same protocol.
At 60 decisions/episode this is 1.68 million decisions; at most 1.68 million base shadow forwards, less when live
policy samples are reused, plus ≈105,000 extra action-head draws for the 1/32 noise sample. Log actual counts.
The R7 rate implies a **5.6-hour collection lower-bound planning baseline**, not a shadow-enabled forecast. Deferred
shadow processing at hypothetical aggregate 20/50 decisions per second adds up to **23.3/9.3 hours**, before extra
heads/I/O; those rates are scenarios, not measured capability. Profile GR00T and π0.5 separately on non-test data.
Full model weights also invalidate stage-1-only lane memory assumptions for inline shadows; four lanes cannot be
assumed available. Freeze the lane/throughput plan after coordinator admission, without reducing the 500 episodes.

## 5. Preliminary evidence and reproducibility

`audit_existing.py` and `evidence.json` are beside this proposal. The completed bounded scan took 0.17 s locally;
it inspected metadata/short prefixes, not entire datasets. It imported no model, simulator or server code.

```bash
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r08/ideation/E5_debug_architecture/audit_existing.py
```

| Sample (one accepted P3 A episode/cell, library 50) | Decisions | Client total MB | Controls JSON MB | Snapshot count / MB | Per-decision server NPZ total MB |
|---|---:|---:|---:|---:|---:|
| π0.5 L10 | 33 | 10.395 | 7.892 | 17 / 2.503 | 4.722 |
| π0.5 Spatial | 17 | 10.755 | 9.476 | 9 / 1.279 | 2.439 |
| GR00T L10 | 33 | 10.782 | 7.826 | 17 / 2.956 | 7.365 |
| GR00T Spatial | 16 | 10.495 | 9.151 | 8 / 1.344 | 3.565 |

These totals exclude server decision JSON, ordinary input NPZ duplicates, receipts and archives; they are not mean
episode sizes. First three active controls occupy 43–106 KB/JSON line, 13–31 KB with simple level-1 zlib; all twelve
have 25 captured actuator substeps and no missing-actuator flag. Predicates are exposed in these samples; grasp
labels remain unannotated. Lossless typed-state/deduplicated estimates in §3 still require measurement.
Sample legacy traces cost 4.25–4.26 MB/decision for π0.5 and 2.888–2.893 MB for GR00T; no simulator field is present.
Sample R7 input archives span 11.74–27.91 MB; raw key arrays account for ≈97.7–98.3% and image arrays are absent.
Exact paths, archive members, first-step HDF5 shapes/dtypes and sample-control availability are in `evidence.json`.

## 6. Risks, open questions and handoff order

- **Highest risk: false invariance.** Global RNG/batching/arrival-dependent methods need controlled proof; P3's
  injected seeded CALL semantics and legacy trace twins must not enter the new observer accidentally.
- **Highest data risk: absent images or reused attempt IDs.** Generic pre-transform capture, persistent fences,
  exact field manifests and accepted-attempt receipts are release requirements, not later report improvements.
- **Shadow interpretation:** policy disagreement is neither a failure label nor causal call value. Deferred shadows
  need their own numerical-validation report; snapshot branches remain uncertified. Causal tools need positivity.
- **Size/throughput are unproven.** Small successful examples underrepresent long failures, complex scenes and
  contact bursts. Admit storage/transport on stratified non-test trajectories and a worst-case offered-load test.
- **Generality has explicit boundaries.** A new method can always be observed; safe internal cloning, restored
  histories, useful stage labels and environment predicates require declared adapters and capability tests.
- **Handoff order:** agree IDs/interface/reader → generic client/control capture and image tap → ordinary method
  diagnostics and immutable catalogs → deferred shadow adapters → parity/completeness/capacity admission → full
  P0 arms → P1 arms → optional P2. Collection infrastructure must pass before committing the 500-episode arms.

All exploration writes stayed in this E5 directory and `/tmp/r8_E5_debug_architecture/`; Python used CPUs
30–33,74–77 with one numerical thread and CUDA hidden. No git operations, services, remote work or simulations ran.
