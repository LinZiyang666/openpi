# P3 R6 — superset Profile v2

This handback supersedes the campaign in `HANDBACK_V1.md`. The profiler remains
one opt-in mode, now with experimental design dimensions. No shared plugin,
`src/`, ops, other-round file, server or client-host installation was changed.
V1 source serving modules remain intact for existing artifacts/smokes. V2 uses
`exp.offline_search.rounds.r06.p3_profiling.v2:Profile`; `enabled=false` still
delegates. With an A base and fixed p=0, executed normalized chunks and vision
cadence equal A in CPU replay. Optional B with p=0 means B's guard/controller
semantics, including its existing cache early-look veto. Profile policy samples
have explicit private seeds; this is not a claim of sample-wise parity with an
unseeded real-model B run.

All Python commands below used repository `.venv/bin/python`, affinity
`2-5,46-49`, OMP/OPENBLAS/MKL threads=1, CUDA hidden, no bytecode, PYTHONPATH=.:src.
No server, LIBERO worker, chain, tmux, port, remote host, GPU inference, git, src
edit or review-test read was performed. At most two matrix children per matrix
were used; overlapping CPU jobs stayed within eight processes.

## Delivered behavior and files

- `v2.py`, `v2_engine.py`: versioned Profile; anchor and blind full-policy
  shadows, isolated blind keys/retrieval, alternative wire chunks, full neighbour
  chunks, measured stage dispatch counters, library/fit/interface provenance,
  modality features and guard calibration. Original `method.py`/`hooks.py`
  remain the v1 implementation. V2 installs a version-selecting dispatcher in
  the importing process; other methods and v1 fits use their previous Engine.
- `design.py`: independent keyed source, duration, hold, delay and episode-dose
  assignment; pre-guard support, hard caps, prefix/cooldown, actual propensity
  and every override. Delayed and held interventions are explicit packages.
- `calibrate.py`: own-library CPU prefit emissions: successful-library
  episode-max empirical reference tables, V7/guard statistics, effective alert
  shares and LOEO residuals. Fitted transforms stay frozen; these are diagnostic
  p-values, not deployment guarantees or an invented universal quality score.
  All eight exported tables are delivered in `calibration_v2/`: 2,005 successful
  library-episode records and 36,688 pseudo-anchors, not new evaluation episodes.
- `telemetry.py`, `worker_v2.py`, `run_gtp_v2.py`, `run_arm_v2.sh`: stock-loop
  dependency injection; complete control/lineage/terminal skeleton, privileged
  physical measurements, exposed task predicates, physics-substep actuator
  commands where the simulator method is writable, and sampled snapshots/RNG.
- `read_v2.py`: strict server/client/outcome joins and tidy anchors, decisions,
  controls, neighbours, action steps, episode and attempt tables.
  `pilot_stats.py`: task-demeaned within-init ICC from episode summaries.
- `campaign_v2.py`, `arms_v2.json`, `schedule_v2.json`, `manifests_v2/`: one
  pilot plus disjoint continuation; ordinary emit_arms format and `<RUN>` paths.
  `prefit.py` also supports v2. `audit_smoke.py` audits the existing real v1 run.
- `test_v2_matrix.py`, `test_v2_units.py`, `test_v2_reader.py`: CPU fixtures and
  real-plugin/store replay. `replay.py` only gained a version-aware marker check.
  `SCHEMA_V2.md` is the detailed field, cost, timing and estimand contract.

`results/install_manifest_v2.json` records final UTC install times and SHA256s.
No shared file required an atomic install; its hash is checked against v1.
Client NPZ writes use a temporary file in the destination directory followed by
`os.replace`. This delivery does not install anything on a client island.

## Complete wishlist disposition

Codes: **a** already in v1 (field named); **b** added to this same run;
**c** configurable experimental dimension of Profile/campaign;
**d** paired branches required; **e** out of scope/unavailable, with reason.
Compound requests are split where they have different dispositions. Counts in
the wishlist are planning targets, not quantities that software can guarantee.

| Request | Class | Field, implementation or exact limit |
|---|---|---|
| Q1.1 proposed cache/policy, normalized residual, retrieval inputs | a | `cache_chunk`, `policy_chunk`, `distance.per_step_rms`, v1 input NPZ keys/state; retained |
| Q1.1 applied transitions, clipping/scaling, intermediate/terminal links, masks and provenance | b | `controls.action_issued`, `actuator_ctrl_each_physics_step`, before/after physical state, chunk offset, timestamps; `read_v2` after-head/commit, actual count and truncation; candidate_wire and catalog hashes. Missing actuator/predicate APIs have explicit reasons. Model-weight hash requires coordinator provenance. |
| Q1.2 existing assignment/support/history | a | `assignment.propensity`, policy seed, baseline_hit, actual_propensity, executed_policy; outcome join |
| Q1.2 support at forced guards and alternate continuations | c | `pre_guard`, logged suppression/override, durations/holds/caps; actual propensity 0/1 is unsupported locally, never treated as randomized |
| Q1.3 paired unserved successor/SR and restoration audit | d | Snapshot + complete input/control trail + both alternative wire chunks are retained. Exact resume is uncertified; restore requirements below. Later matched replay and branches must be run by coordinator. |
| Q1.4 A/B/P10 complete-policy occupancy | c | A, B and P10 whole-episode cohorts; no reweighting claim that mixed trajectories are A trajectories |
| Q1.5 existing bank content/fit/normalization identity | b | Hash-addressed source fits, manifest with episode/source metadata, calibration artifact; raw inputs permit offline alternative-bank search |
| Q1.5 serve another frozen fit in this program | c | Same Profile base_spec/base_kwargs/base_fit/guard_fit and store-root arm dimension; no new instrument needed |
| Q1.5 independent bank acquisition, 24 bank realizations and their outcomes | e | Not supplied by eight existing banks; must construct/freeze additional banks and allocate actual outcome episodes. Logged alternatives cannot supply their SR. No new banks created. |
| Q1.6 physical changes, contacts, goals, first completion | b | Privileged control-resolution pose/velocity/state/contact/predicate/`done` records; first/lost predicates can be derived |
| Q1.6 curated failure/intent/grasp-slip labels | e | Contact or gripper sign alone cannot label intent; raw channels saved, semantic annotation/validated event adapter not invented |
| Q1.7 conditional policy variability | b | Primary + three fresh-head samples at independent 1/16 anchors, seeds/chunks/dispersion/timing; extra heads charged |
| Q1.8 within-blind visual coverage | b | Full isolated shadow/keys at every blind decision, plus cloned retrieval/guard diagnostic; live cadence unchanged |
| Q1.9 repeated seeds, held-out IDs, censoring/attempts | c | Three run blocks; fixed calibration/validation init split; reset hashes/RNG, every attempt/error prefix retained; accepted-attempt rule explicit |
| Q1.9 genuinely new scene distribution | e | Stock 50-init pool only; repeats do not create new scenes |
| Q1.10 held-out robot/benchmark | e | Requires another interface/store/model/client/domain dataset; no such launch authorized here |
| Q1.11 observer validity, seed separation, stage/cost accounting | b | Measured v2 dispatch counts, CPU RNG/byte parity, profiling-only blind/head costs; real dynamics equivalence is still unverified |
| Q1.11 100/300 matched simulator equivalence trials | c | enabled/off and p0 switches support coordinator validation; no simulator trials started here; existing smoke lacks an unprofiled paired baseline |
| Q1.12 local successful-action set | d | Sixteen neighbour chunks + cache/policy/wire alternatives preserved; action-set success requires restored branches, not distances |
| Q1.13 benchmark weighting | b | Equal-task population and init pool declared in provenance, task/init IDs and counts retained |
| Q1.13 external deployment frequencies | e | Need caller traffic/weights; balanced LIBERO cannot infer them |
| Q2.1 assigned episode dose/budget response | c | Fixed 0,1/8,1/4,1/2,1 cohorts plus `episode_doses` randomized dose_mix and its actual assignment probability; cap is a separately named design |
| Q2.2 effective probabilities, RNG/source/precedence/identities | b | Expanded `assignment`, `episode_assignment`, chunk hashes, catalog, source and key references; private seeds independent of injection |
| Q2.3 extra-call/remaining-budget paired outcomes and downstream cost | d | Same-state remaining-budget suffixes require resume plus additional simulation/policy work; inputs saved, outcomes not fabricated |
| Q2.4 task calibration, seed variance, untouched validation | c | Task-balanced paired init blocks, three seeds, initial/reset identity, episode ICC reader and held-out init split. Pilot does not provide tight task-specific guarantees. |
| Q2.5 50/500 quality×budget outcomes | c | Eight endpoint cells × dose cohorts; both scales included for each model/suite |
| Q2.5 independent and intermediate banks | e | New frozen banks/cells need construction and actual rollouts; not hidden in a current-bank logging upgrade |
| Q2.6 controls, ledger denominators and counters | b | Active controls separate from settling, source per request, actual stage counters, actual-control deployment repricing and remaining-cost joins |
| Q2.6 profiler-disabled serving latency | e | All-shadow timing cannot measure the skipped-forward fast path; separate coordinator timing validation required |
| Q2.7 caps/exhaustion/censoring | b | Credits_before/after, cap_exhausted, pending due/holds, original remaining deadline, stock termination reason, error prefixes; cap falls back to cache, never aborts rollout |
| Q2.7 recoverability after a failure | d | Needs larger-budget continuation at same state/deadline; no guard is labeled irreversible |
| Q2.8 policy sampling noise | b | Same primary/resample bundle as Q1.7, not another collection campaign |
| Q2.9 benchmark mixture and cost tails | b | Declared equal-task weights; raw episode/control/call distributions, task/init identities, no silent averaging over missing tasks |
| Q2.9 real arrival distribution | e | External population observations absent |
| Q2.10 held-out domain budget curves | e | Different domain/robot and libraries must be supplied and run |
| Q2.11 acquisition/rejection/refit lifecycle cost | e | Historical unobserved acquisition expense cannot be recovered from serving; new-bank ledgers require acquisition scope. Supplied provenance can reference them. |
| Q3.1 guard and non-guard actual treatment support | c | Pre-guard factorial cohort, independent coins, saved baseline verdict, available treatments and effective probabilities |
| Q3.2 branchable exact same state and paired outcome-to-go | d | Snapshot does not certify replay; all restore dependencies and blockers below |
| Q3.3 call now/next/further with equal allowance | c | Window cohort: cap=1, delays=[0,1,2], common original deadline, A after allowance is spent; trigger/delay propensities and unfulfilled pending calls retained |
| Q3.3 paired recovery windows from the identical prefix | d | Branches needed for individual matched suffixes and counterfactual recovery capability |
| Q3.4 policy durations and handback choices | c | Independently sampled policy duration 5/10 and hold 1/2/3 fresh-policy anchors; five-control duration forces a real next vision; no unsupported horizon extension |
| Q3.4 full source×duration factorial including CACHE5 | d | Current cache commitment is kept at 10 to preserve p0=A; CACHE5 comparison can be a later branch. Policy5 versus Policy10 is randomized, but do not call it a complete source×duration factorial. |
| Q3.5 physical events/progress channels | b | Raw physical/predicate/control stream at every tick; explicit privileged flag and stable exposed entity IDs |
| Q3.5 intended release versus slip, validated event taxonomy | e | Requires semantic adapter/audit or annotation; raw contacts/poses cannot certify intent |
| Q3.6 applied stream, lineage and observation chronology | b | Every control index, source, parent anchor, chunk offset, wire-to-application check, observation hash, timestamps and partial terminal count |
| Q3.7 deployment versus profiling cost-to-go | b | Deployment IR and physical forwards/extra heads separated; reader joins future MISSes and controls. Sparse-path latency remains unmeasured. |
| Q3.8 sparse histories, cooldowns, same-allowance placement | c | Sparse 1/8 cohort, A reference, configurable prefix/cooldown/cap/hold, window cohort; all prior assignment histories preserved |
| Q3.9 synchronized proposals, top16, metric/state/phase | a | v1 chunks, rows/distances/weights/dispersion, metric code, LOEO quantile, guard inputs/output, history input archive |
| Q3.9 modality/calibration/candidate-action provenance | b | Per-camera PCA/cosines, all neighbour chunks, catalog/manifests and empirical guard p-values; source fits are exact bytes |
| Q3.10 stochastic policy uncertainty and clean state contract | b | Resample seeds, isolated stages/keys, stage counts and CPU RNG preservation; real-model v2 smoke still required |
| Q3.11 intervention response across existing eight cells | c | Same schema/design in both models × both suites × both libraries |
| Q3.11 causal bank-content/new-domain transport | e | Existing banks confound size/content/source; LIBERO does not supply another robot's outcomes |
| Q3.12 reserved validation and run reliability | c | Prespecified init split and three blocks, complete attempt/error audit; held-out random-policy data support only estimands with adequate support |
| Q3.12 direct fresh SR of a not-yet-selected placement rule | e | Rule has not been defined/frozen; cannot relabel mixed data as its deployment outcomes. Later normal rollouts or exact branches required. |
| G.P1 policy chunk at every decision | b | `p3_decision` + NPZ primary policy chunk/seed on anchors and blind steps; blind policy is never served |
| G.P2 blind raw keys | b | Two lossless float32 raw keys per blind decision, not just PCA codes; schema is 256 KiB for both current 32768-D cameras, not 131 KiB |
| G.P3 K4 at 1/16 anchors | b | `resampling` independent selection, four seeds/chunks, three additional stage3 calls, no repeat stage2 |
| G.P4 full diagnostics/p-values/fired tests/V7 | b | v1 internals retained; `calibration.statistics/p_values`, `guards.fired_tests`, source-fit V7 arrays; blind clone diagnostics also retained |
| G.P5 seeded per-anchor propensity/injection | a | v1 assignment and same-shadow reuse; v2 expands designs without adding an injection forward |
| G.P6 control-resolution trajectory/fingers/contacts/objects | b | Client EnvTap, available physical/observation fields, missingness explicit |
| G.P7 snapshot/RNG/identity | b | Snapshot v2 pre-inference physics/controller numerics/RNG/identity and selection probability; exact restoration belongs to d, below |
| G.P8 goal predicates | b | `goal_state` or `parsed_problem.goal_state` + `_eval_predicate` where exposed; availability and exceptions logged, no invented subgoals |
| G.P9 per-stage timing | b | s1/s2/s3, key build, verdict, base/shadow-guard query, blind diagnostic, I/O and client env/infer times; scopes documented rather than summing overlapping times |
| G.P10 episode/outcome/cap/hierarchical identity/version/schedule | b | Journal + client end + catalog/assignment joins; layout/style/pin remain explicitly not applicable to current LIBERO init pool |
| G.P11 calibration tables/interface/effective alert level | b | `calibrate` per-task/pooled episode maxima, empirical successful-library alert share, residual score; source V7 tables and own-library distance LOEO; manifest-derived interface |
| G.P11 universal Q1 success score or new library policy noise floor | e | No accepted universal score; prefit is CPU and cannot invent fresh policy samples on library states. Deployment-state K4 draws are supplied; existing floor data may be referenced offline. |
| G.P12 transfer benchmark / third camera / new scene hierarchy | e | Current plugin/store contract is two-camera LIBERO; RoboCasa/robot integration and actual transfer data are outside this authorized implementation/campaign |

The reports explicitly say that all-row LOEO, alternate-fit rescoring and strict
PCA/metric refit-LOEO can be computed offline from existing complete library
data. They are computation gaps, not reasons to recollect identical observations.
The new empirical tables do not substitute frozen-fit LOEO for inductive refits.

## Exact restore assessment

No `src/` change is intrinsically required: the stock runner already accepts a
client factory and run_episode function, and a connection-local wrapper can
implement a future resume message. The stock API does **not** currently expose
an exact plugin-session resume endpoint. This delivery captures data, not a
certified branch executor. `restore_certified=false` is intentional.

Restore needs the same task/scene/model/XML, LIBERO/robosuite/MuJoCo versions,
initial-state hash; complete dynamic time/qpos/qvel/act/ctrl/mocap/solver state;
environment task/termination counters and original remaining deadline; controller
goal/interpolator/filter state including object-valued state; Python/NumPy/env
RNGs; any render/observation state; and server fit, gripper memo, anchor/guard
history, observation/action/hit/vision buffers, pending tail cursor and original
wire chunk, design credits/hold/due state and exact policy noise context.
Numeric controller fields and exposed solver arrays are saved, but skipped
objects are explicitly listed. They are a concrete blocker to claiming a
complete checkpoint. Client state is pre-inference; logged method state is
post-query/pre-action. A restorer must reconstruct the proper side of that
boundary without applying guard updates twice.

Server input tapes and immutable fits permit prefix reconstruction without
replaying simulator controls; use actual has_vision flags, not the observational
blind keys as live history. Both alternative wire chunks now avoid ambiguity
about output normalization. This has not been validated on LIBERO. First restore
and replay the factual controls, compare control-boundary simulator/observation
state, termination and next guard verdict, and report tolerances/rejections.
Only after that test may cache/policy/delay/budget/alternative-action continuations
be labeled paired counterfactuals. Proposed feasibility sample: 100 distinct
parent episodes spread across cells, then 300 if zero failures (upper 95% failure
bound about 2.95%, then <1%). Branch rollouts and their GPU/simulator cost are
not included in the campaign estimate. A visually similar state is not a restore.

## One pilot and continuation campaign

Eight cells: π0.5 and GR00T × l10 and spatial × nominal 50 and 500 libraries.
Within each cell, nine cohorts share the **same superset measurements**:

| Cohort | Baseline and intervention |
|---|---|
| A | A, p=0 |
| dose125 / dose25 / dose50 | A, fixed anchor p=1/8,1/4,1/2 |
| P10 | A proposals logged, p=1, ten-control policy commitment |
| B | Exact deployed B guards/commit behavior, no added injections |
| factorial | B proposal/guard diagnostic, randomize before guard at distance-quantile bins p=.125/.25/.5; policy duration 5/10, hold 1/2/3, cooldown=1 |
| window | A, p=.125 trigger after two cache anchors, cap=1, delay uniformly 0/1/2 anchors, ten-control policy chunk, then A |
| dose_mix | A; ex-ante uniform episode rate from 0,1/8,1/4,1/2,1; source coins then follow that rate |

Every cohort computes policy on all decisions, K4 at 1/16 anchors, raw keys,
retrieval/guard/calibration, control telemetry and snapshots (`snapshot_p=1`,
every=1). No field-specific reruns. The pilot can estimate whether every-anchor
snapshots/physics tracing fit the storage/latency budget; any later sampling
change must be recorded as a design version, not silently spliced.

| Per cell | Pilot | Full cumulative | Disjoint continuation |
|---|---:|---:|---:|
| Episodes per cohort | 60 | 500 | 440 |
| Episodes across nine cohorts | 540 | 4,500 | 3,960 |
| All eight cells | **4,320** | **36,000** | **31,680** |

Pilot: every cohort uses ten tasks × inits 0,1 × three seed blocks (603–605).
Full: block0 inits0..24, block1 inits0..14, block2 inits0..9, all ten tasks.
Thus 250 unique init clusters per cohort/cell, with 100 clusters observed three
times, 50 twice and 100 once. Continuation manifests contain only inits2..end;
pilot tasks are retained, not rerun. All cohorts share the same pairing blocks.
The stock pool remains 50 inits/task; this is repeated-seed validation, not a
new initial-scene distribution. Init%5==0 is calibration; all other init groups
are reserved validation, across every seed block.

216 emit_arms specs = 8 cells × 9 cohorts × 3 seed blocks. A spec has 20 pilot
episodes and 100/150/250 cumulative episodes depending on block. The separate
schedule supplies the exact manifests and `P3_ENV_SEED`; use it rather than
assuming emit_arms supports a client seed override. Model policy seeds are
independently keyed by Profile replicate. With a shared GPU admit one server
line first, at most two only when the coordinator's memory budget permits.
Two lines on the same GPU do not imply twice the GPU throughput. Startup/model
reload and client rendering overhead must be measured in the pilot.

Power: for a paired binary difference with assumed variance/discordance .20,
95% half-width is `1.96 sqrt(.20/n_eff)` and 80%-power MDE is
`2.801621 sqrt(.20/n_eff)`. Repeated-init design effect is
`1 + rho * sum(r_i*(r_i-1))/sum(r_i)` = `1+2rho` for pilot, `1+1.4rho` full.
This is a planning sensitivity, not a measured ICC. At rho=.3, pilot n_eff=37.5
per cohort/cell: ±14.3 pp half-width and 20.5 pp MDE. Full n_eff=352.1:
±4.67 pp half-width and 6.68 pp MDE. At rho=1 full half-width is ±6.07 pp.
A ±5 pp paired target needs 308 effective units; detecting 5 pp with 80% power
needs 628. Pilot does not settle SR effects; full does not promise 1–2 pp safety
or precise per-task effects. Randomized local strata have fewer supported units
and unequal propensity weights; calculate cluster-effective precision from pilot
scores, not from copied terminal outcomes on thousands of anchors.

Use `pilot_stats` on episode Y, residual summaries and cost summaries; task fixed
effects use K−T between-init degrees of freedom and unequal-repeat coefficients.
Analyze paired treatment differences separately; their ICC
need not equal Y's ICC. Freeze primary strata/contrasts before validation. Use
pilot estimates to decide whether the planned continuation reaches the required
precision; any enlargement appends new init/seed slots to these arms.

Cost assumptions: 60 five-control decisions/episode; full shadow at **every**
decision; conservatively up to 60 anchors for K4 costing; each extra stage3 is
priced as an entire full-policy call. Historical eager full times are π0.5
463.394 ms and GR00T 250.131 ms. This deliberately overprices head-only resamples,
but does not measure this shared-GPU server's wall time.

| Campaign phase | GPU compute-equivalent hours | 2× occupancy budget | Raw two-camera keys | Collection storage budget |
|---|---:|---:|---:|---:|
| Pilot | 30.50 | 61.01 | 63.28 GiB | 421.88 GiB |
| Full including pilot | 254.19 | 508.39 | 527.34 GiB | 3,515.63 GiB |
| Continuation only | 223.69 | 447.38 | 464.06 GiB | 3,093.75 GiB |

Storage budget assumes **100 MiB/episode**, not a measurement of real telemetry;
fits, expanded CSVs, duplicated legacy input archives and archival replicas need
additional space. `results/campaign_v2.json` contains the arithmetic and ICC
sensitivity. GPU-hours exclude future branches and library acquisition. Model
startup and rendering can exceed a 2× allowance; measure before committing full
capacity. No lane-throughput claim is inferred from the v1 smoke.

Per-cell planning costs below apply separately to l10/50, l10/500, spatial/50
and spatial/500 within the named model. Equal duration is a simplifying planning
assumption; actual task horizons and intervention success will change these costs.

| Each cell | Pilot episodes | Pilot GPU h compute / budget | Pilot storage | Full episodes | Full GPU h compute / budget | Full storage |
|---|---:|---:|---:|---:|---:|---:|
| π0.5, each of four cells | 540 | 4.95 / 9.91 | 52.73 GiB | 4,500 | 41.27 / 82.54 | 439.45 GiB |
| GR00T, each of four cells | 540 | 2.67 / 5.35 | 52.73 GiB | 4,500 | 22.28 / 44.55 | 439.45 GiB |

The pilot yields the listed observable fields and exercises the c design
dimensions. It does not meet every ideation agent's requested precision target,
independent-bank count or 100/300 matched simulator-equivalence trials. Those
counts are not relabeled as logging fields; the validation limits remain above.

## Coordinator preparation and smoke recipe (not launched)

Define the following shell array from the repository root for CPU preparation:

```bash
P3PY=(taskset -c 2-5,46-49 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.campaign_v2
"${P3PY[@]}" -m exp.offline_search.closed_loop.ops.emit_arms --run-root <RUN> --spec exp/offline_search/rounds/r06/p3_profiling/arms_v2.json
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.calibrate --model pi05 --cell l10_50 --out <RUN>/calibration/pi05_l10_50.json
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.prefit --run-root <RUN> --spec exp/offline_search/rounds/r06/p3_profiling/arms_v2.json
```

Run calibrate once for each of the eight cells before prefit; it never starts a
model or simulator. Alternatively copy the eight supplied `calibration_v2/*.json`
to `<RUN>/calibration/`; prefit checks the referenced guard-fit hash. Do not
recompute into an existing output path. Supply model/checkpoint provenance before freezing fits.
Copy manifests_v2 to `<RUN>/manifests`. Select the **run-specific** v2 client
launcher, preserve stock run_arm separately as the wrapper input, and distribute
the owned package via the coordinator's ordinary deployment process. Apply the
schedule's manifest and P3_ENV_SEED. The client telemetry tree is separate from
ordinary collect.py output; archive it explicitly, along with `p3_inputs/`,
legacy `inputs/`, server decision logs, journal, fits and calibration artifacts.

Before the pilot, use tasks0–1/inits0–1 on the two real-smoke cells with A, P10,
factorial and window v2 specs, enabling the v2 client. These are validation
episodes; do not silently count repeated accepted IDs twice. Check one
p3_decision per dec, one p3_anchor per vision, actual stage counts, primary draw
reuse, blind-key isolation, both duration paths, cap/hold/delay precedence,
client control offsets and partial terminal counts, snapshot RNG/identity, and
explicit predicate/actuator availability. Any unavailable required physical
adapter field is a smoke failure to resolve before the main pilot, not zero.
Then run:

```bash
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.read_v2 --run-root <RUN> --arms <ARM_NAMES> --client-root <COLLECTED_CLIENT_TREE> --out <NEW_TABLE_DIRECTORY>
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.pilot_stats --episodes <NEW_TABLE_DIRECTORY>/episodes.csv --value Y --out <RUN>/pilot_icc.json
```

Full continuation uses the same arm IDs, frozen fits/designs/seeds and only the
continuation manifests. Coordinator bookkeeping must schedule those new slots
despite the pilot arm DONE markers; do not remove accepted pilot records or
overwrite old log trees. No automated chain manipulation is supplied or run.

## Real-model v1 smoke: completed and audited read-only

All four arm DONE markers and CHAIN.DONE were observed under
`/home/weiland/trace_runs/os_closed_loop/r06_p3_smoke/state`.

| Arm | Episodes | Decisions | Vision / p3_anchor | MISS | Policy tails |
|---|---:|---:|---:|---:|---:|
| π0.5 l10/50 p0 | 4 | 312 | 156 | 0 | 0 |
| π0.5 l10/50 p1 | 4 | 211 | 107 | 107 | 104 |
| GR00T l10/500 p0 | 4 | 308 | 155 | 0 | 0 |
| GR00T l10/500 p1 | 4 | 209 | 106 | 106 | 103 |

Strict accepted-attempt joins and input archives pass: exactly one anchor per
vision; p0 has zero MISS; p1 selected chunks are byte-equal to the saved shadow;
every nonterminal anchor has its exact policy tail. The six p1 anchors without
a following tail are terminal decisions, not missing requests. Total 16 episodes,
1,040 decisions, 524 anchors and 213 MISSes. No SR comparison is inferred from
four episodes per arm.

**No-second-forward limitation:** v1 logs declare full_policy_forwards=1 and
additional_forward_for_injection=0, but contain no independent stage invocation
counters. Those logs alone cannot prove the real model did not call stage2/3
twice. CPU staged-path tests verify reuse; v2 now measures the dispatch counts.
The real smoke used the standard client without snapshots: it validates neither
v2 blind shadows/control telemetry nor exact restoration. Evidence:
`results/real_smoke_v1.json`; strict tables `/tmp/p3_v2_real_smoke_all`.

## CPU checks and installation evidence

Fixed-observation replays and synthetic client tests are not closed-loop SR or
simulator-parity experiments. The remaining real v2 validation belongs to the
coordinator under the hard no-server/no-worker constraints.

| Check | Observed result | Evidence |
|---|---|---|
| V2 matrix, both models × both suites × both libraries × 13 variants | 104 jobs; 9,152 decisions; 5,044 vision decisions; 1,640 MISSes. 32 byte/cadence/MISS comparisons: off-A=A, p0-A=A, off-B=B, enabled B p0=B with recorded policy fixtures. PASS | `/tmp/p3_v2_matrix_published/summary.json`, `results/matrix_v2.json` |
| Enabled v2 collection paths in that matrix | 6,336 decisions = 3,620 anchors + 2,716 blind probes; 3,180 extra sampled heads; 9,516 total policy samples. Every enabled input archive audited. PASS | Same matrix, each variant's audit |
| V1 regression repeated | 64 jobs; 5,632 decisions; 2,848 vision; 703 MISSes; 1,424 anchors; 24 exact off/p0 comparisons. PASS | `/tmp/p3_v1_regression_v2/summary.json`, `results/matrix.json` |
| V1 strict input audit | All 32 enabled configurations, 1,424 anchors checked with require_inputs. PASS | `results/input_audit_v2.json` |
| V2 staged CPU paths | π0.5 direct/coordinator and GR00T, p0/p1: six cases, anchor counts s1/s2/s3=1/1/4 at K4; isolated blind counts=1/1/1. Primary draw preserved; global Torch RNG and live key builder unchanged; logging failure closes the attempt. PASS | `/tmp/p3_v2_units_icc_final`, `results/units_v2.json` |
| V1 after importing v2 | Six staged paths still s1/s2/s3=1/1/1; RNG unchanged; wrong chunk rejected before broadcast. PASS | Same units report |
| Randomized design checks | 12,000 units: immediate delayed-CALL rate .103083 at target .1; duration5 fraction .49125 at target .5; cap≤1, pre-guard and override checks. PASS | Same units report |
| Synthetic client | Seven controls, 21 physics substeps, two decisions, one snapshot, partial terminal count two; issued controls unchanged; altered control rejected. PASS | Same units report; no simulator run |
| V2 reader | Four episodes, 96 decisions, 48 anchors, 468 synthetic controls, 768 neighbours, 480 action-step rows; eight corruptions rejected. PASS | `/tmp/p3_v2_reader_delivery/tables`, `results/reader_v2.json` |
| V1 reader | Six corruptions rejected; 48 anchors, 768 neighbours, 480 action rows, four episodes. PASS | `/tmp/p3_v1_reader_v2`, `results/reader.json` |
| V1 units / K5-derived estimator | Six staged paths; LOEO early/main exclusion; 1,970/10,000 p=.2 assignments; null estimate −.01475 (95% CI −.04838,.01890), planted .2 estimate .21625 (.18223,.24629). PASS on synthetic data only | `/tmp/p3_v1_units_v2`, `results/units.json` |
| V2 prefit and artifact reload | π0.5 l10/50 and GR00T l10/500: each two recorded episodes, 24 decisions, 12 anchors, zero MISS, 30 samples including six extra heads; all six calibrated statistics and four modality fields present. PASS | `/tmp/p3_v2_prefit`, `results/artifact_reload_v2.json` |
| Old v1 artifact after v2 import | 48 decisions, 24 anchors, five MISS; served SHA256 `a869f36381e0ac174eca6da7ce9b26d8a683ecc510ab3171ba2fd545f2d9a781` and vision digest match its old report. PASS | `results/v1_artifact_after_v2.json` |
| Eight calibration exports | 2,005 successful library-episode records / 36,688 pseudo-anchors. All six empirical-reference tables emitted. PASS as a CPU diagnostic, not deployment calibration | `calibration_v2/`, `results/calibration_v2.json` |
| Campaign / schema | 216 specs, 432 emitted YAML files; pilot/full/continuation manifests disjoint as specified; all 48 ranked requests represented by 67 disposition rows. 30 Python files parse; launcher `bash -n` passes | `/tmp/p3_v2_emit_valid`, `/tmp/p3_v2_campaign_published`, `results/emit_validation_v2.json` |
| ICC plumbing | Perfect replicated toy ICC=1; task-fixed balanced/unequal ANOVA examples give 7/9 and 11/71; insufficient within-task replication returns null with reason | `results/units_v2.json`, `/tmp/p3_v2_reader_delivery/icc_residual.json` |

The v2 reader corruption cases are missing decision, wrong actual propensity,
wrong nominal propensity, wrong selected action, wrong tail, conflicting outcome,
duplicate client decision and multiple accepted attempts. Development checks
also exposed and fixed B's suppressed-guard early-look behavior, GR00T's null
camera-name manifest, unsupported client seed overrides, top-level v2 marker
precedence and stale success state after a logging failure. The final frozen
check caught a parenthesis error in startup provenance hashing; it was fixed,
all 30 files were parsed, units rerun, and the complete 104-job matrix rerun.
Failed intermediate roots were retained; none is presented as passing evidence.

These are the executed CPU validation entry points (all use P3PY defined above;
use fresh output roots to reproduce because the tests reject existing roots):

```bash
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.test_matrix --out /tmp/p3_v1_regression_v2 --workers 2
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.test_units --out /tmp/p3_v1_units_v2
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.test_reader --matrix /tmp/p3_v1_regression_v2 --out /tmp/p3_v1_reader_v2
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.test_v2_matrix --out /tmp/p3_v2_matrix_published
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.test_v2_units --out /tmp/p3_v2_units_icc_final
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.test_v2_reader --matrix /tmp/p3_v2_matrix_published --out /tmp/p3_v2_reader_delivery
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.audit_smoke --run-root /home/weiland/trace_runs/os_closed_loop/r06_p3_smoke --out exp/offline_search/rounds/r06/p3_profiling/results/real_smoke_v1.json
```

The intermediate `/tmp/p3_v2_matrix_frozen` root is not a passing matrix because
the later syntax error interrupted its GR00T cases. The final complete matrix
and reader source is `published`. Each matrix run also saves exact child commands
in `runs/*/command.json`.
No Q5/K1/K2 shared-file regression is claimed here: their conditional install
requirement did not apply because no shared file changed. The retained shared
plugin hash and both v1 serving-module hashes are checked again at publication.
