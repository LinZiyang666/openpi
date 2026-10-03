# R8 coding brief — closed-loop debug mode, deferred shadows, profile tools, R8 arms

Written 2026-09-30 11:5x CDT by the coordinator. Six GPT-6.1 sol coders work in parallel (S1–S6), each on disjoint
files. Read in this order: this file, `exp/offline_search/debug/SCHEMA.md` (the contract), `rounds/r08/SELECTION.md`
(decisions, arms), then the explorer proposals relevant to you (`rounds/r08/ideation/E*/PROPOSAL.md`; E5 = debug
architecture, E3 = failure forensics, E4 = look less / look half, E1 = segmentation, E2 = call value).

## 0. Hard rules (all coders)

- **Behaviour invariance.** With debug off every code path and every existing log byte is unchanged. With debug on
  the observer only copies; no policy/encoder/method/RNG call on the live path, no mutation of method state, a
  capture failure never changes an action (it becomes a telemetry status). Backpressure (blocking) is allowed;
  silent drops are not.
- **Generality.** Nothing keyed on LIBERO task names, suite names or hand-set LIBERO-unit thresholds in the
  collection path. Model/environment specifics go through small adapters (model manifest: cameras, state/action
  dims, valid action dims, H; environment adapter: entities, contacts, predicates).
- **Lossless, compact:** schema §0. No pickle/object arrays in data files. Python 3.8 compatible for anything that
  runs in the LIBERO client (timan107 worker venv is Python 3.8; see `rounds/r06/p3_profiling/check_python38.py`).
- **Commands:** from `/home/weiland/projects/openpi`, `taskset -c <your CPUs> env OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
  .venv/bin/python ...`; ≤ 3 Python processes at once; never CPUs 38–43 / 82–87. No GPU in your sandbox — write
  GPU code with a CPU fake for tests and list the exact GPU validation commands for the coordinator.
- **Do not** start servers, workers, chains, simulators on remote hosts, tmux sessions or ports 23100–23199; do not
  touch other processes; no git state changes; no `rm -rf`; no `pkill -f`; do not read `tests/review_tests/`.
  `/home/weiland/trace_runs` is read-only for you; write scratch/fits under `/tmp/r8_<you>/`.
- Existing files outside your ownership are read-only unless listed under your "may edit". If you need a change in
  someone else's file, describe it precisely in your HANDBACK (the coordinator integrates).
- Deliver `HANDBACK_<you>.md` in `exp/offline_search/debug/handbacks/` (S5: `rounds/r08/methods/HANDBACK.md`):
  what you built, file list, how to run, tests run and their output summary, GPU/remote validation commands for the
  coordinator, known gaps. Keep it factual. Finish with a ≤ 15-line summary as your final message.

## 1. Shared interfaces (fixed now)

**Package** `exp/offline_search/debug/` (system tool, not round-local). `schema.py` (S1 writes it **first**, within
your first 15 minutes, so others can import): schema version `"osdebug.v1"`, status enum, identity helpers
(`episode_key(task_uid, attempt)`, `decision_id(...)`, sampling predicates `rawkeys_sampled`, `snapshot_sampled`,
`draws_sampled` using `sha256(f"{campaign}|{task_uid}|{decision_seq}")` as in SCHEMA §3–§5), file-name patterns,
block sizes (16 decisions / 64 controls), and `write_npz_block(path, arrays)` / `read_npz_block(path)` helpers
(atomic publish: write `.part`, fsync, `os.replace`). Until it exists, code against SCHEMA.md names.

**Reader API** (S3 implements; S4/S6 consume; exact names):
```python
from exp.offline_search.debug import reader
arm = reader.open_arm(run_root, arm_name)          # -> ArmData
arm.manifest; arm.server_meta                       # dicts
arm.journal()                                       # DataFrame: task_uid, task_id, init, attempt, accepted, success
arm.episodes(accepted_only=True)                    # DataFrame: episode.json fields + journal outcome
arm.decisions(columns=None, accepted_only=True)     # DataFrame: server jsonl joined with client events on decision_id
arm.decision_arrays(keys, decision_ids)             # dict[str, np.ndarray] aligned to decision_ids (from blocks)
arm.controls(episode_key, keys=None)                # dict[str, np.ndarray] for one attempt, concatenated blocks
arm.iter_controls(keys=None, accepted_only=True)    # yields (episode_key, dict)
arm.images(decision_ids, cameras=None)              # dict[camera, uint8 (n,H,W,3)]
arm.snapshots(episode_key)                          # dict[name, dict]
arm.aug(kind, decision_ids=None)                    # dict of arrays keyed/aligned by decision_id
arm.catalog()                                       # library row table (DataFrame) or None
reader.pair(arm_a, arm_b)                           # accepted episodes joined on (task_id, init)
reader.p3v2_adapter(run_root, arm_name)             # ArmData-like view over R6 P3 v2 pilot data (for dev/tests)
```
S3 also ships `debug/fixtures.py: make_synthetic_arm(dir, n_episodes=6, model="pi05", method="A", seed=0)` that
writes a small schema-valid arm (server + client + aug) — S4/S6 test against it.

**Method diagnostics protocol** (S1 defines, S5 implements for new methods): a method may define
`debug_record(self) -> dict` returning JSON-safe scalars/short lists for the decision just served (called by the
observer after the response is fixed; must be read-only). S1 writes read-only adapters for existing classes that do
not define it (BlindAWM/A, R6/R7 CallController CU/CT, R7 StageFollow SF, R7 StageWrist SW, SeededInference).

**Privileged oracle channel** (only for the diagnostic oracle arms): the client puts `obs["__oracle__"] = {...}`
(S2 computes it); the plugin pops it and passes it to the method **only** when started with `--os-oracle`; otherwise
it is dropped and logged as an error. Decision records carry `oracle` fields so these arms are never mistaken for
deployable methods.

## 2. Assignments

### S1 — server observer (CPUs 14-17,58-61)
Own: `exp/offline_search/debug/{__init__.py,schema.py,server/}`, `exp/offline_search/debug/tests/test_server*.py`.
May edit (minimal hooks, byte-identical when flags absent): `exp/offline_search/closed_loop/{plugin.py,
serve_pi05.py,serve_groot.py,blind.py,stage_overrides.py}`. Note these already carry uncommitted R7 edits
(`--os-request-cameras`); keep them.
Build:
1. Flags `--os-debug-dir <dir>` (enables) and `--os-debug-config <json>` (campaign, sampling rates, writer queue
   bytes). `__debug__` pop always; echo when enabled (SCHEMA §1). `__oracle__` handling per §1.
2. Capture per request (SCHEMA §3): exact wire images/state/prompt **before** transforms (copies), all serving
   fields, retrieval, randomization, dispatch counters, timings, method diagnostics (protocol above + adapters),
   compact keys, cache/policy/served chunks, served wire chunk; raw keys on the hashed 1/16 sample.
   Find where each value lives (`_ConnPolicy.infer`, `_BlindAdapter`, `after_infer`, `_write_inputs`, method
   results/extras, R6 `method_c`, R7 `c1_follow`/`c2_wrist`/`c3_calls`) and capture it **before** `after_infer`
   clears per-decision state. The existing 24/40-scalar extras cap does not apply to debug records.
3. Writer: background thread, bounded queue by bytes (default 512 MiB, blocking put), ≤16 decisions per block,
   atomic publish, jsonl flushed with its block, `writer_stats.json`, clean drain on SIGTERM/exit, `meta.json` at
   startup (model manifest, method spec, library/fit/stage-table sha, git head + dirty-diff sha).
4. Tests: (a) request-tape replay parity: record ≥ 200 requests per method family through the plugin with a fake
   policy/encoder (CPU), replay with debug off vs on → byte-identical responses, identical method state digests and
   RNG consumption; families: A (BlindAWM), CU and CT (R7 CallController), SF (R7 StageFollow), SW (R7 StageWrist,
   per-request cameras), pure policy; (b) schema/round-trip of blocks; (c) writer backpressure and drain; (d)
   existing closed-loop tests still pass (`exp/offline_search/closed_loop/selftest.py` and any tests under
   `exp/offline_search/**/test_*.py` that touch the plugin — list what you ran).
5. HANDBACK: exact server command-line additions, GPU parity commands for the coordinator (real π0.5/GR00T
   servers, replay a recorded tape debug off/on), measured per-decision overhead on CPU.

### S2 — client capture, transport, ops, library backfill, oracle flag (CPUs 18-21,62-65)
Own: `exp/offline_search/debug/{client/,transport/,ops/,backfill/}`, `exp/offline_search/debug/tests/test_client*.py`,
`test_transport*.py`. Read-only reuse (import or copy with attribution): `rounds/r06/p3_profiling/{telemetry.py,
snapshots.py,worker_v2.py,run_gtp_v2.py,stream_*.py,dispatch_fence.py,client_*,deploy_client.sh,build_client_bundle.py}`,
`examples/libero/{episode_runner.py,worker_entry.py,main.py}`, `closed_loop/ops/{chain.sh,remote/run_arm.sh,collect.py}`.
Build:
1. Client entry `debug.client.worker` (Python 3.8): wraps the stock runner by injection like P3 `worker_v2.py`
   (no P3 marker dependency): request envelope + echo stripping + `events.jsonl`; per-control capture per SCHEMA §4
   with **correct geom/body naming via the MuJoCo model id→name maps** (E3 found P3's contact geom ids misaligned
   with its name list — add a self-check: at reset every resting object must contact the table), contacts with
   `mj_contactForce`, predicates from the task's goal state (read-only evaluation), every movable object/articulation
   body, actuator substeps when exposed; entity catalog once; snapshots per SCHEMA §4.4; keep the stock env seed
   (`--seed`, default 7, identical for every arm — log it). Capture must not perturb the simulation: prove with a
   fake-env unit test and give the coordinator a real-env check (same actions, capture on vs off → identical
   `qpos/qvel` sequences, ≥ 3 episodes, both suites).
2. Transport: stream client files to a receiver on weilandserver reusing P3's offset-ACK/receipt/spill protocol with
   a new allowlist (`episode.json`, `events.jsonl`, `controls_\d{4}.npz`, `snap_(reset|final|\d{6}).npz`); receiver
   writes `<RUN>/runs/<arm>/debug/client/<episode_key>/` + receipts; dispatch fence persisted in both stream and file
   modes; `collect/verify` command that reconciles receipts with the journal. timan107 `/scratch` has only ~35 GB
   free, so streaming is the default and spills must be small and replayable.
3. Ops: `debug/ops/chain_debug.sh` derived from `closed_loop/ops/chain.sh` (same arguments/markers) that (a) ensures
   the receiver for the run, (b) adds `--os-debug-dir <arm>/debug/server_<port> --os-debug-config <json>` to the
   server args, (c) launches the remote client through a new `debug/ops/remote/run_arm_debug.sh` (derived from
   `closed_loop/ops/remote/run_arm.sh`) with the debug entry and stream env, (d) writes the DONE marker only after
   capture verification; plus `debug/ops/build_client_bundle.sh`/`deploy_client.sh` for the timan107 tree
   `/scratch/zixuans8/openpi_trace` (the coordinator runs deployment).
4. Oracle flag (diagnostic arms only; enabled by an env/config switch): per decision compute from simulator truth
   whether the end effector is inside the grasp window of an un-lifted, unsatisfied goal object (E3 §4 5a/5b and
   its `forensics.py`/`grasp_drop_pairs.py` definitions: distance to object, lift state, goal predicate status),
   plus the distance; send as `obs["__oracle__"]`. Goal objects come from the task's goal predicates (no task names).
5. Library backfill (`debug/backfill/`): replay library demonstrations in the simulator from their initial state and
   executed controls through the same capture, verify replayed robot state vs the library `rs` and success, write
   per-episode client-format records under a given output root. First determine feasibility: where the library's
   initial states and executed control sequences come from (store arrays, `dual_20260923` traces, B-pool records);
   report exactly which libraries can be replayed. Deliver the CLI; the coordinator runs it on timan107.
6. Tests: fake-env capture tests, geom naming, contiguity, envelope, transport with local sockets (ephemeral ports
   on 127.0.0.1 only), spill/replay, verification.

### S3 — reader, validation, capacity, deferred augmentation (CPUs 22-25,66-69)
Own: `exp/offline_search/debug/{reader.py,validate.py,capacity.py,fixtures.py,aug/}`, tests `test_reader*`,
`test_validate*`, `test_aug*`.
Build:
1. `reader` exactly as §1 (pandas/pyarrow available locally; derived parquet caches under `debug/derived/`), plus
   `p3v2_adapter` over `/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot` (tables + client_telemetry) so tools can
   be developed on real data now.
2. `validate` (SCHEMA §7, per-arm PASS/INCOMPLETE with exact missing items), `capacity` (bytes per field/episode
   p50/p95/max, forecast for remaining arms, receiver/writer stats), `budget` (N/V/M, owner IR recomputed from
   counts with the arm's price table; π0.5 `.152/.848`, GR00T `.148/.852`, wrist look `.0646`, completion `.0502`
   measured prices, plus R4 assumed prices as a second ledger).
3. Deferred augmentation `debug/aug/` (SCHEMA §5): a batch job that loads π0.5 (`pi05_libero` PyTorch checkpoint)
   or GR00T (as `closed_loop/serve_groot.py` does) once and computes, from stored wire images/state/prompt:
   `policy_shadow` for every decision, `policy_draws` on the 1/32 sample, `shadow_look` (encoder keys → the arm's
   frozen A fit retrieval; pure-policy arms against both libraries), `camera_shadow` for π0.5 (wrist-only using the
   R7 wrist fit; third-only fit by the same library rule). Private seeded generators, batch inference, resumable
   (skip done parts), priority ordering by arm list, writes only under `aug/`. Reuse the model loading and stage
   code of `closed_loop/serve_*.py`, `stage_overrides.py`, `rounds/r07/analysis_scripts/a3_wrist_latency.py`.
   CPU fake-model tests; exact GPU validation commands for the coordinator (throughput per model at batch sizes
   1/8/32, and agreement of `policy_shadow` with live policy chunks on MISS decisions within sampling noise).
4. `fixtures.make_synthetic_arm` early (S4/S6 depend on it).

### S4 — decision-level profile tools (CPUs 26-29,70-73)
Own: `exp/offline_search/debug/tools/decision/`, tests there. Consume the reader API; develop on
`reader.p3v2_adapter` and the synthetic fixture until R8 smoke data exists. One CLI per tool
(`python -m exp.offline_search.debug.tools.decision.<tool> --run-root R --arms ... --out DIR`), JSON + CSV + a short
markdown summary, explicit denominators, `unavailable` instead of guesses, task/init cluster bootstrap where
intervals are reported, `--procs` explicit (default 4). Tools (from E1/E2/E4/E5 proposals; read their tables):
`provenance` (neighbour stage/mode/unknown mass, effective demos, cross-stage kernels, unsupported tails),
`divergence` (cache/served vs shadow policy by future offset, blind age, stage, library size; policy–policy noise
floor from `policy_draws`), `follow_vs_look` (closed-loop: served blind/follow block vs `shadow_look` at the same
observation; per-row follow-gap map; E4 prototype `rounds/r08/ideation/E4_lazy_levers/follow_vs_look.py`),
`camera_shadow` (wrist/third vs full retrieval agreement by stage), `stage_ledger` (work/IR by stage, per-visit
calls/looks, measured vs assumed prices), `call_value` (randomized call excursions with actual propensities,
ESS, supported strata only; E2 §2.2), `exposure_hazard` (follow lottery randomized contrasts by stage × age; E4),
`churn` (outcome flips vs A replicates / placebo; net bias vs symmetric churn), `trigger_vs_onset` (lead time and
per-stage hit rate of each online trigger vs S6's onset labels file).

### S5 — R8 arm methods, arm emission and prefits (CPUs 30-33,74-77)
Own: `exp/offline_search/rounds/r08/methods/`, `rounds/r08/ops/` (emit/prefit scripts), tests there.
Build new method classes (each with `debug_record()` and an identity test against its base where applicable):
1. **Follow lottery** — R7 `c1_follow` FollowExtension: at each cache anchor, structural support for E = 1, 2 extra
   blocks; hashed coin (`hash(seed, task_id, init, decision_seq)`) picks E uniformly among supported values
   {0..E_max}; log support, propensities, drawn E; compute R7 stage gate and state valve in shadow (logged, never
   enforced). Identity: forcing E = 0 reproduces A byte-for-byte.
2. **Wrist every look** — R7 `c2_wrist` StageWrist without the stage gate (full camera only at step 0 / forced looks).
3. **Wrist every 5 controls** and **A every 5 controls** (both cameras) — no blind block (check whether BlindAWM
   `budget=0` already does this; reuse if so).
4. **Shifted A placebo** — A whose first anchor commits 5 controls instead of 10, then unchanged.
5. **Identification probe** — independent call coin p = .25 at each fresh anchor, no stall, no cooldown, policy
   chunk committed like CU (policy tail one block); reuse R6/R7 CallController machinery if it can express this.
6. **Pure policy with 5-control decisions** — policy every anchor, policy tail on the blind decision (IR .5), live
   cache proposal computed where cheap; prefer reusing CallController (p ≡ 1) or `rounds/r04/k4_eval` SeededInference.
7. **Oracle grasp-window calls** (diagnostic; privileged `__oracle__` flag): 5a call the policy at every anchor
   inside the window; 5b only within 5 cm and at most 2 calls per goal object (E3 §4).
Then `rounds/r08/ops/emit_arms.py`: arm specs for SELECTION §2 (plus oracle 5a × 2, 5b × 4) in the R7 `arms.json`
format, reusing R7 fits for unchanged configurations (A/CU/CT/SF1/SW) by sha, and a prefit script for the new ones
(fits written to `/tmp/r8_S5/fits`; the coordinator copies). Budgets ρ = .30 (50) / .18 (500) with R7 calibration
artifacts; new random seeds for CU/CT (document). Tests: identity/propensity/coverage tests on library replays.

### S6 — physical / forensics / segmentation profile tools (CPUs 10-13,54-57)
Own: `exp/offline_search/debug/tools/physical/`, tests there. Same CLI/output conventions as S4. Start from E3's
scripts (`rounds/r08/ideation/E3_failure_forensics/{extract_compact,forensics,grasp_drop_pairs}.py`) and E4's
`twin_divergence.py`; generalize through an environment adapter (goal objects + destinations + success
sub-predicates; LIBERO first, interface ready for RoboCasa365/MetaWorld). Tools: `selfcheck` (E3 T9: naming,
post-settle reference, redundancy), `forensics` (T1 labels + onset, T2 sim-truth stages per control and per
decision), `grasp_audit` (T3, object-frame pre-grasp pose), `drop_audit` (T4), `paired_diverge` (T5, event-aligned),
`twin_divergence` (E4, GR00T bit-identical pairs), `blind_drift` (E4, robot and object-relative drift during blind
execution vs followed demos; uses backfilled library physics when present, else robot-only), `segmentation_bench`
(E1: R7 gripper baseline, stop/waypoint/change-point candidates, boundary quality vs T2 truth, causal prefix replay,
exposure-matched risk of the next 20 controls; holdout = inits 30–49 per task), `episode_card` / `arm_rollup`
(static HTML + PNG per episode and per arm). Develop on `reader.p3v2_adapter` (R6 pilot, inits 0–1) and fixtures.

## 3. Integration and order

S1 `schema.py` and S3 `fixtures.py` first. Coordinator integrates: GPU parity (S1), real-env capture parity (S2),
deferred-shadow throughput (S3), then a non-test smoke campaign (SELECTION §3) before the 500-episode arms.
