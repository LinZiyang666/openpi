# R7 ideation brief — stage-level allocation of inference budget

Written 2026-09-30 00:3x CDT by the coordinator. Owner is asleep; pause point 1 is waived. Read this whole file, then the
files it points to that matter for your direction.

## 0. Your role: explorer, not coder

You are one of **five explorers**. Your job is **exploration**: understand the problem, survey methods (robotics,
control, learning, systems), check ideas against the data we already have, and propose methods. You **may** write small
analysis scripts to get preliminary evidence, but writing implementation code is **not** the point; four separate coding
agents will implement whatever is selected. You also specify the **profile tools** you want: after coding, the
coordinator runs profiling first and then calls you back (same thread) to analyze your own profile results before the
full evaluation.

## 1. The question and the owner's framing

Goal of the line: keep success rate (SR) close to pure policy inference while pushing inference cost (IR) down, using a
retrieval cache over a demonstration library ("Commit-Cache").

Owner's R7 framing (2026-09-29, translated): classical robot control split a trajectory into stages (approach, grasp,
transport, place, retreat, ...), some hard and some easy. Today's learned robot policies spend exactly the same inference
on every step. LLM serving already routes easy queries to small models; robot policies have no such fine-grained cost
control. **R7 studies stage-level allocation: split trajectories into stages, tell hard stages from easy ones, spend more
on hard stages and apply the cheap "atomic levers" on easy ones.** The missing piece is the **signal for when it is safe
to be lazy**.

**Terminology (owner, fixed — use it):**
- **Atomic levers** (how to save): (a) *fewer calls* — lower the policy-call (MISS) rate; (b) *look less* — execute
  longer without vision by following the library trajectory; (c) *look half* — encode only one camera. (d) *cheaper call*
  (few-step denoising) exists but is shelved for R7.
- **Signals** (why it is safe to save here): state deviation (proprioceptive state vs the demo trajectory, free), stall
  detection (progress vs demo templates), stage label/difficulty. Predicted cache–policy disagreement (LOEO residual) was
  shown useless in R6.
- **Allocation policy** (at what granularity the budget is distributed): uniform (= random calls, the baseline) →
  task-level → **stage-level (R7)** → event-triggered; they can nest. Above them sits the **budget knob** (target IR).

**Method principles (owner, hard requirements):** *simplicity, effectiveness, generality.* The method must still work
when the robot, environment or benchmark changes (e.g. RoboCasa365, a real arm, a different policy). **No LIBERO-special
tuning**: no task names, no suite names, no hand-set thresholds in LIBERO units. Anything numeric must come from a
documented general rule applied to the library (and at most a handful of non-test recordings). Use only signals any
manipulator has: proprioceptive state (end-effector pose, gripper), the executed actions, the library itself, and the
policy's own vision encoder on the robot's cameras. **Do not replace or add models** (no CLIP/DINO/small encoders in the
method; the owner forbade model swaps).

## 2. The system today (facts, all measured)

Cost (owner IR, per 5-control decision): π0.5 `.152·v + .848·m`, GR00T `.148·v + .852·m`, where v = share of decisions
that run the vision encoder (both cameras) and m = share that call the full policy. Pure policy L10 (one inference per
10 controls) = .5, L5 = 1.0. Retrieval/cache CPU time (1–2 ms) is not counted. Wrist-only vision was priced at .055 per
look (+.050 for the other camera on a call) by a proportional-latency assumption in R4 (`rounds/r04/cost_table_owner.json`).
Latency split of one π0.5 inference (CUDA graph): vision 10.3 ms, language-model prefix 27.7 ms, denoising 29.6 ms.

- **A = Commit-Cache (pure cache, no calls).** At an anchor: vision keys (4×4-pooled tokens per camera) → PCA-64 per
  camera + 8-d state → per-task action-supervised Mahalanobis metric → top-16 kernel-weighted synthesis of library
  action chunks → execute the whole 10-control chunk (anchor decision + one blind decision). IR ≈ .076.
- **B = A + four guards** (stuck, at-end, overtime, no-progress) that trigger a policy call whose chunk is also executed
  for 10 controls. Only no-progress matters (it carries all of B's gain on L10-50, is harmful on GR00T Spatial-50).
- **C (R6) = A + budget knob ρ + random calls + calibrated stall detection** (template alignment of recent keys to
  successful library demos; `rounds/r06/ideation_Q3/stall/`). The R-placement part was useless and is removed.

Three-replicate A/B and pure references (SR @ IR):

| cell (model, suite, library) | A | B | pure L10 |
|---|---|---|---|
| π0.5 LIBERO-10, 50 | .714 @ .076 | .827 @ .181 | .904 @ .5 |
| π0.5 LIBERO-10, 500 | .827 @ .077 | .879 @ .159 | .904 |
| π0.5 Spatial, 50 | .837 @ .078 | .910 @ .140 | .986 |
| π0.5 Spatial, 500 | .976 @ .078 | .982 @ .119 | .986 |
| GR00T LIBERO-10, 50 | .611 @ .074 | .725 @ .221 | .866 |
| GR00T LIBERO-10, 500 | .830 @ .074 | .872 @ .188 | .866 |
| GR00T Spatial, 50 | .867 @ .075 | .876 @ .147 | .938 |
| GR00T Spatial, 500 | .964 @ .076 | .959 @ .127 | .938 |

Lowest observed IR whose SR point estimate reaches pure L10 (R6 frontier, `rounds/r06/frontier_final/`): π0.5 L10-500
.197, π0.5 Sp-50 .442, π0.5 Sp-500 .118, GR00T L10-50 .453, GR00T L10-500 .184, GR00T Sp-50 .298, GR00T Sp-500 .052;
π0.5 L10-50 never reaches it (best .894 @ .458).

Evidence relevant to R7:
- **Failures are diffuse for call placement:** calls placed at random anchors equal calls placed by predicted
  disagreement (8-cell −0.15 pp). R5 Q3 (`rounds/r05/q3_callvalue/HANDBACK.md`) found no context where skipping a call
  at a landmark was supportably safe. Calibrated stall detection adds ≈ +1.15 pp at equal budget (post hoc).
- **But failures concentrate by task:** giving the worst k tasks entirely to the policy (chosen on half the inits,
  evaluated on the other half) reaches pure SR at IR ≈ .29 on π0.5 L10-50 and ≈ .35 on GR00T L10-50
  (`rounds/r06/analysis_r6/per_task_routing/`). One task carries 44% of the cache gap on π0.5 Sp-50.
- **Grasp/place moments matter:** in R5, 36 of 97 cache failures (π0.5 L10-50) involved a grasp result that contradicted
  the library; replanning every 5 controls interrupted grasp/place micro-sequences, which is why committing 10 controls
  helped on LIBERO-10.
- **Look half (wrist camera only), π0.5, older retrieval:** Spatial-500 same SR at 41% less IR; Spatial-50 +3.6 pp
  (replicated, p = .009); L10-50 unchanged; L10-500 −2.9 pp. Wrist + blind + tail on Spatial-500: .990 @ .074
  (R4/R5 ledger; `rounds/r05/q6_wrist_blind/`). Not yet re-tested on today's A, not tested on GR00T.
- **Look less:** GR00T executing 15 of its 16-step chunk costs −0.3 to −4 pp for 1/3 less vision. π0.5 has 10-step
  chunks; going further requires following library successor rows — never tested. R1 profiling found that
  "following the library trajectory forward" had the lowest action error of all retrieval behaviours.

## 3. Data and tools you can use (read-only)

- Library store `/home/weiland/trace_runs/offline_search_store/library/<m>_<s>/<name>/` (m ∈ pi05, groot; s ∈ l10,
  spatial; name = current (50-episode) | bpool_all / bpool_cs (500)). Arrays per decision row: `key_v0/key_v1`, `rs`
  (state), `action` (H×32, first 7 dims valid; H = 10 π0.5, 16 GR00T), `task_id`, `episode`, `step`, `ep_len`,
  `progress`, `success`, `traj`, `prev`/`next` row indices. See ledger §5.1 (`logs/offline_search_exploration.log.md`).
- Closed-loop run roots `/home/weiland/trace_runs/os_closed_loop/<run>/runs/<arm>/`: `summary.json` (cost_ledger),
  `client/journal.jsonl` (accepted outcomes), `server_*/decisions_*.jsonl` (one row per decision: vision, src, judge,
  look_reason, extras incl. stall state / coin / p for C arms). Per-(task, init) outcomes of every arm:
  `rounds/r06/frontier_final/outcomes.json`; arm table `frontier_points.csv`.
- **Natural randomized experiments:** the uniform-lottery arms (`r06_c_validation/runs/r6c_*_U30`, `_U18`, and the risk
  lottery arms in `r06_frontier`) place calls by a hashed coin per (task, init, step) — useful to estimate *where* a call
  is worth something. R4 K5 randomized arms (`r04_k5`).
- **Superset profiler recordings** (R6 P3 pilot, `rounds/r06/p3_profiling/SCHEMA_V2.md`): strict tables under
  `/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/<cell>/` (decisions, anchors, controls = per-control
  telemetry incl. robot state, action_steps, neighbours) with a same-observation shadow policy chunk at every anchor
  (tasks 0–9 × test inits 0–1). B-val (non-test) recordings of A with shadow chunks: `r06_c_cal/tables/<cell>/`.
- Profile tools `exp/offline_search/profile/` (ledger §5.3): coverage, explain, breakdown, compare, timeline, repr,
  runprof. The closed-loop plugin and its hooks: `exp/offline_search/closed_loop/README.md`, blind decisions
  `closed_loop/blind.py`, R4 blind/wrist code `rounds/r04/k1_blind/`, `rounds/r04/k3_cost/`.
- Round summaries: `rounds/r0{1..6}/FINDINGS.md`; R6 report `rounds/r06/ANALYSIS.md`.

## 4. Deliverable

Write `exp/offline_search/rounds/r07/ideation/<your dir>/PROPOSAL.md` (English, ≤ ~400 lines) containing:
1. **Direction summary** — what you explored, what you read/checked, what surprised you.
2. **1–3 method proposals**, each with: hypothesis; mechanism (which levers, which signals, which allocation policy, how
   stages are defined and how hard/easy is decided); **generality argument** (inputs used, how every number is
   calibrated from the library by a general rule, what changes on another robot/benchmark); expected effect per cell and
   on the IR–SR frontier; cost tier and online CPU cost; implementation sketch against the plugin (what the method's
   `query`/judge/blind hooks must do); **kill criteria** (what result would make us drop it).
3. **Preliminary evidence** from existing data (numbers, with the command/script you used; keep scripts in your dir).
4. **Profile tools wanted**: each with the question it answers, inputs (existing data or new telemetry), outputs, and
   which proposal decision it informs. Prefer tools that run offline on existing logs/recordings; say explicitly if new
   closed-loop telemetry is required and what must be logged.
5. **Risks and open questions.**

## 5. Rules

- Write only inside your ideation directory and `/tmp/r7_<your dir>/`. Everything else is read-only.
- Every Python command: `taskset -c <your CPUs> env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python` from `/home/weiland/projects/openpi`;
  at most 2 Python processes at once. Never use CPUs 38–43 or 82–87. No GPU.
- Do not start servers, workers, chains or simulators; do not touch tmux, ports, other processes or remote hosts.
- No git state changes. No `rm -rf`, no `pkill -f`. Do not read `tests/review_tests/`.
- Aim to finish in about 60–90 minutes of work. Be concrete; numbers over adjectives.
