# R8 ideation brief — profile tooling for stage research, and a full data-collection round

Written 2026-09-30 11:3x CDT by the coordinator. The owner is away; pause point 1 is waived. Read this whole file,
then whatever it points to that matters for your lens.

## 0. Your role: explorer, not coder

You are one of **five explorers** (three GPT-6 astra, two Claude Opus), each with a different lens (§4). Your job is
**exploration and specification**: understand what we must measure to study trajectory **stage segmentation** and
stage-level allocation, look at the data we already have, and specify the **profile tools** and **debug-mode data
fields** you want. You may write small analysis scripts to get preliminary evidence; implementation is **not** your
job (separate GPT-6.1 sol coding agents will implement the selected tools). After the tools are built and the data
collected, the coordinator will call you back on the same thread to analyze your own profile results.

## 1. The owner's R8 directive (verbatim, then translated)

> R8我们专心做profile工具，把profile集成到debug工具，并收集真实运行的trajectory数据全量eval 500集，你先发opus，
> astra去研究为了研究stage分段，他们还需要什么profile工具，或者他们还想要其他的什么profile工具，之后真实运行全量eval
> 收集数据，可以做的组是1.纯inference（最早的debug工具好像收集了），2. 没有stage切分时的系统，（只有A+停滞+随机miss）
> 3.带上你之前切分的系统 4.你不是上个阶段做了更少看/看一半吗，可以系统研究这两个的影响 5. 以及其他你想做的实验并收集
> 数据，数据理论上放在home盘，如果放不下，就把之前一些古早的现在用不到的数据移动到arxive。

Translation: **R8 is a data-collection round.** (a) Build profile tools and integrate them into the system's own debug
tool, which should collect *everything that can be collected*. (b) Collect real closed-loop trajectory data, full
evaluation, **500 episodes per arm** (the standard 10 tasks × 50 test inits). (c) Explorers first decide what profile
tools are needed to study stage segmentation, and what other profile tools they want. Arm groups the owner listed:
1. **Pure policy inference** (the earliest debug tool may already have collected this — see §3.1);
2. **The system without stages**: A + calibrated stall + uniform random calls;
3. **The system with R7's stage split** (stage-tilted calls; optionally the stage-gated levers);
4. **A systematic study of the two R7 levers "look less" and "look half"** (their effect, not a method claim);
5. **Anything else the explorers want measured.**
Data goes on `/home` (SSD, 2.1 TB free now); if it does not fit, old unused data is moved to `/archive`.

Owner principles still hold: **simplicity, effectiveness, generality** — the debug tool must work for any method and
both models, and should port to other benchmarks (RoboCasa365, MetaWorld) and robots; enabling it must not change
what the robot does.

## 2. Why R8 (what went wrong in R7)

R7 tried stage-level allocation. Its stages were the crudest possible version: a two-cluster split of the library's
executed gripper command (open/closed), with ±1 library row around each gripper change called "event/hard". Spatial
tasks get 2 stages (before grasp / after grasp), LIBERO-10 gets 4. The owner's verdict: *this segmentation segments
nothing*. The genuinely hard moments (fine alignment before a grasp, placement alignment, insertion) sit inside the
"interior = easy" stage. R7's explorer E1 also built a finer geometric segmentation (16–17 segments on L10, 7 on
Spatial) and a difficulty score, but the score failed to transfer (hard/other under-motion risk ratio 2.67 on GR00T
L10-50 vs .57 on π0.5 L10-50; boundary-tightness AUROC < .5 in 7/8 cells), so only the gripper split was used.

Consequences:
- R7's result "placing calls by stage does not help" (first run +2.7 pp, replicate +0.85 pp [−0.80, +2.55], dense
  libraries flat) is **conditional on that crude segmentation**; it does not refute the owner's idea.
- We have **no per-episode root-cause diagnosis**. The formal R7 runs logged per decision: 16 neighbours (library
  rows, weights, scores), 8-d robot state, served actions, look/call flags and reasons, timings. They did **not** log
  camera images, the policy's own action at cache decisions, or any simulator ground truth (object poses, contacts,
  grasp state). R7's profile run (20 non-test episodes per arm) had images but no shadow policy and no sim state,
  because the R6 superset client (§3.2) only works with the R6 profiler method and was bypassed.
- R7 levers, for reference (pooled over cells; SR vs A): look-less with stage gate + state valve −1.78 pp at IR
  ≈ .060–.069 vs A's ≈ .076; the same lever without gating is ≈ 2 pp worse still; wrist-only look in easy stages
  (π0.5) −0.05 pp at measured IR .062–.073. Small libraries (5 demos/task) give only 3.6–4.2 effective demos per
  extension and ~2× the drift of 500-episode libraries.

Current reference numbers (SR @ owner IR; 500 test pairs):

| cell (model, suite, library) | A (pure cache) | pure L10 policy |
|---|---|---|
| π0.5 LIBERO-10, 50 / 500 | .714 / .827 @ .076 | .904 @ .5 |
| π0.5 Spatial, 50 / 500 | .837 / .976 @ .078 | .986 |
| GR00T LIBERO-10, 50 / 500 | .611 / .830 @ .074 | .866 |
| GR00T Spatial, 50 / 500 | .867 / .964 @ .075 | .938 |

Owner IR per 5-control decision: π0.5 `.152·v + .848·m`, GR00T `.148·v + .852·m` (v = share of decisions that run the
vision encoder, m = share that call the policy). Systems: **A** = per-camera PCA-64 + state → per-task
action-supervised Mahalanobis → top-16 kernel synthesis → execute 10 controls (anchor + one blind decision).
**C / "uniform calls + stall"** = A + budget knob ρ + a hashed-coin random policy call at each anchor + calibrated
stall detection (template alignment to successful demos) that forces a call. **Stage-tilted calls** = same, with call
weight ×1/h near gripper events and at the first entry into high state deviation (λ re-solved to keep IR = ρ).

## 3. Existing collectors, tools and data (all read-only for you)

### 3.1 The earliest debug tool: trace collector (pure inference already collected)
- Server flags `--trace-out <dir> --trace-build-cache` (`scripts/serve_policy.py`, `src/openpi/cache/interceptor.py`,
  `src/openpi/cache/config.py`): every module runs at every step and is written to per-episode HDF5.
- Data `/home/weiland/trace_runs/dual_20260923/` (629 GB): 8 arms × 500 episodes, π0.5 and GR00T × LIBERO-10/Spatial,
  **pure inference** (`_inf`, top-1 old-style cache computed in shadow) and **pure top-1 cache** (`_cache`, policy in
  shadow). Cadence: replan every 5 controls (L5). Per step: `vision_*`, `prompt_emb`, `robot_state`, clean and all
  noise actions, input/raw images, `trace/actions/{executed,full_hit,full_inference}`, query keys, search top-k.
  Report: `exp/trace_dual/analysis/results.md`, `data_quality.md`. The offline library store was extracted from it.
- Gap to check: does it contain simulator ground truth (object poses, contacts)? It uses the old retrieval (weighted
  score sum, top-1), not today's Commit-Cache, and L5, not the L10 reference cadence.

### 3.2 R6 superset profiler (P3 v2): the richest collector we have
- `exp/offline_search/rounds/r06/p3_profiling/` — `SCHEMA_V2.md` (read first), `HANDBACK.md`, `STREAMING.md`,
  `telemetry.py`, `snapshots.py`, `worker_v2.py`, `run_gtp_v2.py`, `read_v2.py`, `client_bundle*`.
- Server side (`v2:Profile` method): per decision the shadow policy chunk (private noise), cache proposal, 16
  neighbour chunks, keys, retrieval, guard diagnostics, stage timing; lossless input NPZ.
- Client side (on timan107 workers): `controls.jsonl` — every issued control with `after.*` simulator fields
  (`body_xpos/xquat`, `contacts`, `cfrc_ext`, `efc_force`, `observation_numeric.<object>_pos/_quat/_to_robot0_eef_*`,
  `event_labels.grasp_slip_release`, actuator commands per physics substep); `step_N.npz` snapshots (pre-inference
  simulator state, RNG states, images; `restore_certified=false`). Streaming to a local receiver.
- Data: `/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/` (261 GB ≈ 60 MB/episode; 216 arms, 4,320 episodes,
  test inits 0–1 only). Strict tables `tables/<cell>/{episodes,decisions,anchors,controls,neighbours,action_steps}.csv`.
- **Limitation:** the v2 client wrapper requires a v2 server marker; ordinary plugin arms fail every episode with it.

### 3.3 Ordinary closed-loop logs (every arm since R2)
- Run roots `/home/weiland/trace_runs/os_closed_loop/<run>/runs/<arm>/`: `summary.json` (cost ledger),
  `client/journal.jsonl` (accepted outcomes), `client/per_step.jsonl` (per decision, client side),
  `server_*/decisions_*.jsonl` (per decision: rows/weights/scores of 16 neighbours, robot_state, served_head,
  vision/src/hit, look_reason, blind_age, timings, method extras). `--os-log-inputs` adds per-episode input NPZ
  (images/keys; used in `r07_profile_bval1`, ≈ 13–28 MB/episode on GR00T).
- Plugin and flags: `exp/offline_search/closed_loop/README.md`, `plugin.py`, `blind.py`, `ops/` (chain, collect,
  remote launchers). Run lanes and chains are coordinator-owned.
- R7 formal runs: `r07_main` (44 arms), profile runs `r07_profile_bval1`. R7 report `rounds/r07/ANALYSIS.md` (§6–§8
  mechanism tables), selection `rounds/r07/SELECTION.md`.

### 3.4 Offline profile tools
- `exp/offline_search/profile/` (coverage, explain, breakdown, compare, timeline, repr, runprof; README).
- R7 toolkit `exp/offline_search/rounds/r07/c4_profile/` (follow_audit, camera_audit, stage_value, failure_clock,
  profile_report; README) and R7 stage table `rounds/r07/stages/{stages.py,STAGES_API.md}`.
- R7 explorer reports `rounds/r07/ideation/E1..E5/{PROPOSAL,PROFILE_ANALYSIS}.md` (E1 = segmentation work above).

### 3.5 Library store
`/home/weiland/trace_runs/offline_search_store/library/<m>_<s>/<name>/` (m ∈ pi05, groot; s ∈ l10, spatial; name =
`current` (50 episodes) | `bpool_all` / `bpool_cs` (500)): per decision row `key_v0/key_v1`, `rs` (8-d state),
`action` (H×32, first 7 dims valid), `task_id`, `episode`, `step`, `ep_len`, `progress`, `success`, `prev`/`next`.
Ledger §5.1 in `logs/offline_search_exploration.log.md`.

## 4. Lenses (one per explorer)

- **E1 — Stage segmentation (astra).** What must be measured to build and *validate* a segmentation whose "hard"
  stages are where the cache actually fails or where a call pays off? Survey classical and learned segmentation
  (gripper/contact events, velocity minima, waypoint extraction, object-relative bottleneck poses, demo dispersion,
  cache–policy divergence, change-point methods) and say, for each, which data fields and tools are needed, and what
  validation would convince us. Keep generality in view: inputs any manipulator has.
- **E2 — Where a call or a look is worth it (astra).** Tools for cache-vs-policy divergence over time and by stage,
  error growth with time since the last look, the value of a call placed at a given moment (natural randomized
  experiments from random calls; snapshot-branch counterfactuals — are they feasible given `restore_certified=false`,
  and what would certification need?), and per-stage cost accounting.
- **E3 — Failure forensics (Opus).** Automatic per-episode diagnosis from simulator ground truth: failure taxonomy
  (missed grasp, slip/drop, wrong object, misplacement, collision, stall/oscillation, timeout), onset detection,
  timelines, and paired comparison of the same (task, init) under pure policy vs cache to find where trajectories
  diverge. Which client fields are required; what the standard report looks like.
- **E4 — Look less / look half, systematically (Opus).** Owner group 4: design the factorial study of the two levers
  (extension length, gating none / state valve / stage, camera choice, library size, model), the telemetry that
  explains *why* each helps or hurts (drift during blind execution vs the demos, wrist-only retrieval quality vs
  full, by stage), an arm list with priorities, and its time/storage cost.
- **E5 — Debug-tool integration (astra).** Inventory §3.1–§3.4 in code, then design **one debug mode** for the
  closed-loop system that any method and both models can switch on: server fields (shadow policy with isolated RNG,
  images, neighbour provenance incl. library episode/step/success/stage, all method decisions and reasons) and client
  fields (per-control simulator ground truth, snapshots policy), storage format and per-episode size budget,
  streaming/collection for ~50 arms × 500 episodes, behaviour invariance (identical actions with debug on vs off:
  how to prove it), what to reuse from P3 v2 / trace / C4 versus rewrite, and how it ports to RoboCasa365/MetaWorld.

## 5. Deliverable

`exp/offline_search/rounds/r08/ideation/<your dir>/PROPOSAL.md` (English, ≤ ~350 lines). Opus explorers: return the
full text as your final answer (the coordinator saves it; subagents cannot write report .md files here).
1. **Summary** — what you read/checked, what surprised you.
2. **Profile tools wanted** — table: tool name; question it answers; inputs (existing data vs new debug fields);
   outputs; which stage-segmentation decision it informs; priority P0/P1/P2; rough CPU cost. Prefer tools that run
   offline on the debug data.
3. **Debug-mode data fields required** — table: field; side (server per decision / client per control / per
   episode); why; size estimate; priority. Say what is *not* worth collecting.
4. **Arms you want collected** (500 episodes each) — per owner group 1–5: which cells (model × suite × library),
   which variants/parameters, why, priority. Keep the total realistic (§6).
5. **Preliminary evidence** (optional) from existing data, with the command/script you used (scripts in your dir).
6. **Risks and open questions.**

## 6. Budget facts

- Throughput: R7 ran ≈ 5k episodes/hour with 4 lanes (1 server each on the local RTX 4090, 22 LIBERO workers each on
  timan107; timan107 holds ≈ 85 LIBERO-10 or ≈ 120 Spatial workers). A shadow policy inference at every decision makes
  a cache arm cost like pure inference on the GPU; GR00T inference is the slow one.
- Storage: `/home` has 2.1 TB free. R6 superset ≈ 60 MB/episode; trace collector ≈ 4 MB/decision (π0.5).
  50 arms × 500 episodes at 60 MB/episode would be 1.5 TB — so field choices and sampling matter.
- Realistic total: on the order of 40–60 arms.

## 7. Rules

- Write only inside your ideation directory and `/tmp/r8_<your dir>/`. Everything else is read-only.
- Every Python command: `taskset -c <your CPUs> env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python` from
  `/home/weiland/projects/openpi`; at most 2 Python processes at once. Never use CPUs 38–43 or 82–87. No GPU.
  Big HDF5/CSV scans: read only what you need (the SSD is shared with running work).
- Do not start servers, workers, chains or simulators; do not touch tmux, ports, other processes or remote hosts.
- No git state changes. No `rm -rf`, no `pkill -f`. Do not read `tests/review_tests/`.
- Aim to finish in about 60–90 minutes. Be concrete; numbers over adjectives.
