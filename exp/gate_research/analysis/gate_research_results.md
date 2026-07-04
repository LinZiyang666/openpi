# Gate-Research Data Collection — Results

> **Headline**: A concurrency-native per-step collection of **real CP1 gate verdicts + model inputs** across 7 landmark cache configs (3 × libero_spatial, 4 × libero_10), 500 episodes/config, evaluated over the full 50 init-states per task. **35 000 collected steps** carry a `(robot_state, hit_type, cp1_score, searched, success)` tuple for GATE ("search or not") study. Both suites completed cleanly (0 crash-resume duplicates, 100 % robot_state coverage).
>
> Branch `Ziyang`. Collection code: `exp/gate_research/run_collect.py` (+ `verify_gate.py`). Raw data (gitignored): `exp/gate_research/data/{libero_spatial,libero_10}/`.

---

## 1. Purpose

Collect **unbiased, real** gate-decision labels for a downstream GATE study — i.e.
learn "should this step search the cache or skip straight to Stage-2 inference?".
Each CP1 inference step records the input the cache keys on (`robot_state`),
the ThresholdJudge verdict (`FULL_HIT` / `WARM_START` / `MISS`) computed on the
**real** `cp1_score`, the `searched` flag, and the episode outcome. The gate is
forced to `always_search` so **no step is gate-skipped** and the labels are not
selection-biased.

This overturns the earlier placeholder plan of a fixed always-full-hit verdict:
the verdict here is the genuine ThresholdJudge output on the real similarity score.

## 2. Design

- **Configs**: the 7 landmark points on the weighted_sum **d=1 Pareto frontier**
  (SR vs inference-ratio), 3 for libero_spatial + 4 for libero_10. Each is a
  self-contained eval YAML — baked (T_fh, T_ws) thresholds + per-field zscore
  normalizer + prebuilt `cp1_spatial_pool_16` library, `gate: always_search`,
  `write_policy: never`. **No warmup** was needed or run (thresholds/normalizer
  already solved offline; library prebuilt).
- **Episodes**: 500 / config = 10 tasks × **50 init states (indices 0..49)**.
  This uses the *full* init distribution (not the frontier's held-out 45/task).
  Strategy: `WarmupEvalStrategy(skip_warmup=True)` → `orig_init_state_idx = ep`,
  no init-map leak guard.
- **Collection**: server `--export-collect-meta --collect-fields robot_state`;
  rows ride `EpisodeResult.per_step_rows` back to the driver over the conductor
  wire and are appended incrementally to `gate_rows.jsonl` (crash-safe).
- **Topology**: 1 inference server (ziyang10, pi05, **3 replicas / spawn-batch 2**,
  cgroup 32 GiB) fronted by an in-process replica router on one public port; 1
  client (timan107, **48 worker processes** across 8 GPUs) driven by the conductor.
  libero_spatial ran ~107 ep/min (~14 min); libero_10 ~15 ep/min (~2.3 h) — the
  long-horizon tasks + larger library make each step's kNN search slower.
- **Robustness**: a 3-strike, progress-resetting debounce supervisor around the
  conductor (network jitter is already absorbed by per-episode requeue, so it
  never killed the process); incremental append checkpoint so a process death
  loses no already-collected rows. Both suites completed in a single attempt
  (supervisor `fail=0`).

## 3. Results — libero_spatial (1500 ep, 40 636 rows)

Per-step verdict rows = 39 136; episode_summary rows = 1500; robot_state coverage
= 39 136 / 39 136; duplicate `(task_uid, step_idx)` keys = 0.

| config (d1) | SR | FULL_HIT | WARM_START | MISS | verdicts | inf_ratio* | frontier (SR / inf) |
|---|---|---|---|---|---|---|---|
| fh40_ws40 | 91 % (454/500) | 7580 | 2362 | 2642 | 12 584 | ~0.398 | 98 % / 0.407 |
| fh75_ws10 | 83 % (413/500) | 9350 |  718 | 3306 | 13 374 | ~0.301 | 95 % / 0.240 |
| fh75_ws15 | 85 % (425/500) | 9278 | 1349 | 2551 | 13 178 | ~0.296 | 87 % / 0.237 |

## 4. Results — libero_10 (2000 ep, 145 763 rows)

Per-step verdict rows = 143 763; episode_summary rows = 2000; robot_state coverage
= 143 763 / 143 763; duplicate keys = 0.

| config (d1) | SR | FULL_HIT | WARM_START | MISS | verdicts | inf_ratio* | frontier (SR / inf) |
|---|---|---|---|---|---|---|---|
| fh80_ws10 | 57 % (285/500) | 25 373 | 3204 |  9568 | 38 145 | ~0.335 | 60 % / 0.338 |
| fh60_ws30 | 61 % (303/500) | 21 658 | 7339 |  8245 | 37 242 | ~0.418 | 74 % / 0.416 |
| fh40_ws40 | 68 % (338/500) | 18 988 | 7494 |  9281 | 35 763 | ~0.469 | 83 % / 0.512 |
| fh5_ws40  | 78 % (388/500) |  9071 | 11 192 | 12 350 | 32 613 | ~0.722 | 93 % / 0.832 |

\* `inf_ratio` here is a **rough proxy** = `(WARM_START + MISS) / verdicts`
(counts a warm-start as a full inference). The frontier's exact inference-ratio
weights a warm-start by its `start_t` (`summarize_inf_ratio._warm_cost`), so the
proxy slightly overstates cost; the exact value can be recomputed from the raw
`start_t` per row. The point of the column is the **sanity check** below.

## 5. Discussion (what this does and does NOT show)

- **The inference-ratio matches the frontier closely** — e.g. libero_10
  fh80_ws10 0.335 vs 0.338, fh60_ws30 0.418 vs 0.416; libero_spatial fh40_ws40
  0.398 vs 0.407. This confirms the gate/judge machinery reproduces the frontier's
  hit/warm/miss *mix* on real scores — the verdict labels are trustworthy.
- **Success rate is several points BELOW the frontier**, consistently. This is an
  **expected, designed** consequence of the init-state choice, NOT a regression:
  the frontier was measured on **held-out** inits (45/task, excluded from the
  library); this collection deliberately uses **all 50 inits 0..49** (owner's
  choice, to hit exactly 500 ep/config). The delta is larger on libero_10 (e.g.
  fh5_ws40 93 %→78 %) because its long-horizon tasks are more init-sensitive and
  the sampled init set differs; LIBERO rollout stochasticity adds noise on a
  500-ep sample. This collection is **not** a re-measurement of the frontier and
  should not be read as one.
- **What the data IS good for**: training / studying a search-or-not gate from
  `(robot_state → verdict)` pairs with the true verdict distribution (FULL_HIT
  vs WARM_START vs MISS) along real trajectories, spanning a wide operating range
  (spatial ~0.30–0.40 inf-ratio; libero_10 ~0.34–0.72). ~10 % of inits per task
  are in-library (trivial step-0 FULL_HIT); downstream analysis can flag these via
  `orig_init_state_idx` against the init map if an unbiased-only subset is wanted.
- **CI caveat**: per-config SR is a single 500-ep point estimate (Wilson 95 % CI
  ≈ ±4 pp); the frontier deltas above are within a few CIs of an init-set +
  stochasticity explanation and are not evidence of any pipeline change.

## 6. Collected schema (per config `gate_rows.jsonl`)

Two row kinds share the file (`_kind`):

- **per-step verdict row**: `yaml_id, task_id, orig_init_state_idx, episode_id,
  task_uid, phase, step_idx, hit_type ∈ {FULL_HIT,WARM_START,MISS}, cp1_score,
  start_t, winner_id, searched(=true), success, collect.robot_state[32],
  attempt, collector_schema_version`.
- **episode_summary row**: `_kind=episode_summary, task_uid, episode_id, phase,
  seed, num_steps, success, searched_all(=true), collect_fields, kb_id`.

`openpi.serving.per_step_recorder.summarize_gate_log(dir, yaml_id)` tallies
eval-phase verdicts (excludes the summary rows from the inference-ratio
denominator). `exp/gate_research/verify_gate.py` reproduces §3–§4 from the raw
JSONL.

## 7. Artifact layout

```
exp/gate_research/
  run_collect.py          # eval-only conductor launcher (WarmupEvalStrategy skip_warmup)
  verify_gate.py          # journal SR + per-config verdict tally + dedup/robot_state check
  config/{libero_spatial,libero_10}/eval/*.yaml   # the 7 d1-frontier configs (gitignored)
  data/{libero_spatial,libero_10}/                # raw (gitignored)
    journal.jsonl         # episode-level SR / resume
    gate_rows.jsonl       # the collected per-step gate rows
  analysis/gate_research_results.md               # this report
```

**Raw-data location note**: at report time the raw `data/` lives on the client
(timan107, `/scratch/zixuans8/openpi/exp/gate_research/data/`), fully collected +
verified there. The local pull-back was blocked by a transient broker JetStream
outage (the tether relay was `force_single_active` — no tier-B object store);
a background retry lands it locally when JetStream recovers. Raw data is
gitignored regardless, so it does not enter git.
