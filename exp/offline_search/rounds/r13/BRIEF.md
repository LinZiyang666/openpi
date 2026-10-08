# R13: Layer 4 in detail. Periodic calls vs distance-threshold calls

Coordinator brief, 2026-10-04 CDT. It is given to the explorer (astra). The coding agent (sol) builds what the explorer specifies.

## 1. The owner's goal
Owner, 10-04: "下一个阶段研究两种第4层调用详细对比" ("next phase: a detailed comparison of the two layer-4 call methods").

On 10-03 the owner put both methods formally into layer 4 and asked for a careful comparison later:
- **periodic calls:** a guard-aware gap cap;
- **distance-threshold calls:** call the policy when the retrieval distance is above a threshold.

Layer 4's job, in the owner's words, is to **close the SR gap between pure cache and pure inference**. How far to turn the knob is set by the user's SR target. So both methods are knobs that trade IR for SR, and the comparison is about that trade.

R11 measured both, but not on equal terms:
- the distance thresholds were calibrated offline;
- in closed loop they spent far less than targeted (−.080 IR on average, worst −.19, 2/17 arms within ±.02);
- so R11 could only compare the two on the realized-IR frontier, with distance arms crowded at low IR.

R13 must give the owner a **fair, detailed comparison**. That means making distance a working knob as well (or showing that it cannot be one), then comparing the two in closed loop at matched realized IR, and explaining where and why each one wins.

## 2. The deployed system (five layers; only layer 4 is studied)
1. **Cache:** R4 BlindAWM retrieval synthesis on both cameras with a per-task Mahalanobis metric. It commits 10 controls per decision; kref is 5 or 8.
2. **Only-no-progress guard.**
3. **In-library LOEO motion corrector,** with strength attenuated by distance.
4. **IR knob:** periodic and distance-threshold (this round).
5. **Per-task partition:** per-task retrieval scope, metric and corrector heads, inside layers 1–3.

Hard rule: **layer 4 has no task-indexed parameters.** A cell-wide scalar per model × suite × library size is allowed. Task and init may only key random coins. If something you propose needs per-task values, say so: it then belongs to layer 5 and is out of scope unless you argue for it explicitly.

Look-half and the R12 H2 guard are in the appendix and are **not** part of this round. Layer 2 is the original no-progress guard.

**Code:**
- `exp/offline_search/rounds/r11/knob/` (`R11Knob`, nine methods; `off` is decision-for-decision equal to R10Recipe):
  - controller and recipe: `recipe.py`, `controller.py`;
  - calibration and fitting: `calibration.py`, `fitting.py`;
  - builder: `build.py`;
  - read its `README.md` and `HANDBACK.md`.
- `exp/offline_search/rounds/r10/recipe/` (`R10Recipe`, layers 1–3+5).
- The closed-loop plugin: `exp/offline_search/closed_loop/{plugin.py, blind.py}`.
- Look recording: `closed_loop/look_recording.py` (reader `read_looks`; see `rounds/r12/prep/HANDBACK.md`).

**Method definitions:**
- R11 periodic: `rounds/r11/opus/SPEC.md` §2. It is a dithered gap cap; a guard call resets the gap.
- R11 distance: `rounds/r11/astra/SPEC.md`. Nearest geometric distance in the inherited metric, with one cell-wide threshold calibrated on whole-episode-held-out library replays.

## 3. What R11 established (aggregates; details in `rounds/r11/REPORT.md`, `RESULTS_TEST.md`, `opus/ANALYSIS.md`, `astra/ANALYSIS.md`)
- **Budget control:**
  - periodic was within ±.02 on 17/17 arms (mean −.006);
  - distance was within ±.02 on 2/17 (mean −.080).
- **Why distance underspends:** closed-loop query distances sit lower than the calibration distances. In π0.5 L10-50, the median was 9.4 against a threshold of 12.46, so only 19% of looks crossed the threshold instead of 48%. The larger donor bank explains only 0.1–2.7 pp of this; the rest is closed-loop state shift, which is not proven to be pure demo tracking.
- **Value:**
  - At matched (cell, target), periodic − random was +1.48 pp (p = .0005) and distance − random was −1.19 pp. But distance spent much less, so this comparison is not at matched realized IR.
  - Efficiency (ΔSR / ΔIR over off) was periodic .293 and distance .215.
  - Periodic was the only method stably above the "whole-episode mix" line.
- **Mechanism (periodic):**
  - few calls right after a guard call;
  - fewer calls in doomed long episodes;
  - cache-only runs of at most about 30 controls.
  - Gains come almost entirely from inits that the cache fails.
- **Noise:**
  - single test arms carry 2–5 pp batch noise; the same configuration measured twice differed by 4.8 pp once;
  - GR00T 4090 vs H100 can differ by about 5 pp in spatial (R12), so all arms of one GR00T cell run on one host;
  - π0.5 shows no host effect.
- **astra's R11 future hypothesis:** calibrate on external B episodes with the full donor bank, and drive total cost with a global committed-ledger feedback. It is in `rounds/r11/astra/ANALYSIS.md` (last section). It is a hypothesis, not a result.

## 4. What the comparison must answer (minimum; add what you think matters)
- **Q1, budget control.** Can each method hit a target realized IR in closed loop, using library-only or B-dev-only calibration, across cells and doses?
  - For distance, find the best honest way to make it a knob: closed-loop B-dev calibration of the threshold, fixed-point recalibration (its own calls change the state distribution), online cost feedback on the threshold, a quantile tracker, or anything else.
  - It must stay a distance-threshold call, meaning distance decides *when* to call.
  - Report what each needs: fitted scalars, closed-loop dev data, and what happens at library sizes with no dev inits (size 500).
- **Q2, value at matched realized IR.** Compare SR against realized IR per cell, from knob off up to near pure inference (include high doses such as .40–.45 where the gap is still open). Run paired comparisons on the same inits, in the same batch and on the same host, with enough power to resolve about 1.5–2 pp pooled. Which wins, by how much, and where: model, suite, library size, dose?
- **Q3, mechanism.**
  - Where calls land: episode phase or progress, relation to guard calls, gripper events, and how concentrated they are by task (descriptive only).
  - Rescue of cache-failing inits versus harm to cache-succeeding inits.
  - Cache-only run lengths, episode length and time to success.
  - Wasted calls, i.e. calls where cache ≈ policy. Use shadow labels on B dev.
- **Q4, signal quality in closed loop.** Does distance actually predict where a policy call matters in closed loop (policy-vs-cache action error, or outcome)? Compare this with the offline library picture.
- **Q5, combination.** Both methods are in layer 4. Is a combination better than either alone at matched IR? Examples: distance placement under a periodic budget, a periodic floor with a distance trigger, or a distance trigger with a periodic gap cap. Optional, but if you test it, test it properly.
- **Q6, operating properties.**
  - Sensitivity to the guard rate and to library size.
  - Behaviour on long versus short suites.
  - Determinism and latency (maximum cache-only run).
  - What a deployer must calibrate.

## 5. Data and the hard data rules
**Allowed for fitting, calibration, selection and design:**
- B-pool libraries: `/home/weiland/trace_runs/offline_search_store/library/<model>_<suite>/{bpool_cs | bpool_all | current}` and R10 nested subsets (`rounds/r10/data.py` `SubsetLibrary`).
- B-pool dev closed-loop roots under `/home/weiland/trace_runs/os_closed_loop/`. These use B-pool inits outside the served subsets:
  - `r11_dev_current`, `r11_dev_size50`, `r11_dev_size200`;
  - `r11_devknob_50` (20 arms, 50 eps; knob off, random, periodic, distance and error-hybrid at .32 in the four size-50 cells);
  - `r12_devlooks_{50L,50S,200,200_local,50S_local,400,knob}`: look recordings with per-look keys of both cameras, distances, weights and guard state; knob off, plus periodic .32 for the two L10-50 cells.
  - Selections: `r11_dev_k10_size{50,200,400}`.
- The decision logs (`runs/<arm>/server_*/decisions_*.jsonl`) hold every decision with extras. R11 knob arms log their knob score and state.

**Test set A is measure-only:**
- Nothing recorded on the official test set A may be used to fit, calibrate, select or design anything. That covers `offline_search_store/queries/*` and every test-A root (R1–R12 test arms, including `r11_knob_*`, `r11_local_k*` and `r12_test_*`).
- **One exception, for Q3 only:** you may run **descriptive** analyses of the already-measured R11 test arms of the frozen R11 periodic, distance, random and off methods (`r11_knob_1..4`, `r11_local_k3/k4`), because describing measured methods is measurement.
  - Keep those analyses in a clearly separated section or file (`TEST_DESCRIPTIVE.md`).
  - Never let them choose a threshold, gain, dose, variant or cell for the new grid.
  - If you are unsure whether a use is descriptive, treat it as forbidden.

**New closed-loop dev data:**
- The coordinator is already preparing a B-dev recording root on the K=10 dev selection, 100 episodes per arm (paired inits across arms of a cell). It covers four cells (π0.5 / GR00T × LIBERO-10 / Spatial, library size 50):
  - periodic at .25 / .32 / .40;
  - R11-frozen distance at .25 / .32 / .40;
  - knob off with **shadow policy chunks** at every look.
  - Look recording is on (keys, distances, weights, guard state; no images or tokens).
  - The location will be announced in `exp/offline_search/rounds/r13/DATA.md`.
- If you need more (other cells, sizes, doses, recalibrated variants, more episodes, shadow on knob arms), write `REQUESTS.md` in your dir **early**, with exact arms, cells, episode counts and why. Note that the dev pool is finite: size 50 has 450 dev inits per suite and size 200 has 300. Plan disjoint calibration and validation splits if you need them.
- You cannot launch anything yourself.

## 6. Compute (for planning the grid)
**Two fleets:**
- local RTX 4090 serving with timan107 workers;
- h100 serving with timan108 workers. h100 is shared: another project currently holds about 67 GB, so its capacity may be reduced.

**Timing:**
- A 500-episode test-A arm takes about 5–20 minutes of episode time plus about 10 minutes of overhead. GR00T L10 is slowest, and calls make it slower.
- About 40–70 arms of 500 episodes per day is realistic across both fleets.
- GR00T cells must keep all arms of a cell on one host.

**Statistics:** at about .85–.95 SR, 500 paired episodes resolve roughly 3–4 pp per arm. Pool across cells and doses, and/or add replicates, for anything finer.

## 7. Deliverables (in `exp/offline_search/rounds/r13/astra/`)
1. **`OFFLINE.md` + code:** library and B-dev analyses, including Q4 and anything that shapes the design.
2. **`SPEC.md`:** implementation-ready specification for sol. It covers:
   - every new or changed layer-4 variant (exact algorithm, fitted scalars, and how they are fitted from library or B-dev only);
   - the frozen test grid: arms, cells, doses, replicates, host pinning;
   - the dev pilot needed first, with its pass/stop rules (e.g. realized IR within ±.02, engagement checks).
   - Reuse `R11Knob` where possible; R11 periodic is the reference periodic.
3. **`PREDICTION.md`**, timestamped before any closed-loop result of the new variants:
   - predicted realized IR per arm;
   - predicted SR differences;
   - which method wins where.
4. **`REQUESTS.md`** (early) for dev data.
5. **Later**, after the test grid runs (the coordinator will resume you):
   - `ANALYSIS.md`;
   - `REPORT.md`, with the owner section first in plain Chinese and no codenames, then the technical part;
   - `HANDBACK.md`.
   - Keep `TEST_DESCRIPTIVE.md` separate.

## 8. Operating rules
- **Writes:** only in `exp/offline_search/rounds/r13/astra/`.
- **No launches:** no server, chain, worker, sync or remote command.
- **Prohibited commands:** no git, no `rm -rf`, no `pkill -f`.
- **CPU:** `taskset -c 10-21,54-65`, `.venv/bin/python`, `PYTHONPATH=.:src`, and single-threaded BLAS unless you deliberately parallelize inside your CPU set.
- **GPU:** the local RTX 4090 may be used for offline computation only, under 20 GB, and only after checking `nvidia-smi`. Closed-loop serving runs on it, so be modest.
- **Every number in your reports** must come from code you ran in your directory.
- **Time:** take what you need (several hours is fine). Hand back the SPEC/PREDICTION/plan as soon as they are frozen and solid, because the coordinator wants the GPUs busy. Write early notes to `STATUS.md` so the coordinator can follow.
