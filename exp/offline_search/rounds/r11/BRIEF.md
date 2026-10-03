# R11 — Layer 4: the IR knob (convert inference into success efficiently)

Coordinator brief, 2026-10-02 20:3x CDT. Shared by the two explorers (opus, astra). Read fully.

## 1. Where layer 4 sits
The deployment design (owner, 2026-10-02) has five layers:
1. **Cache** — R4 BlindAWM retrieval synthesis, ten-control anchor-tail commit, kref 5 at five episodes per task, otherwise 8.
2. **Only-no-progress guard** — frozen R8 onlynp judge; thresholds calibrated in-library by LOEO pseudo-queries; calls the policy when retrieved-demo progress stalls.
3. **In-library LOEO motion corrector** — distance-attenuated strength `.5*clip((2-r)/1.25,0,1)`; this is R10 GC_dist.
4. **This round: the IR knob.**
5. **Per-task partition** — per-task retrieval scope, metric and corrector heads. Layers 1–3 already use it. **Layer 4 must not add anything task-indexed.** No per-task thresholds, rates or schedules. Knob parameters are scalars per cell-size (model × suite × library).

Layers 1–3 (+5) are implemented as `exp.offline_search.rounds.r10.recipe.recipe:R10Recipe`. Its builder fits everything from one library; see `rounds/r10/recipe/README.md`. sol is finishing it now.

## 2. Owner's definition of layer 4 (paraphrased faithfully)
- **Purpose.** When the cache is not strong enough, push SR as close to pure inference as possible while keeping IR as low as possible. Extra inference has to be spent anyway, so the problem is how to convert IR into SR **efficiently**.
- **The knob.** A single setting that trades IR for SR. Earlier rounds already had an IR knob: the probability of a random miss.
  - The random knob is worth keeping, because it ties realized IR to theoretical IR.
  - Every candidate knob's mapping knob → IR must be **constructed offline by LOEO on the library itself**.
- **Owner's candidate methods:**
  1. Random miss: each decision calls the policy with probability ρ.
  2. Periodic miss: call the policy at regular intervals along the episode.
  3. Distance-threshold miss: call the policy when the query is far from the library.
  - Other methods are welcome.

## 3. Facts you need
**IR (owner's definition).**
- Formulas: π0.5 IR = .152·v + .848·m; GR00T IR = .148·v + .852·m. Normalization: pure policy with a ten-control commit (P10) has IR = .5.
- Per ten-control decision, a cache look costs ≈ .076 (π0.5) / .074 (GR00T); a miss (policy call) costs ≈ .5.
- Details: memory `project_inference_ratio_definition`, the cost ledger in `exp/offline_search/closed_loop/plugin.py`, and R6 ANALYSIS §5 (exact cadence DAG IR model).

**R10 aggregates** (official test set A, 500 episodes per arm; `rounds/r10/REPORT.md`). 3-layer = GC_dist column. The weak-cache gap is where layer 4 should earn:

| cell | 3-layer | pure policy |
|---|---|---|
| π0.5 L10-50 | .826 @ .168 | .908 @ .5 |
| GR00T L10-50 | .754 @ .207 | .898 @ .5 |
| π0.5 Sp-50 | .910 @ .135 | .988 |
| GR00T Sp-50 | .896 @ .127 | .940 |

- At ≥ 200 episodes (long tasks) and ≥ 100 (short tasks), the 3-layer design is already near pure policy, e.g. L10-500 .894 @ .150 and .906 @ .175.
- Pure cache costs ≈ .075.

**R6 prior exploration** (`rounds/r06/ANALYSIS.md` §5, `rounds/r06/ideation_Q2/` incl. `frontier/adapters/methods.py` RiskLottery / uniform / CycleTail, `rounds/r06/ideation_Q1/` LOEO residual R):
- Knob ρ = target owner IR, calibrated from the library plus ≤ 10 non-test B-val recordings per cell. Realized IR landed within ±.02 of target in 26/28 arms.
- Placing calls by LOEO-predicted disagreement (R) did **no better than uniform** at matched ρ (R − U ≈ 0 pp pooled).
- Extra calls helped sparse cells. They hurt the dense GR00T Sp-500 cell, below pure cache: there was no "do not spend" gate.
- Pure-L10 SR was not reached at IR ≤ .45 on sparse cells.
- The base then was different: cache + R6 guards, no corrector. Treat R6 as a prior, not as an answer.

## 4. Hard data rules
- **Test set A is measurement only.** Test set A = the official LIBERO inits 0–49 × tasks 0–9, i.e. every closed-loop eval in R1–R10.
  - You must NOT read any client / journal / summary / server log / episode-level output under `/home/weiland/trace_runs/os_closed_loop/` for any round.
  - You MAY read `arms.json`, `eval500.json`, `fits/` and `config/` of the `r10_*` roots, to reuse fitted artifacts.
  - You MAY read REPORT / ANALYSIS markdown files, which hold aggregates only.
- **All fitting, calibration, thresholds and selection come from B-pool libraries:**
  - `/home/weiland/trace_runs/offline_search_store/library/<model>_<suite>/{bpool_cs (π0.5 500), bpool_all (GR00T 500), current (curated 50)}`;
  - the R10 nested subsets (`rounds/r10/subsets/`, loaded via `rounds/r10/data.py` `SubsetLibrary`).
  - R6's non-test B-val recordings (`/home/weiland/trace_runs/os_closed_loop/r06_c_cal/`) are allowed, with disclosure. This is the only os_closed_loop data you may read beyond §4's list.
- Library rows are pure-policy rollouts on B-pool inits, including failed episodes. Each row's recorded action chunk is the policy's output.
- **Whole-episode-out for distances.** A new episode's queries are 1.4–2.7× farther from the library than in-library LOEO queries (R10 opus finding). Any distance-based calibration must therefore hold out whole episodes, as GC_dist's scale does.

## 5. Scope and grid (proposal; you may argue)
- **Cell-sizes:** R10 nested subsets — L10 at 50 / 200 / 500 and Spatial at 50, both models (8 cell-sizes). Optionally the curated `current` 50 libraries.
- **Knob levels:** expressed as **target realized owner IR**, ≤ 3 levels per method per cell-size, for example .25 / .32 / .40 on weak cells and fewer on strong ones.
- **Final evaluation:** the coordinator measures every frozen arm once on test set A (500 episodes per arm), with a same-batch 3-layer base (knob off). Total budget ≈ 130 arms.
- **Optional dev pilot:** before the final sweep, the coordinator may run a small pilot on **B-pool inits that are not in the evaluated subset library** (non-test), to check realized-vs-target IR.

## 6. Ownership split (do not duplicate)
- **opus — schedule-type knobs and the IR accounting model:**
  - (1) random miss and (2) periodic miss;
  - other schedule variants you find worth it: policy tail length after a miss, refractory interaction with guard calls, front- or back-loaded schedules over episode progress, and so on;
  - the shared offline model knob → theoretical owner IR for the 3-layer base: looks + guard calls + knob misses + their overlap. It must be computable from the library only; state honestly what cannot be estimated offline (e.g. the closed-loop guard call rate) and how you bound or estimate it.
- **astra — state-dependent knobs:**
  - (3) distance-threshold miss;
  - any other signal-triggered placement: predicted policy-vs-cache error magnitude trained by LOEO, neighbour disagreement, imminent gripper transitions in the retrieved chunk, an online adaptive threshold that tracks a target miss rate, …;
  - hybrids with a random floor, which keeps the random knob's IR link.
  - Reuse or cross-check opus's IR model when it appears in `rounds/r11/opus/`; until then, build your own minimal one.
- Each of you may compute the other's methods **only as offline baselines** for comparison.

## 7. Deliverables (each in your own dir `exp/offline_search/rounds/r11/<you>/`)
1. **SPEC.md** — implementation-ready for sol: exact algorithm and parameter formulas. Cover:
   - what triggers a miss;
   - what a miss executes: one policy chunk then back to cache? a tail?;
   - corrector / guard state handling after a miss;
   - the interaction with guard calls;
   - the exact offline calibration procedure from a library to the knob setting for a target IR, with reference code paths in your dir.
2. **OFFLINE.md + code** — the LOEO knob curves (knob value → predicted miss fraction → predicted owner IR) per cell-size. Also offline value evidence: at matched miss fraction, how much LOEO policy-vs-cache error (or another stated risk proxy) the misses capture, compared with random. Remember R6: offline disagreement capture did not translate into SR, so say how much weight this evidence deserves.
3. **Proposed arm grid** for your methods — cell-sizes × target IR levels — with a one-line rationale per method.
4. **PREDICTION.md**, timestamped before any closed-loop result: predicted realized IR per arm, and predicted SR direction / size vs the 3-layer base.
5. **REPORT.md** — an owner section first in plain Chinese with no codenames, then a technical section.
6. **HANDBACK.md** — what sol must implement, files, open questions.

## 8. Operating rules
- **Writes:** only into your own dir. Do not modify any other file, including sol's `rounds/r10/*`, `rounds/r10/recipe/*`, `closed_loop/*` and run roots.
- **No launches:** no h100 / timan sync / chain / server / worker, no GPU.
- **Prohibited commands:** no git, no `rm -rf`, no `pkill -f`.
- **You are explorers, not production coders.** Write analysis / simulation code in your dir. sol implements the frozen SPECs into R10Recipe and emits arms.
- **Python:** `.venv/bin/python`, `PYTHONPATH=.:src`, `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=`. CPU sets:
  - astra: `taskset -c 10-21,54-65`;
  - opus: `taskset -c 2-9,46-53`;
  - CPUs 22–37 / 66–81 belong to sol; 38–43 / 82–87 belong to another project.
- **Timebox:** aim to hand back within ~3 hours. If a method looks useless offline, say so and stop spending on it.
