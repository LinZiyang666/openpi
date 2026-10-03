# R10: cache size × in-library corrector

Owner brief: 2026-10-02. All work is local CPU code/artifact preparation. No sync,
chain, server, worker, h100/timan launch, git command, or remote operation is run.
Python is `.venv/bin/python`, `PYTHONPATH=.:src`; all work is pinned to
`22-37,66-81`, with OMP/OPENBLAS/MKL threads set to 1.

## Data boundary and fixed design

Fit inputs are the B-pool parent libraries under
`/home/weiland/trace_runs/offline_search_store/library/{model}_{suite}/`:
pi05 `bpool_cs`, GR00T `bpool_all`, for l10 and spatial. Failed episodes are kept.
`data.assert_fit_input` resolves symlinks and refuses every path under
`/home/weiland/trace_runs/os_closed_loop` for fitting, including R10 run outputs,
and refuses R8-derived inputs. Raw fit arrays must be under the library directory.
No R1–R9 fit artifact or recording is opened. No B-val recordings are needed or used.
Guard calibration is the original R8/R6 prefit path, using library LOEO pseudoqueries.

Three closed-loop files are read only as explicitly requested metadata:
`r08_main/arms.json` (A constructor settings), `r08_abl/arms.json` (G settings), and
`r09_recipe_full_p/eval500.json` (evaluation pair identities). They never supply
fit/calibration values. A is exactly tasks 0–9 × official inits 0–49, 500 pairs.
The standard control runner uses LIBERO `init_files`; there is no init-state override.
No escalation is present.

Sizes are 50/100/200/300/400/500 episodes, 5/10/20/30/40/50 per task. Each task's
sorted 50 parent episode IDs is permuted by NumPy PCG64 with seed `20261002 + task_id`.
Each size takes a prefix of that same permutation. `subsets/*.json` records episode
IDs, full permutations, failed-episode counts and hashes; `subsets/*.npy` records
sorted parent row IDs. The 500 subset is asserted to contain every parent row.
The fixed rule is kref=5 at size 50, kref=8 for sizes 100–500.

## Stage 1

Status: **COMPLETE**, locally prepared and reviewable; evaluation is not launched.
48 fits and standard emitted arms, 48 passing CPU plugin selftests (2,304 decisions,
96 forced no-progress guard triggers), and both standard `control plan` commands pass.
`stage1_deployment.json` records 1,709,507,834 new run/mirror dependency bytes
(1.592 GiB), **0 extra h100 store bytes**, and 25,062,122,446 bytes in the union of
the full dependency plans, including existing shared baseline/store assets.
`STAGE1_SOURCES.sha256` lists all nine new Python sources; artifact hashes are under
each root's `validation/artifact_shas.json`. The ten semantic unit tests pass in `tests.log`.

Run roots:

- `/home/weiland/trace_runs/os_closed_loop/r10_size_pi05`
- `/home/weiland/trace_runs/os_closed_loop/r10_size_groot`

Each root contains 24 arms: two suites × six sizes × A/G. `arms_in.json` feeds the
standard `closed_loop.ops.emit_arms`; `arms.json`, server YAMLs and client matrices
are standard emitter products. Every arm explicitly selects `eval500.json`.

Storage decision, after inspecting `harness/store.py`, `closed_loop/plugin.py`,
R4 `BlindAWM`, R8 judge/prefit, and `ops/h100/assets.py`: use indexed parent arrays.
The store has no row-view protocol, but the fitted method can implement one.
`SubsetLibrary` presents compact local rows during fitting and remaps prev/next.
The adapter routes candidate and fit library names to the same subset. PCA uses
the unchanged R4 randomized SVD recipe, fitted separately to each subset; only
fresh R10 PCA results are shared between A/G. The metric, sigma, confidence and
guard tables are fitted on that subset. Sigma is std of subset policy chunks
`[:, :5, :7]`, following R9's action-normalization recipe, with a 1e-6 floor.

Serving `IndexedRows` pickles only the parent action filename and row indices.
The fitted cache/judge keeps compact internal IDs; the outer adapter maps returned
vision/blind IDs to parent IDs and declares the parent library. The plugin already
opens parent payload tables. **Exact extra h100 store bytes: 0.** No subset key,
action, task, state, episode, PCA, or token arrays are added to the h100 store.
Local R10 PCA files are fit outputs and are not serving dependencies. Fit artifacts,
configs and manifests consume separate run/mirror space, measured in
`stage1_deployment.json` and the per-root standard `h100_sync/plan.json`.

`selftest.py` runs the real installed plugin, storage facade, orchestrator and
connection lifecycle on recorded B-library keys/state/chunks. Per arm: 48 decisions,
two simultaneous connections, four reset episodes. G uses a test-only forced
no-progress span at decision 2; it must produce flags=8, a real MISS, and policy-tail
reuse. Production artifacts contain no forced hook. A must produce blind cache hits.
Reports live under each root's `selftest/<arm>/selftest_report.json`.

Local reproduction (never launches evaluation):

```sh
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r10.build subsets
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r10.build stage1 --workers 12
```

## Stage 2

Status: **COMPLETE**, locally prepared and reviewable; evaluation is not launched.
48 GC fits and standard emitted arms, 48 passing CPU plugin selftests (2,304 decisions,
192 forced no-progress triggers, with correction checked on each judge synthesis),
and both standard `control plan` commands pass. All 48 head files contain ten task
heads with the required shapes. All ten semantic tests pass in `tests.log`.
`feature_audit.json` verifies byte equality through actual serving correction on 96
recorded B queries across all 24 cell-sizes. `episode_coverage_audit.json` verifies
every selected episode is represented in both methods and every episode belongs to
exactly one held-out fold, for all 240 task heads. Both stages' relocation audits pass.

`offline_cv.csv` and `offline_cv.json` contain all 48 cell-size-method summaries;
`training/<cell>_<size>/cv_{loeo,pair}.json` contains task/fold errors and episode IDs.
The final heads use **252,218 LOEO examples** and **3,880,668 PAIR examples**.
The reported MSE is normalized 10-step, six-motion-channel error; the fixed-cache
qualification below applies to every reported CV value.

`stage2_deployment.json` records **0 extra h100 store bytes** and 3,208,300,045 new
run/mirror dependency bytes (2.988 GiB). This includes explicit G-fit/head dependencies
required by the standard planner, even though the GC serving pickle already embeds
the fitted head. Across both stages, `combined_deployment.json` records
**4,917,807,879 new dependency bytes** (4.580 GiB), plus 57,727 bytes for all nine new
Python source files (14,723 bytes for the three runtime modules). Existing shared
baseline/store assets account for the rest of the full plan sizes. The stated h100
43 GB free / 32 GB reserve leaves room for these incremental artifacts, assuming
the existing standard baseline/store dependencies remain mirrored as described.

Run roots:

- `/home/weiland/trace_runs/os_closed_loop/r10_corr_pi05`
- `/home/weiland/trace_runs/os_closed_loop/r10_corr_groot`

Each root contains 24 arms: two suites × six sizes × GC_loeo/GC_pair. Both use the
same G fit and head class/hyperparameters. Correction is the R9
`RecipeCorrectedBase` applied on the judge's `os_synth` path, with blend .5,
motion channels 0–5 for the first 10 steps; gripper and remaining steps are untouched.
The existing anchor-tail commit semantics are preserved.

For every library row, training treats it as a fresh policy-backed decision:
step 0 uses the cache's early metric; later decisions use `prev_hit=False` and the
previous stored policy chunk for continuity. Exclude all query-episode rows before
computing the fresh distance median, kNN and kernel reference. Same-task stable
top-k, k=16, kref rule and synthesis are the deployed cache's own operations.
LOEO target is stored policy chunk minus this served synthesized chunk.

PAIR donors are other-episode rows of the same task, within radius r; nearest
first, capped at **16 pairs per query**. r is fixed to the 95th percentile of the
library's LOEO 16th-neighbor distances in the same deployed step0/fresh metric.
Input uses the query's keys/state and the donor's cached chunk; target is policy
query chunk minus donor chunk. Both methods give every episode equal total weight;
within an episode, query rows have equal weight before radius-empty omissions,
and each query's pair weight is split over its donors.

Features call the actual R9 serving implementation: PCA keys (128), state (8),
normalized cached 10×7 chunk (70), decision index capped at 120 (1), task one-hot
(10), total 217 float32 inputs. Standardized/clipped inputs plus 384 RFF; ridge
alpha 100; seed 0; ten task heads, each coefficient shape 60×601. Fit normalizer,
RFF and intercept follow the R9 head equation, with float64 batched sufficient
statistics to bound pair memory. No R9 trained head or data is loaded.

`tests.py` checks the fit boundary, indexed serialization, nesting, own-episode
exclusion, exact deployed restricted-cache retrieval/synthesis, byte-identical
features through the actual serving `_correction` path, episode weights,
cross-episode pairs, direct weighted-ridge equation, and per-task head shapes.

Offline reports are three folds by episode. They condition on the fixed deployment
subset's PCA/metric/sigma: **they measure residual-head generalization for a fixed
cache, not end-to-end unseen-library generalization**. Training query and donor
episodes both exclude the held-out fold, and validation retrieves only training-fold
donors. Each fold recalculates its pair radius from its training episodes. Head input
standardization and ridge fitting use training samples only. Error is averaged within
episode then over episodes. Report baseline, corrected blend .5, and full residual
MSE in normalized motion units. CV is diagnostic only; it chooses no size, controller,
radius quantile, cap, head parameter or blend. The final heads use all selected B episodes.

Local reproduction:

```sh
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r10.train all --workers 10
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r10.tests
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r10.build stage2 --workers 8
```

`STAGE1_SOURCES.sha256` / `STAGE2_SOURCES.sha256` list every new source file for h100.
Each run's `validation/artifact_shas.json` hashes its fits. Standard plans include exact
remapped artifact hashes and sizes. These files are preparation artifacts, not evidence
of transfer or closed-loop evaluation. Closed-loop results remain to be obtained by the owner.


## Stage 2b

Coordinator correction: 2026-10-02. This section supersedes the Stage 2 v1
training regime, radius and weight rules above. Stage 1 is accepted and running;
its roots and fits are preserved. The superseded `r10_corr_*`, original `heads/`,
`training/` and `artifacts/` outputs are preserved. The current trainer writes
only `heads2b/` and `training2b/`; GC canonical copies go to `artifacts2b/`.

Status: **COMPLETE**, locally prepared; no evaluation is launched by this work.
All **48 corrected head files** (480 task heads) and 48 new GC fits are ready.
All **14 semantic tests**, **48 CPU plugin selftests** and both standard
`control plan` commands pass. Selftests cover 2,304 decisions, 192 forced
no-progress guard triggers and 1,152 serving-correction checks. Serving features
are byte-identical on 192 B-library inputs across both training variants and
all 24 cell-sizes. The episode/weight audit covers all 240 task datasets: every
selected episode is represented, every episode has one held-out fold, and final
and CV PAIR weight sums equal their anchor counts. All 48 relocated GC artifacts
pass the path/head audit. A hash-only preservation audit verifies 260 protected
Stage 1/v1 artifact and runtime-source files remain unchanged.

Final training totals: **252,218 LOEO rows**; **2,788,317 PAIR samples** from
**230,247 represented anchors**, with total PAIR weight **230,247**. The
remaining 21,971 library rows have no eligible PAIR donor within the fixed radius;
they remain in the library and LOEO training. No episode is removed.

Exact bytes from the standard plans and destination-union audit:

- **Extra h100 store bytes: 0**; no new store arrays.
- Stage 2b run/mirror dependency bytes: **3,208,437,231** (2.988 GiB), including
  unchanged canonical G dependencies required by the standard planner.
- Additional run/mirror destinations beyond Stage 1 plus superseded v1:
  **2,293,321,104** bytes (2.136 GiB). Shared canonical G files are counted once.
- Union of all three stages' complete dependency plans: **30,563,743,595** bytes,
  including existing shared baseline/store assets. The R10 run/mirror additions
  across Stage 1, v1 and 2b total **7,211,128,983** bytes.
- `STAGE2B_SOURCES.sha256` lists ten Python files, **72,187** bytes outside the
  dependency plans; the three serving modules remain unchanged at **14,723** bytes.
  These are required bytes, not evidence that anything was transferred to h100.

At size 500, B-only fixed-cache three-fold CV gives the following relative change
in normalized ten-step motion MSE at blend .5 (negative is lower error):

| Cell | GC_loeo | GC_pair |
|---|---:|---:|
| pi05 l10 | -7.133% | +0.411% |
| pi05 spatial | -10.537% | -2.330% |
| groot l10 | -2.610% | +4.113% |
| groot spatial | -2.862% | +2.386% |

All 48 summaries are in `offline_cv2b.csv/json`; the fixed-cache qualification
below applies to these values. Every requested arm is retained; this diagnostic
selects no arm, size or training parameter.

Run roots:

- `/home/weiland/trace_runs/os_closed_loop/r10_corr2_pi05`
- `/home/weiland/trace_runs/os_closed_loop/r10_corr2_groot`

Each root has 24 GC arms, two suites × six sizes × GC_loeo/GC_pair. Arm names
retain `r10_<model>_<suite>_<size>_GC_{loeo,pair}` inside the new roots. The official
A manifest remains exactly 500 task/init pairs with standard LIBERO `init_files`.
No A recordings or closed-loop results supply fitting inputs. No B-val recording,
escalation, guard refit or evaluation launch is used. G is loaded exclusively from
the preserved B-only canonical `artifacts/r10_<model>_<suite>_<size>_G.pkl`; the
head metadata binds its SHA256. Closed-loop paths still fail `assert_fit_input`.

Both LOEO synthesis and PAIR neighbor selection now use pseudoqueries with
`prev_hit=True`: the deployed regime 2 after hits, including G's committed policy
tail. Step 0 keeps the cache's early metric. Every query excludes all its own
episode's candidate rows before retrieval, retaining task filtering, k=16,
stable nearest-first ties, kref, kernel and deployed `os_synth` arithmetic.
PAIR radius is the **per-task median of LOEO 16th-neighbor distance**, calibrated
from that selected B library in the same metric. PAIR keeps at most 16 nearest
cross-episode donors within that radius. Each CV fold independently recalculates
its task radius using only training-fold query and donor episodes.

Every anchor's pairs divide one anchor-row weight. Each represented episode has
equal total weight, and each represented anchor within an episode has equal
weight. The final total weight is **the number of represented anchor rows**, for
both LOEO and PAIR. `fit_head` preserves that supplied mass; it never renormalizes
to pair count. Radius-empty anchors have no eligible pair and contribute no PAIR
sample or mass. Task/fold reports explicitly record anchor counts and weight sums.

Head class and other settings are unchanged: R9 `RecipeCorrectedBase` on the
judge `os_synth` path; exact 217 float32 serving features; 384 RFF, seed 0,
alpha 100, ten task heads (60×601 coefficients), blend .5, six motion channels
for the first ten steps, and the same G fit/PCA/metric/sigma. No parameter or
library-size selection uses the diagnostic CV. The three-fold episode CV still
conditions on the fixed deployment-cache PCA/metric/sigma; it does not estimate
end-to-end unseen-library generalization.

Evidence files for this revision:

- `offline_cv2b.csv/json` and `training2b/<cell>_<size>/cv_{loeo,pair}.json`
- `tests2b.log`, `stage2b_feature_audit.json`, `stage2b_episode_weight_audit.json`
- `stage2b_preservation_audit.json`, `stage2b_relocation_audit.json`
- `stage2b_deployment.json`, `stage2b_incremental_deployment.json`
- `STAGE2B_SOURCES.sha256` and each new root's `validation/artifact_shas.json`

Local reproduction (CPU preparation only):

```sh
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r10.train all --workers 10
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r10.tests
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r10.build stage2b --workers 8
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r10.report 2b
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r10.audit2b
```

## R10Recipe

Coordinator follow-up: one library-fitted, three-layer deployment design for
every cell and library size. Entry point:
`exp.offline_search.rounds.r10.recipe.recipe:R10Recipe`.
Builder and usage: [recipe/README.md](recipe/README.md).

Status: **COMPLETE**; 28 artifacts, four emitted current arms, 19 semantic
tests, all equivalence/current CPU replays, standard control plan, and relocated
artifact audits pass. Exact new current-root deployment dependencies:
**118,304,675 bytes** (including **118,297,119 fitted-artifact bytes**),
**0 extra h100 store bytes**. The full standard dependency plan contains
**23,470,919,287 bytes in 78 files**, including existing shared assets. No
external G-fit, head, or calibration dependency is added. The standard plan is
`r10_recipe_current/h100_sync/plan.json`; current artifact hashes are in that
root's `validation/artifact_shas.json`.

The recipe fits fresh selected-library PCA and R4 BlindAWM metrics, the frozen
R8 only-no-progress judge through its original library LOEO pseudoquery path,
and ten Stage 2b LOEO motion heads. Correction uses astra's exact GC_dist
arithmetic: strength `.5*clip((2-r)/1.25,0,1)`, step 0 uncorrected, geometric
distance to retrieved rows, and an episode-balanced five-fold held-out metric
distance scale with fixed selected-library PCA. The kref rule is 5 when every
task has five episodes, 8 otherwise. No escalation, parameter selection, A
recording, B-val recording, previous G fit, head file, or calibration file is
used for fitting. Failed selected library episodes remain included.

The single plugin pickle embeds all learned cache, guard, head, and distance
state. Only the parent action payload remains external, through `IndexedRows`;
vision, blind, and policy-tail provenance is mapped to parent row IDs. There
are no `source_fit`, `head_path`, or `calibration_path` constructor arguments.
Serving imports no exploration module. Full libraries and custom whole-episode
subset JSONs are supported; existing R10 subset JSONs and `--r10-size` are
supported directly. Actual fit-file opens are audited and reject query,
derived, other-library, and closed-loop inputs, including symlinks.

**Equivalence passes for all 24 R10 cell-sizes:** 32 available frozen
`r10_corr3_{pi05,groot}(_b)` fits, **11,520 compared decisions, 0 differing
decisions**, including **1,920 forced no-progress triggers**. Each recipe is
independently fitted from raw selected library rows. Cache arrays, guard
calibration tables, head arrays, and distance scales are byte-identical to the
corresponding frozen fit. Real CPU plugin replays cover all ten tasks, two
simultaneous connections, 30 episodes per arm, static and changing B-library
inputs, blind cache continuation, policy-tail reuse, and exact full action
bytes, scores, confidence, rows, weights, flags, and extras. Only runtime
identity and timing fields are excluded from decision comparison. Forced
hooks are test-only and never serialized in the deployment artifacts.

All four current-library artifacts and arms are prepared in:
`/home/weiland/trace_runs/os_closed_loop/r10_recipe_current`.
Arm names are `r10_recipe_{pi05,groot}_{l10,spatial}_current`.
Their four CPU plugin selftests pass: **1,440 decisions, 240 forced triggers,
720 serving-correction checks**. The standard emitted arms use full models,
cost ledgers, replan_steps 5, one policy-tail block, and guard-only verdicts.
The A manifest is byte-identical to both `r10_size_*/eval500.json` files:
SHA256 `2c3b0477794c04b9f9925100346f0d0adbca4dcfe42f3c2eb26278f18b4d9a29`.
Standard LIBERO init files are used. No evaluation or transfer is launched.

**Stored-library discrepancy:** pi05 Spatial `current` actually contains
**49 episodes**, including four on task 8; its own manifest also reports 49.
It is fitted as stored with **kref 8**, following the unchanged rule. The other
three current libraries contain 50 episodes, five per task, and use kref 5.
No missing episode is fabricated or borrowed. The emitted protocol records
all per-task counts. The five-fold calibration skips task/fold combinations
without held-out episodes; it keeps the same metric and balanced-median rule.

Evidence and deployment files are under `recipe/`:

- `equivalence.json`, `equivalence/*.json`, and `fitted_state_audit.json`
- `current_selftests.json`, `artifact_audit.json`, and `relocation_audit.json`
- `tests.log` (five recipe tests) and `training_tests.log` (14 R10 semantic tests)
- `final_audit.json`, `deployment.json`, and `protected_inputs.json`
- `H100_SOURCES.sha256`, `ALL_SOURCES.sha256`, and `source_list.json`
- `DOCUMENTS.sha256` binds this handback and the recipe README

Local reproduction, with no evaluation or transfer:

```sh
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m exp.offline_search.rounds.r10.recipe.build fit --model pi05 --suite l10 --library current --output /tmp/r10_recipe_current.pkl
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m exp.offline_search.rounds.r10.recipe.build prepare --workers 8
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m exp.offline_search.rounds.r10.recipe.replay all --workers 6
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m exp.offline_search.rounds.r10.recipe.finalize finalize
```
