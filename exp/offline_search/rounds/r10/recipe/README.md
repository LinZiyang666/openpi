# R10Recipe

`exp.offline_search.rounds.r10.recipe.recipe:R10Recipe` is the single serving
entry point for all models, suites, and library sizes. It fits the three layers
from the selected library and writes one standard plugin pickle. No existing G
fit, head, distance calibration, recording, or closed-loop result supplies fitting
values. Failed library episodes remain included when selected.

The rule is fixed: R4 BlindAWM with a ten-control anchor-tail commit, kref 5 at
five episodes per task and 8 otherwise; frozen R8 only-no-progress guard with
library LOEO calibration; Stage 2b in-library LOEO motion corrector attenuated
by `.5*clip((2-r)/1.25,0,1)`. Step 0 is uncorrected. There is no escalation.
Gripper, padding, and controls after the first ten are unchanged by correction.

The pickle embeds PCA, metrics, guard tables, ten task heads, distance scale,
episode selection, and fit provenance. Its only external numerical payload is
the parent library's action array through R10 `IndexedRows`. Returned row IDs
always refer to that parent library, including blind and policy-tail results.
An artifact can be served without raw keys, PCA caches, subset specs, G fits,
head files, or calibration JSON files.

All Python commands below use this prefix:

```sh
taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python
```

Append one of these commands to the prefix to build a single artifact:

```sh
-m exp.offline_search.rounds.r10.recipe.build fit --model pi05 --suite l10 --library current --output /tmp/r10_current.pkl
-m exp.offline_search.rounds.r10.recipe.build fit --model pi05 --suite spatial --library bpool_cs --output /tmp/r10_bpool.pkl
-m exp.offline_search.rounds.r10.recipe.build fit --model groot --suite l10 --r10-size 200 --output /tmp/r10_subset.pkl
-m exp.offline_search.rounds.r10.recipe.build fit --model pi05 --suite l10 --subset /path/to/episodes.json --output /tmp/r10_episodes.pkl
```

`--subset` accepts a JSON object with `library` and `episode_ids_by_task`, or the
existing R10 subset JSON format with `parent` instead of `library`:

```json
{
  "model": "pi05",
  "suite": "l10",
  "library": "bpool_cs",
  "episode_ids_by_task": {
    "0": [38, 7, 45, 18, 17]
  }
}
```

The abbreviated example must be extended to all task IDs 0–9. Each task needs
at least three distinct whole episodes of that task. The builder validates the
cell and, when provided, the parent path and episode metadata hash. It inlines
the episode IDs into the artifact constructor metadata. Their order determines
five-fold distance calibration (`position % 5`), matching frozen R10 subsets.
Full libraries use a deterministic per-task PCG64 permutation, seed
`20261002 + task_id`. Rows remain sorted in parent order. Existing output files
are refused by `fit`; choose a new output path to refit.

PCA uses the unchanged randomized SVD, fitted afresh on the selected rows.
The head uses exact deployed LOEO retrieval and synthesis with `prev_hit=True`,
except the cache's step-0 branch. Targets are stored policy chunks minus the
synthesized cache chunk. Features are exactly 217 float32 values; task heads
use 384 RFF, seed 0, ridge alpha 100, and equal episode weights whose total
mass is the number of selected anchor rows. No diagnostic CV selects parameters.

The distance scale reproduces astra's final calibration: five episode-held-out
folds, fixed selected-library PCA, training-fold action sigma and metric,
same-task nearest geometric distances, and the episode-balanced median of
non-step-zero held-out distances. This is the frozen GC_dist definition;
the fixed full-library PCA remains part of that definition.

Prepare all deliverables, then validate and plan locally:

```sh
-m exp.offline_search.rounds.r10.recipe.build prepare --workers 8
-m exp.offline_search.rounds.r10.recipe.replay all --workers 6
-m exp.offline_search.rounds.r10.recipe.tests
-m exp.offline_search.rounds.r10.recipe.finalize finalize
```

`prepare` fits 24 R10 subsets under `recipe/artifacts/` and four deployed current
libraries under `/home/weiland/trace_runs/os_closed_loop/r10_recipe_current/fits/`.
The stored pi05 Spatial current library has 49 episodes, including four on task
8, so the unchanged kref rule gives it 8. The other three current libraries
have five episodes per task (50 total) and use kref 5. The protocol records these
actual counts. No episode is fabricated or imported to fill the missing slot.
It emits four standard full-model, cost-ledger arms in `r10_recipe_current`.
Every arm uses the byte-identical official A 500-pair manifest copied from
`r10_size_*/eval500.json`, standard LIBERO init files, replan_steps 5, blind
serving, one committed policy-tail block, and the guard-only plugin judge.

`replay` verifies against every available frozen `r10_corr3_{pi05,groot}(_b)`
fit for each of the 24 cell-sizes. The real plugin, orchestrator, storage,
connection cloning, and history lifecycle run on B-library inputs. Each arm
has 360 decisions over all ten tasks, two simultaneous connections, three
episode resets per task, static and changing states, and 60 forced no-progress
misses. Comparison is exact: full float32 action bytes, retrieval and synthesis
rows/weights, confidence, flags, extras, and every decision field except runtime
identity and timing. Test-only hooks are installed after loading, never saved in
production fits. Frozen artifacts enter only this verification process.

Review `equivalence.json`, `current_selftests.json`, `fitted_state_audit.json`,
`artifact_audit.json`, `relocation_audit.json`, and `final_audit.json`.
`deployment.json` gives exact bytes from the standard current-root control plan.
`H100_SOURCES.sha256` lists the complete runtime source closure, including frozen
dependencies; `ALL_SOURCES.sha256` lists new builder and verification sources.
`protected_inputs.json` binds preserved R10 sources, subsets, heads, G fits,
calibrations, and corr3 fits. No sync, server, client, or evaluation is launched.
